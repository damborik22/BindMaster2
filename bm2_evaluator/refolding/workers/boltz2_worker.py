"""Boltz2 refolding worker — runs INSIDE the Mosaic/Boltz2 environment.

This script is invoked via subprocess from the evaluator env:
    conda run --no-banner -p /path/to/venv python -m \
        bm2_evaluator.refolding.workers.boltz2_worker \
        --fasta input.fasta --out_dir /path/to/output

It uses the Mosaic/Boltz2 JAX API directly and produces standardized
output files (pae.npy, metrics.json, structure.pdb).

Output contract:
    out_dir/
    +-- pae.npy         PAE matrix (N x N), chain order = binder_first
    +-- structure.pdb   Predicted structure
    +-- metrics.json    Standardized metrics (WorkerOutput format)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [boltz2_worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_fasta(fasta_path: Path) -> dict[str, str]:
    """Parse a simple FASTA file into {header: sequence}."""
    sequences = {}
    current_header = ""
    current_seq: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_seq:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)

    if current_header and current_seq:
        sequences[current_header] = "".join(current_seq)

    return sequences


def _nan_safe(val):
    """Replace NaN/inf with None for JSON serialization."""
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    return val


def run_prediction(
    binder_seq: str,
    target_seq: str,
    out_dir: Path,
    recycling_steps: int = 3,
    num_samples: int = 1,
) -> dict:
    """Run Boltz2 prediction using Mosaic API.

    Chain ordering: binder FIRST, target SECOND (Mosaic convention).
    This is recorded in the output so the evaluator can slice correctly.
    """
    # These imports only work inside the Mosaic/Boltz2 environment
    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain

    logger.info(
        f"Initializing Boltz2 (binder={len(binder_seq)}aa, "
        f"target={len(target_seq)}aa)"
    )

    folder = Boltz2()

    # Construct features: binder first, target second (Mosaic convention)
    binder_chain = TargetChain(sequence=binder_seq, use_msa=False)
    target_chain = TargetChain(sequence=target_seq, use_msa=True)

    features, structure = folder.binder_features(
        binder_length=len(binder_seq),
        chains=[binder_chain, target_chain],
    )

    logger.info("Running prediction...")
    import jax

    key = jax.random.PRNGKey(42)

    prediction = folder.predict(
        features=features,
        recycling_steps=recycling_steps,
        key=key,
    )

    L_b = len(binder_seq)
    L_t = len(target_seq)

    # Extract PAE matrix (binder-first ordering)
    pae_matrix = np.array(prediction.pae)
    assert pae_matrix.shape == (
        L_b + L_t,
        L_b + L_t,
    ), f"PAE shape mismatch: {pae_matrix.shape} != ({L_b + L_t}, {L_b + L_t})"

    # Extract pLDDT (0-1 scale for Boltz2)
    plddt = np.array(prediction.plddt)
    plddt_binder = plddt[:L_b]
    plddt_target = plddt[L_b:]

    # Extract confidence metrics
    iptm = float(prediction.iptm)
    ptm = float(getattr(prediction, "ptm", 0.0))

    # Save PAE matrix as .npy (standardized format)
    pae_path = out_dir / "pae.npy"
    np.save(pae_path, pae_matrix)
    logger.info(f"PAE matrix saved: {pae_path} shape={pae_matrix.shape}")

    # Save structure as PDB
    structure_path = out_dir / "structure.pdb"
    try:
        structure.save(str(structure_path))
    except Exception as e:
        logger.warning(f"Failed to save via structure.save(): {e}")
        # Fallback: try prediction.save or write_pdb
        if hasattr(prediction, "save"):
            prediction.save(str(structure_path))

    # Extract interchain PAE means
    pae_bt = pae_matrix[:L_b, L_b:]  # binder->target
    pae_tb = pae_matrix[L_b:, :L_b]  # target->binder

    metrics = {
        "engine": "boltz2",
        "chain_order": "binder_first",
        "target_length": L_t,
        "binder_length": L_b,
        "iptm": _nan_safe(iptm),
        "ptm": _nan_safe(ptm),
        "plddt_binder_mean": _nan_safe(float(np.mean(plddt_binder))),
        "plddt_binder_min": _nan_safe(float(np.min(plddt_binder))),
        "plddt_target_mean": _nan_safe(float(np.mean(plddt_target))),
        "plddt_complex_mean": _nan_safe(float(np.mean(plddt))),
        "plddt_scale_max": 1.0,
        "pae_matrix_file": "pae.npy",
        "structure_file": "structure.pdb",
        "pae_bt_mean": _nan_safe(float(np.mean(pae_bt))),
        "pae_tb_mean": _nan_safe(float(np.mean(pae_tb))),
        "pae_interaction_mean": _nan_safe(
            float((np.mean(pae_bt) + np.mean(pae_tb)) / 2.0)
        ),
        "success": True,
        "error": None,
    }

    # Try to extract aux metrics (ipSAE, etc.) if available
    try:
        aux_metrics = _extract_aux_metrics(folder, features, binder_seq, L_b)
        metrics.update(aux_metrics)
    except Exception as e:
        logger.warning(f"Failed to extract aux metrics: {e}")

    return metrics


def _extract_aux_metrics(folder, features, binder_seq, L_b) -> dict:
    """Extract auxiliary metrics from Mosaic loss evaluation.

    These include ipSAE computed by Mosaic (for cross-validation
    against our own ipSAE implementation).
    """
    extra = {}
    try:
        import jax
        import jax.numpy as jnp

        TOKENS = "ACDEFGHIKLMNPQRSTVWY"
        pssm = jax.nn.one_hot(
            jnp.array([TOKENS.index(c) for c in binder_seq]), 20
        )

        from mosaic.losses.structure_prediction import (
            BinderTargetContact,
            WithinBinderContact,
        )

        loss_fn = folder.build_loss(
            loss=BinderTargetContact() + WithinBinderContact(),
            features=features,
        )
        _, aux = loss_fn(pssm)

        # Extract named metrics from aux dict
        for key in ["bt_ipsae", "tb_ipsae", "ipsae_min", "bt_iptm", "binder_ptm"]:
            if key in aux:
                val = float(aux[key])
                if not (np.isnan(val) or np.isinf(val)):
                    extra[f"aux_{key}"] = val

    except Exception as e:
        logger.debug(f"Aux metric extraction skipped: {e}")

    return extra


def run_monomer(
    binder_seq: str,
    out_dir: Path,
    recycling_steps: int = 3,
) -> dict:
    """Refold binder alone for monomer validation."""
    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain

    logger.info(f"Monomer prediction (binder={len(binder_seq)}aa)")

    folder = Boltz2()
    chain = TargetChain(sequence=binder_seq, use_msa=False)

    features, structure = folder.target_only_features(chains=[chain])

    import jax

    key = jax.random.PRNGKey(42)

    prediction = folder.predict(
        features=features,
        recycling_steps=recycling_steps,
        key=key,
    )

    plddt = np.array(prediction.plddt)

    structure_path = out_dir / "monomer.pdb"
    try:
        structure.save(str(structure_path))
    except Exception:
        if hasattr(prediction, "save"):
            prediction.save(str(structure_path))

    return {
        "plddt_mean": _nan_safe(float(np.mean(plddt))),
        "plddt_scale_max": 1.0,
        "structure_file": "monomer.pdb",
        "success": True,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Boltz2 refolding worker (runs inside Mosaic env)"
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        default="complex",
        choices=["complex", "monomer"],
        help="Prediction mode",
    )
    parser.add_argument(
        "--recycling_steps", type=int, default=3, help="Recycling steps"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = _parse_fasta(Path(args.fasta))

    try:
        if args.mode == "complex":
            # Expect 2 sequences: "target" and "binder"
            if "target" in sequences and "binder" in sequences:
                target_seq = sequences["target"]
                binder_seq = sequences["binder"]
            else:
                # Take first as target, second as binder
                keys = list(sequences.keys())
                target_seq = sequences[keys[0]]
                binder_seq = sequences[keys[1]]

            metrics = run_prediction(
                binder_seq=binder_seq,
                target_seq=target_seq,
                out_dir=out_dir,
                recycling_steps=args.recycling_steps,
            )
        else:
            # Monomer mode: single sequence
            binder_seq = list(sequences.values())[0]
            metrics = run_monomer(
                binder_seq=binder_seq,
                out_dir=out_dir,
                recycling_steps=args.recycling_steps,
            )

        # Save metrics.json
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Success. Output at {out_dir}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        error_metrics = {
            "engine": "boltz2",
            "success": False,
            "error": str(e),
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(error_metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
