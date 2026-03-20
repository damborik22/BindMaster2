"""AF2-Multimer refolding worker — runs INSIDE the binder-eval-af2 environment.

This script is invoked via subprocess from the evaluator env:
    conda run --no-banner -n binder-eval-af2 python -m \
        bm2_evaluator.refolding.workers.af2_worker \
        --target_pdb target.pdb --target_chain A \
        --binder_seq MKWAS... --out_dir /path/to/output

It uses ColabDesign's AF2 API directly and produces standardized
output files (pae.npy, metrics.json, structure.pdb).

Output contract:
    out_dir/
    +-- pae.npy         PAE matrix (N x N), chain order = target_first
    +-- structure.pdb   Predicted structure
    +-- metrics.json    Standardized metrics (WorkerOutput format)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [af2_worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _find_af2_data_dir() -> str:
    """Find AlphaFold2 model parameters directory.

    Walks up from common locations looking for params/ directory.
    """
    candidates = [
        os.environ.get("AF2_DATA_DIR", ""),
        os.path.expanduser("~/BindMaster/bindcraft-tools/af2_params"),
        os.path.expanduser("~/BindCraft/params"),
        "/home/david/BindMaster/bindcraft-tools/af2_params",
    ]

    # Also search relative to BindCraft installation
    for base in [
        os.path.expanduser("~/BindMaster"),
        os.path.expanduser("~/BindCraft"),
    ]:
        for root, dirs, files in os.walk(base):
            if "params" in dirs:
                params_dir = os.path.join(root, "params")
                if any(
                    f.startswith("params_model") for f in os.listdir(params_dir)
                ):
                    candidates.append(params_dir)
            if len(candidates) > 10:
                break

    for path in candidates:
        if path and os.path.isdir(path):
            return path

    raise FileNotFoundError(
        "AF2 data directory not found. Set AF2_DATA_DIR environment variable."
    )


def _nan_safe(val):
    """Replace NaN/inf with None for JSON serialization."""
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    return val


def run_prediction(
    binder_seq: str,
    target_pdb: str,
    target_chain: str,
    out_dir: Path,
    num_recycles: int = 3,
    num_models: int = 1,
) -> dict:
    """Run AF2-Multimer prediction using ColabDesign API.

    Chain ordering: target FIRST, binder SECOND (ColabDesign convention).
    This is recorded in the output so the evaluator can slice correctly.
    """
    # These imports only work inside the ColabDesign environment
    from colabdesign.af import mk_af_model

    af2_data_dir = _find_af2_data_dir()
    logger.info(f"AF2 data dir: {af2_data_dir}")

    logger.info(
        f"Initializing AF2-Multimer (binder={len(binder_seq)}aa, "
        f"target_pdb={target_pdb}, chain={target_chain})"
    )

    af = mk_af_model(
        protocol="binder",
        use_multimer=True,
        data_dir=af2_data_dir,
        model_names=[f"model_{i}_multimer_v3" for i in range(1, num_models + 1)],
    )

    af.prep_inputs(
        pdb_filename=target_pdb,
        chain=target_chain,
        binder_len=len(binder_seq),
    )
    af.set_seq(seq=binder_seq)

    logger.info("Running AF2 prediction...")
    af.predict(num_recycles=num_recycles)

    # Extract lengths
    # ColabDesign convention: target first, binder second
    L_b = len(binder_seq)
    total_len = af.aux["plddt"].shape[0] if "plddt" in af.aux else L_b
    L_t = total_len - L_b

    # Extract PAE matrix (target-first ordering)
    pae_matrix = np.array(af.aux["pae"])
    assert pae_matrix.shape[0] == total_len, (
        f"PAE shape mismatch: {pae_matrix.shape[0]} != {total_len}"
    )

    # Extract pLDDT (0-100 scale for AF2)
    plddt = np.array(af.aux["plddt"])
    plddt_target = plddt[:L_t]
    plddt_binder = plddt[L_t:]

    # Extract ipTM
    iptm = 0.0
    for key in ["i_ptm", "iptm"]:
        if key in af.aux:
            iptm = float(af.aux[key])
            break
        if "log" in af.aux and key in af.aux["log"]:
            iptm = float(af.aux["log"][key])
            break

    ptm = float(af.aux.get("ptm", af.aux.get("log", {}).get("ptm", 0.0)))

    # Save PAE matrix as .npy (standardized format)
    pae_path = out_dir / "pae.npy"
    np.save(pae_path, pae_matrix)
    logger.info(f"PAE matrix saved: {pae_path} shape={pae_matrix.shape}")

    # Save structure as PDB
    structure_path = out_dir / "structure.pdb"
    af.save_pdb(str(structure_path))

    # Interchain PAE means
    pae_bt = pae_matrix[L_t:, :L_t]  # binder rows -> target cols
    pae_tb = pae_matrix[:L_t, L_t:]  # target rows -> binder cols

    metrics = {
        "engine": "af2",
        "chain_order": "target_first",
        "target_length": L_t,
        "binder_length": L_b,
        "iptm": _nan_safe(iptm),
        "ptm": _nan_safe(ptm),
        "plddt_binder_mean": _nan_safe(float(np.mean(plddt_binder))),
        "plddt_binder_min": _nan_safe(float(np.min(plddt_binder))),
        "plddt_target_mean": _nan_safe(float(np.mean(plddt_target))),
        "plddt_complex_mean": _nan_safe(float(np.mean(plddt))),
        "plddt_scale_max": 100.0,
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

    return metrics


def run_monomer(
    binder_seq: str,
    out_dir: Path,
    num_recycles: int = 3,
) -> dict:
    """Refold binder alone for monomer validation."""
    from colabdesign.af import mk_af_model

    af2_data_dir = _find_af2_data_dir()
    logger.info(f"Monomer prediction (binder={len(binder_seq)}aa)")

    af = mk_af_model(
        protocol="hallucination",
        use_multimer=False,
        data_dir=af2_data_dir,
    )

    af.prep_inputs(length=len(binder_seq))
    af.set_seq(seq=binder_seq)

    af.predict(num_recycles=num_recycles)

    plddt = np.array(af.aux["plddt"])

    structure_path = out_dir / "monomer.pdb"
    af.save_pdb(str(structure_path))

    return {
        "plddt_mean": _nan_safe(float(np.mean(plddt))),
        "plddt_scale_max": 100.0,
        "structure_file": "monomer.pdb",
        "success": True,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AF2-Multimer refolding worker (runs inside ColabDesign env)"
    )
    parser.add_argument("--target_pdb", help="Target PDB file path")
    parser.add_argument("--target_chain", default="A", help="Target chain ID")
    parser.add_argument("--binder_seq", help="Binder sequence (for complex mode)")
    parser.add_argument("--fasta", help="Input FASTA (for monomer mode)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        default="complex",
        choices=["complex", "monomer"],
        help="Prediction mode",
    )
    parser.add_argument(
        "--num_recycles", type=int, default=3, help="Number of recycles"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "complex":
            if not args.target_pdb or not args.binder_seq:
                parser.error("Complex mode requires --target_pdb and --binder_seq")

            metrics = run_prediction(
                binder_seq=args.binder_seq,
                target_pdb=args.target_pdb,
                target_chain=args.target_chain,
                out_dir=out_dir,
                num_recycles=args.num_recycles,
            )
        else:
            # Monomer mode
            if args.fasta:
                seqs = {}
                with open(args.fasta) as f:
                    header = ""
                    seq_lines: list[str] = []
                    for line in f:
                        line = line.strip()
                        if line.startswith(">"):
                            if header and seq_lines:
                                seqs[header] = "".join(seq_lines)
                            header = line[1:]
                            seq_lines = []
                        elif line:
                            seq_lines.append(line)
                    if header and seq_lines:
                        seqs[header] = "".join(seq_lines)
                binder_seq = list(seqs.values())[0]
            elif args.binder_seq:
                binder_seq = args.binder_seq
            else:
                parser.error("Monomer mode requires --fasta or --binder_seq")

            metrics = run_monomer(
                binder_seq=binder_seq,
                out_dir=out_dir,
                num_recycles=args.num_recycles,
            )

        # Save metrics.json
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Success. Output at {out_dir}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        error_metrics = {
            "engine": "af2",
            "success": False,
            "error": str(e),
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(error_metrics, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
