"""AF2-Multimer refolding engine.

Calls the af2_worker script via subprocess in the binder-eval-af2 env.
Parses standardized output (pae.npy, metrics.json, structure.pdb).

Env: conda env 'binder-eval-af2' (or configurable)

Note: ColabFold CLI (colabfold_batch) is NOT installed on this system.
This engine uses ColabDesign's Python API instead, which requires
a target PDB file (not just a FASTA sequence).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from bm2_evaluator.refolding.base import (
    MonomerResult,
    RefoldingEngine,
    WorkerOutput,
)

logger = logging.getLogger(__name__)

_WORKER_MODULE = "bm2_evaluator.refolding.workers.af2_worker"


class AF2Engine(RefoldingEngine):
    """AF2-Multimer refolding via worker subprocess.

    The worker runs inside the binder-eval-af2 conda environment
    and uses ColabDesign's mk_af_model API. It produces standardized
    output files that this class parses.

    pLDDT: 0-100 scale (AF2 convention).
    PAE: .npy format, target-first ordering.

    Note: AF2 via ColabDesign requires a target PDB file (not just
    sequence). The target_pdb must be set before calling refold_complex.
    """

    def __init__(
        self,
        conda_env: str = "binder-eval-af2",
        target_pdb: Optional[Path] = None,
        target_chain: str = "A",
        timeout: int = 7200,
        num_recycles: int = 3,
        num_models: int = 1,
    ):
        """Initialize AF2 engine.

        Args:
            conda_env: Conda environment name with ColabDesign installed.
            target_pdb: Path to target PDB file (required for predictions).
            target_chain: Target chain ID in the PDB.
            timeout: Subprocess timeout in seconds (default 2 hours).
            num_recycles: Number of recycling iterations.
            num_models: Number of AF2 models to use (1-5).
        """
        self.conda_env = conda_env
        self.target_pdb = target_pdb
        self.target_chain = target_chain
        self.timeout = timeout
        self.num_recycles = num_recycles
        self.num_models = num_models

    @property
    def name(self) -> str:
        return "af2"

    def refold_complex(
        self,
        binder_seq: str,
        target_seq: str,
        output_dir: Path,
    ) -> WorkerOutput:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.target_pdb is None:
            return WorkerOutput(
                engine="af2",
                chain_order="target_first",
                target_length=len(target_seq),
                binder_length=len(binder_seq),
                iptm=0.0,
                ptm=0.0,
                plddt_binder_mean=0.0,
                plddt_binder_min=0.0,
                plddt_target_mean=0.0,
                plddt_complex_mean=0.0,
                plddt_scale_max=100.0,
                pae_matrix_file="pae.npy",
                structure_file="structure.pdb",
                success=False,
                error="target_pdb not set. AF2 via ColabDesign requires a target PDB.",
            )

        cmd = self._build_cmd(
            f"--target_pdb {self.target_pdb} "
            f"--target_chain {self.target_chain} "
            f"--binder_seq {binder_seq} "
            f"--out_dir {output_dir} "
            f"--mode complex "
            f"--num_recycles {self.num_recycles}"
        )

        logger.info(f"Running AF2 worker: {' '.join(cmd[:6])}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            logger.error(f"AF2 worker failed:\n{result.stderr[-1000:]}")
            return WorkerOutput(
                engine="af2",
                chain_order="target_first",
                target_length=len(target_seq),
                binder_length=len(binder_seq),
                iptm=0.0,
                ptm=0.0,
                plddt_binder_mean=0.0,
                plddt_binder_min=0.0,
                plddt_target_mean=0.0,
                plddt_complex_mean=0.0,
                plddt_scale_max=100.0,
                pae_matrix_file="pae.npy",
                structure_file="structure.pdb",
                success=False,
                error=result.stderr[-500:] if result.stderr else "Unknown error",
            )

        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            return WorkerOutput(
                engine="af2",
                chain_order="target_first",
                target_length=len(target_seq),
                binder_length=len(binder_seq),
                iptm=0.0,
                ptm=0.0,
                plddt_binder_mean=0.0,
                plddt_binder_min=0.0,
                plddt_target_mean=0.0,
                plddt_complex_mean=0.0,
                plddt_scale_max=100.0,
                pae_matrix_file="pae.npy",
                structure_file="structure.pdb",
                success=False,
                error="No metrics.json produced by worker",
            )

        return WorkerOutput.from_json(metrics_path)

    def refold_monomer(
        self,
        binder_seq: str,
        output_dir: Path,
    ) -> MonomerResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = output_dir / "input.fasta"
        self._write_fasta({"binder": binder_seq}, fasta_path)

        cmd = self._build_cmd(
            f"--fasta {fasta_path} --out_dir {output_dir} "
            f"--mode monomer --num_recycles {self.num_recycles}"
        )

        logger.info("Running AF2 monomer worker")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            return MonomerResult(
                plddt_mean=0.0,
                plddt_scale_max=100.0,
                structure_path=output_dir / "monomer.pdb",
                success=False,
                error=result.stderr[-500:] if result.stderr else "Unknown error",
            )

        import json

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path) as f:
            data = json.load(f)

        return MonomerResult(
            plddt_mean=float(data.get("plddt_mean", 0.0)),
            plddt_scale_max=float(data.get("plddt_scale_max", 100.0)),
            structure_path=output_dir / data.get("structure_file", "monomer.pdb"),
            success=bool(data.get("success", False)),
            error=data.get("error"),
        )

    def check_available(self) -> bool:
        """Check if the AF2 conda env exists."""
        try:
            result = subprocess.run(
                [
                    "conda",
                    "run",
                    "--no-banner",
                    "-n",
                    self.conda_env,
                    "python",
                    "-c",
                    "import colabdesign; print('ok')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _build_cmd(self, worker_args: str) -> list[str]:
        """Build subprocess command for the worker."""
        parts = [
            "conda",
            "run",
            "--no-banner",
            "-n",
            self.conda_env,
            "python",
            "-m",
            _WORKER_MODULE,
        ]
        parts.extend(worker_args.split())
        return parts
