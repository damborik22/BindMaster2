"""Boltz2 refolding engine.

Calls the boltz2_worker script via subprocess in the Mosaic venv.
Parses standardized output (pae.npy, metrics.json, structure.pdb).

Env: Mosaic venv at /home/david/BindMaster/Mosaic/.venv
     (or configurable via venv_path / conda_env)
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

# Path to the worker module
_WORKER_MODULE = "bm2_evaluator.refolding.workers.boltz2_worker"


class Boltz2Engine(RefoldingEngine):
    """Boltz2 refolding via worker subprocess.

    The worker runs inside the Mosaic/Boltz2 environment and uses
    the Mosaic JAX API (proven on this system). It produces standardized
    output files that this class parses.

    pLDDT: 0-1 scale.
    PAE: .npy format, binder-first ordering.
    """

    def __init__(
        self,
        venv_path: Optional[str] = None,
        conda_env: Optional[str] = None,
        timeout: int = 3600,
        recycling_steps: int = 3,
    ):
        """Initialize Boltz2 engine.

        Args:
            venv_path: Path to Mosaic venv (uses -p flag with conda run).
                       Default: ~/BindMaster/Mosaic/.venv
            conda_env: Named conda env (uses -n flag). Mutually exclusive
                       with venv_path.
            timeout: Subprocess timeout in seconds (default 1 hour).
            recycling_steps: Number of recycling iterations.
        """
        if venv_path is None and conda_env is None:
            venv_path = str(
                Path.home() / "BindMaster" / "Mosaic" / ".venv"
            )
        self.venv_path = venv_path
        self.conda_env = conda_env
        self.timeout = timeout
        self.recycling_steps = recycling_steps

    @property
    def name(self) -> str:
        return "boltz2"

    def refold_complex(
        self,
        binder_seq: str,
        target_seq: str,
        output_dir: Path,
    ) -> WorkerOutput:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write input FASTA with target and binder labels
        fasta_path = output_dir / "input.fasta"
        self._write_fasta(
            {"target": target_seq, "binder": binder_seq}, fasta_path
        )

        # Build subprocess command
        cmd = self._build_cmd(
            f"--fasta {fasta_path} --out_dir {output_dir} "
            f"--mode complex --recycling_steps {self.recycling_steps}"
        )

        logger.info(f"Running Boltz2 worker: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            logger.error(f"Boltz2 worker failed:\n{result.stderr[-1000:]}")
            return WorkerOutput(
                engine="boltz2",
                chain_order="binder_first",
                target_length=len(target_seq),
                binder_length=len(binder_seq),
                iptm=0.0,
                ptm=0.0,
                plddt_binder_mean=0.0,
                plddt_binder_min=0.0,
                plddt_target_mean=0.0,
                plddt_complex_mean=0.0,
                plddt_scale_max=1.0,
                pae_matrix_file="pae.npy",
                structure_file="structure.pdb",
                success=False,
                error=result.stderr[-500:] if result.stderr else "Unknown error",
            )

        # Parse standardized output
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            return WorkerOutput(
                engine="boltz2",
                chain_order="binder_first",
                target_length=len(target_seq),
                binder_length=len(binder_seq),
                iptm=0.0,
                ptm=0.0,
                plddt_binder_mean=0.0,
                plddt_binder_min=0.0,
                plddt_target_mean=0.0,
                plddt_complex_mean=0.0,
                plddt_scale_max=1.0,
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
            f"--mode monomer --recycling_steps {self.recycling_steps}"
        )

        logger.info(f"Running Boltz2 monomer worker")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            return MonomerResult(
                plddt_mean=0.0,
                plddt_scale_max=1.0,
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
            plddt_scale_max=float(data.get("plddt_scale_max", 1.0)),
            structure_path=output_dir / data.get("structure_file", "monomer.pdb"),
            success=bool(data.get("success", False)),
            error=data.get("error"),
        )

    def check_available(self) -> bool:
        """Check if Mosaic venv exists and has Boltz2."""
        if self.venv_path:
            python = Path(self.venv_path) / "bin" / "python"
            return python.exists()
        # For named conda env, try running a test
        try:
            cmd = self._build_cmd("--help")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _build_cmd(self, worker_args: str) -> list[str]:
        """Build the subprocess command to invoke the worker."""
        if self.venv_path:
            python = str(Path(self.venv_path) / "bin" / "python")
            parts = [python, "-m", _WORKER_MODULE]
        elif self.conda_env:
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
        else:
            parts = [sys.executable, "-m", _WORKER_MODULE]

        parts.extend(worker_args.split())
        return parts
