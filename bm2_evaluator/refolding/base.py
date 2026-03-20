"""Abstract base class for refolding engines.

Each engine wraps a structure prediction tool (Boltz2, AF2) and produces
standardized output: PAE matrix (.npy), confidence metrics (.json),
structure file (.pdb), and a metrics summary (.json).

The worker script approach: the engine class runs in the evaluator env
(pure numpy/biopython) and launches a worker script via subprocess in
the target env (Mosaic venv for Boltz2, binder-eval-af2 for AF2).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorkerOutput:
    """Standardized output from a refolding worker script.

    This is the contract between worker scripts (running in engine envs)
    and engine classes (running in evaluator env). Workers produce 4 files
    plus this metadata in metrics.json.
    """

    engine: str  # "boltz2" or "af2"
    chain_order: str  # "binder_first" or "target_first"
    target_length: int
    binder_length: int
    iptm: float  # 0-1
    ptm: float  # 0-1
    plddt_binder_mean: float  # raw scale
    plddt_binder_min: float  # raw scale
    plddt_target_mean: float  # raw scale
    plddt_complex_mean: float  # raw scale
    plddt_scale_max: float  # 1.0 for Boltz2, 100.0 for AF2
    pae_matrix_file: str  # relative path: "pae.npy"
    structure_file: str  # relative path: "structure.pdb"
    success: bool
    error: Optional[str] = None
    # Extra metrics from the engine (aux metrics, etc.)
    extra: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> WorkerOutput:
        """Load from a metrics.json file produced by a worker."""
        with open(path) as f:
            data = json.load(f)
        extra = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "engine",
                "chain_order",
                "target_length",
                "binder_length",
                "iptm",
                "ptm",
                "plddt_binder_mean",
                "plddt_binder_min",
                "plddt_target_mean",
                "plddt_complex_mean",
                "plddt_scale_max",
                "pae_matrix_file",
                "structure_file",
                "success",
                "error",
            }
            and isinstance(v, (int, float))
        }
        return cls(
            engine=data["engine"],
            chain_order=data["chain_order"],
            target_length=int(data["target_length"]),
            binder_length=int(data["binder_length"]),
            iptm=float(data.get("iptm", 0.0)),
            ptm=float(data.get("ptm", 0.0)),
            plddt_binder_mean=float(data.get("plddt_binder_mean", 0.0)),
            plddt_binder_min=float(data.get("plddt_binder_min", 0.0)),
            plddt_target_mean=float(data.get("plddt_target_mean", 0.0)),
            plddt_complex_mean=float(data.get("plddt_complex_mean", 0.0)),
            plddt_scale_max=float(data.get("plddt_scale_max", 1.0)),
            pae_matrix_file=data.get("pae_matrix_file", "pae.npy"),
            structure_file=data.get("structure_file", "structure.pdb"),
            success=bool(data.get("success", False)),
            error=data.get("error"),
            extra=extra,
        )

    def to_json(self, path: Path) -> None:
        """Save to a metrics.json file."""
        data = {
            "engine": self.engine,
            "chain_order": self.chain_order,
            "target_length": self.target_length,
            "binder_length": self.binder_length,
            "iptm": self.iptm,
            "ptm": self.ptm,
            "plddt_binder_mean": self.plddt_binder_mean,
            "plddt_binder_min": self.plddt_binder_min,
            "plddt_target_mean": self.plddt_target_mean,
            "plddt_complex_mean": self.plddt_complex_mean,
            "plddt_scale_max": self.plddt_scale_max,
            "pae_matrix_file": self.pae_matrix_file,
            "structure_file": self.structure_file,
            "success": self.success,
            "error": self.error,
        }
        data.update(self.extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def get_binder_slice(self) -> slice:
        """Get PAE matrix slice for binder residues."""
        if self.chain_order == "binder_first":
            return slice(0, self.binder_length)
        else:  # target_first
            return slice(self.target_length, self.target_length + self.binder_length)

    def get_target_slice(self) -> slice:
        """Get PAE matrix slice for target residues."""
        if self.chain_order == "binder_first":
            return slice(self.binder_length, self.binder_length + self.target_length)
        else:  # target_first
            return slice(0, self.target_length)


@dataclass
class MonomerResult:
    """Result from monomer (binder-alone) refolding."""

    plddt_mean: float  # raw scale
    plddt_scale_max: float  # 1.0 or 100.0
    structure_path: Path
    success: bool
    error: Optional[str] = None


class RefoldingEngine(ABC):
    """Abstract base for a refolding engine.

    Each implementation calls a worker script via subprocess in the
    engine's conda env. The worker produces standardized output files.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier: 'boltz2' or 'af2'."""

    @abstractmethod
    def refold_complex(
        self,
        binder_seq: str,
        target_seq: str,
        output_dir: Path,
    ) -> WorkerOutput:
        """Refold binder+target complex via worker subprocess.

        Args:
            binder_seq: Binder amino acid sequence.
            target_seq: Target amino acid sequence.
            output_dir: Directory for output files.

        Returns:
            WorkerOutput with paths to standardized files.
        """

    @abstractmethod
    def refold_monomer(
        self,
        binder_seq: str,
        output_dir: Path,
    ) -> MonomerResult:
        """Refold binder alone for monomer validation.

        Args:
            binder_seq: Binder amino acid sequence.
            output_dir: Directory for output files.

        Returns:
            MonomerResult with pLDDT and structure path.
        """

    def check_available(self) -> bool:
        """Check if this engine's environment is available."""
        return False

    def _write_fasta(
        self, sequences: dict[str, str], output_path: Path
    ) -> None:
        """Write sequences to FASTA file.

        Args:
            sequences: {chain_id: sequence} dict.
            output_path: Path to write FASTA file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for chain_id, seq in sequences.items():
                f.write(f">{chain_id}\n{seq}\n")


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
