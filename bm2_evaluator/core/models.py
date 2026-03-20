"""Core data models for the BM2 Evaluator.

All dataclasses used throughout the package. These define the contract
between ingestion, metrics, scoring, and reporting layers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class SourceTool(str, enum.Enum):
    """Design tool that produced a binder design."""

    BINDCRAFT = "bindcraft"
    BOLTZGEN = "boltzgen"
    MOSAIC = "mosaic"
    PXDESIGN = "pxdesign"
    RFDIFFUSION = "rfdiffusion"
    COMPLEXA = "complexa"
    GENERIC = "generic"


@dataclass
class IngestedDesign:
    """Standardized representation of a design from any tool.

    Produced by every ingestor. Input to the evaluation pipeline.
    tool_metrics stores raw tool-reported values and is never modified.
    """

    design_id: str
    source_tool: SourceTool
    binder_sequence: str
    binder_chain: str
    target_sequence: str
    target_chain: str
    binder_length: int
    target_length: int
    complex_structure_path: Path  # PDB or CIF
    binder_structure_path: Optional[Path] = None
    tool_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EngineResult:
    """Metrics from one refolding engine for one design."""

    engine: str  # "boltz2" or "af2"
    pae_matrix_path: Path
    pae_matrix_shape: tuple[int, int]
    bt_ipsae: float  # binder->target ipSAE
    tb_ipsae: float  # target->binder ipSAE
    ipsae_min: float  # min(bt, tb) -- Overath "weakest link"
    ipsae_max: float  # max(bt, tb) -- Dunbrack convention
    iptm: float  # 0-1
    ptm: float  # 0-1
    plddt_binder_mean_raw: float
    plddt_binder_mean_norm: float  # always 0-1
    plddt_target_mean_raw: float
    pae_interaction_mean: float
    pae_binder_mean: float
    refolded_structure_path: Path
    n_interface_contacts: int = 0


@dataclass
class EvaluationResult:
    """Complete evaluation of one design across all engines."""

    design: IngestedDesign
    engine_results: dict[str, EngineResult] = field(default_factory=dict)
    # Rosetta metrics (None if PyRosetta unavailable)
    rosetta_dG: Optional[float] = None
    rosetta_dSASA: Optional[float] = None
    rosetta_shape_comp: Optional[float] = None
    rosetta_hbonds: Optional[int] = None
    rosetta_clash_score: Optional[float] = None
    # Monomer validation
    monomer_rmsd: Optional[float] = None
    monomer_passes: Optional[bool] = None


@dataclass
class ScoredDesign:
    """A design with composite score, tier, and rank."""

    evaluation: EvaluationResult
    composite_score: float = 0.0
    tier: str = "unscored"  # consensus_hit|strong|moderate|weak|fail
    rank: int = 0
    ensemble_ipsae_min: float = 0.0
    ensemble_iptm: float = 0.0
    multi_model_agreement: Optional[float] = None


@dataclass
class EvalConfig:
    """Configuration for an evaluation run. All thresholds documented."""

    # PAE cutoff for ipSAE qualifying residues.
    # Default 10A: Adaptyv convention (uniform for Boltz2/AF2).
    # Dunbrack default is 15A. Configurable.
    pae_cutoff: float = 10.0

    # ipSAE threshold for consensus_hit tier.
    # BM2 default, calibrated from Overath 2025 dataset
    # (bioRxiv 2025.08.14.670059). Make configurable per target.
    ipsae_consensus_threshold: float = 0.61

    # ipSAE threshold for "strong" tier.
    ipsae_strong_threshold: float = 0.40

    # Monomer RMSD threshold (angstroms).
    monomer_rmsd_threshold: float = 3.0

    # Refolding engines to use.
    engines: list[str] = field(default_factory=lambda: ["boltz2", "af2"])

    # Use PyRosetta if available.
    use_rosetta: bool = False

    # pLDDT minimum for passing (normalized 0-1).
    plddt_min_norm: float = 0.70

    # ipTM minimum for "moderate" tier.
    iptm_min_moderate: float = 0.6

    # Output directory.
    output_dir: Path = field(default_factory=lambda: Path("./bm2_eval_output"))

    # Number of parallel refolding jobs.
    n_workers: int = 1
