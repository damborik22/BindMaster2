"""Configuration loading and validation.

Environment variable overrides:
    BM2_PAE_CUTOFF       -> EvalConfig.pae_cutoff
    BM2_IPSAE_THRESHOLD  -> EvalConfig.ipsae_consensus_threshold
    BM2_OUTPUT_DIR       -> EvalConfig.output_dir
    BM2_ENGINES          -> EvalConfig.engines (comma-separated)
    BM2_N_WORKERS        -> EvalConfig.n_workers
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from bm2_evaluator.core.models import EvalConfig


def load_config(path: Optional[Path] = None) -> EvalConfig:
    """Load config from optional TOML file, then apply env overrides.

    Args:
        path: Optional path to a TOML config file.

    Returns:
        Validated EvalConfig.
    """
    config = EvalConfig()

    if path is not None:
        config = _load_from_toml(path, config)

    config = _apply_env_overrides(config)

    errors = validate_config(config)
    if errors:
        raise ValueError(f"Invalid config: {'; '.join(errors)}")

    return config


def _load_from_toml(path: Path, config: EvalConfig) -> EvalConfig:
    """Override config values from a TOML file."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)

    eval_section = data.get("eval", data)

    if "pae_cutoff" in eval_section:
        config.pae_cutoff = float(eval_section["pae_cutoff"])
    if "ipsae_consensus_threshold" in eval_section:
        config.ipsae_consensus_threshold = float(
            eval_section["ipsae_consensus_threshold"]
        )
    if "ipsae_strong_threshold" in eval_section:
        config.ipsae_strong_threshold = float(eval_section["ipsae_strong_threshold"])
    if "monomer_rmsd_threshold" in eval_section:
        config.monomer_rmsd_threshold = float(eval_section["monomer_rmsd_threshold"])
    if "engines" in eval_section:
        config.engines = list(eval_section["engines"])
    if "use_rosetta" in eval_section:
        config.use_rosetta = bool(eval_section["use_rosetta"])
    if "plddt_min_norm" in eval_section:
        config.plddt_min_norm = float(eval_section["plddt_min_norm"])
    if "iptm_min_moderate" in eval_section:
        config.iptm_min_moderate = float(eval_section["iptm_min_moderate"])
    if "output_dir" in eval_section:
        config.output_dir = Path(eval_section["output_dir"])
    if "n_workers" in eval_section:
        config.n_workers = int(eval_section["n_workers"])

    return config


def _apply_env_overrides(config: EvalConfig) -> EvalConfig:
    """Apply environment variable overrides."""
    if v := os.environ.get("BM2_PAE_CUTOFF"):
        config.pae_cutoff = float(v)
    if v := os.environ.get("BM2_IPSAE_THRESHOLD"):
        config.ipsae_consensus_threshold = float(v)
    if v := os.environ.get("BM2_OUTPUT_DIR"):
        config.output_dir = Path(v)
    if v := os.environ.get("BM2_ENGINES"):
        config.engines = [e.strip() for e in v.split(",")]
    if v := os.environ.get("BM2_N_WORKERS"):
        config.n_workers = int(v)
    return config


def validate_config(config: EvalConfig) -> list[str]:
    """Validate config values. Returns list of errors (empty = valid)."""
    errors = []

    if config.pae_cutoff <= 0:
        errors.append(f"pae_cutoff must be positive, got {config.pae_cutoff}")
    if not (0 < config.ipsae_consensus_threshold <= 1):
        errors.append(
            f"ipsae_consensus_threshold must be in (0, 1], "
            f"got {config.ipsae_consensus_threshold}"
        )
    if not (0 < config.ipsae_strong_threshold <= 1):
        errors.append(
            f"ipsae_strong_threshold must be in (0, 1], "
            f"got {config.ipsae_strong_threshold}"
        )
    if config.monomer_rmsd_threshold <= 0:
        errors.append(
            f"monomer_rmsd_threshold must be positive, "
            f"got {config.monomer_rmsd_threshold}"
        )
    if not config.engines:
        errors.append("engines list must not be empty")
    if not (0 < config.plddt_min_norm <= 1):
        errors.append(
            f"plddt_min_norm must be in (0, 1], got {config.plddt_min_norm}"
        )
    if config.n_workers < 1:
        errors.append(f"n_workers must be >= 1, got {config.n_workers}")

    return errors


def save_config(config: EvalConfig, path: Path) -> None:
    """Save config as TOML for reproducibility."""
    lines = [
        "[eval]",
        f"pae_cutoff = {config.pae_cutoff}",
        f"ipsae_consensus_threshold = {config.ipsae_consensus_threshold}",
        f"ipsae_strong_threshold = {config.ipsae_strong_threshold}",
        f"monomer_rmsd_threshold = {config.monomer_rmsd_threshold}",
        f'engines = {config.engines}',
        f"use_rosetta = {str(config.use_rosetta).lower()}",
        f"plddt_min_norm = {config.plddt_min_norm}",
        f"iptm_min_moderate = {config.iptm_min_moderate}",
        f'output_dir = "{config.output_dir}"',
        f"n_workers = {config.n_workers}",
    ]
    path.write_text("\n".join(lines) + "\n")


def resolve_output_dir(config: EvalConfig, run_id: str) -> Path:
    """Return config.output_dir / run_id, creating if needed."""
    out = config.output_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def pae_matrix_path(output_dir: Path, design_id: str, engine: str) -> Path:
    """Standardized path for saved PAE matrices."""
    pae_dir = output_dir / "pae" / engine
    pae_dir.mkdir(parents=True, exist_ok=True)
    return pae_dir / f"{design_id}.npy"
