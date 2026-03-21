"""Proteina-Complexa (NVIDIA) tool launcher.

Complexa uses Hydra YAML configs and a UV virtual environment.
CLI: complexa design <config.yaml> ++run_name=X ++generation.task_name=X
Pipeline: generate → filter → evaluate → analyze

Note: This launcher writes to Complexa's install directory (target registration
in configs/targets/targets_dict.yaml and PDB copy to assets/target_data/custom_targets/).
This is a necessary exception because Complexa's Hydra config system resolves targets
from targets_dict.yaml — there's no CLI override for arbitrary target paths.
BM2 entries are prefixed 'bm2_' for identification. The operation is idempotent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class ComplexaLauncher(ToolLauncher):
    """Launch Proteina-Complexa via UV venv."""

    def __init__(
        self,
        install_dir: Path | None = None,
    ):
        self._install_dir = Path(
            install_dir or Path.home() / "BindMaster" / "Proteina-Complexa"
        )

    @property
    def name(self) -> str:
        return "complexa"

    @property
    def env_spec(self) -> str:
        return f"venv:{self._install_dir / '.venv'}"

    def check_installed(self) -> bool:
        return (self._install_dir / ".venv" / "bin" / "complexa").exists()

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)

        target = campaign.target
        task_name = f"bm2_{campaign.name}"
        lo, hi = run_config.binder_length_range
        hotspots = run_config.hotspot_residues or (
            target.hotspot_residues if target else []
        )

        # Copy target PDB to Complexa's custom_targets directory
        custom_dir = self._install_dir / "assets" / "target_data" / "custom_targets"
        custom_dir.mkdir(parents=True, exist_ok=True)
        target_filename = f"{campaign.name}"
        dest_pdb = custom_dir / f"{target_filename}.pdb"
        if target and target.pdb_path:
            shutil.copy2(target.pdb_path, dest_pdb)

        # Build target_input string (e.g., "A1-115")
        chains = target.chains if target else ["A"]
        target_input = ",".join(chains)
        if target and target.target_length:
            target_input = f"{chains[0]}1-{target.target_length}"

        # Register target in targets_dict.yaml
        targets_yaml = (
            self._install_dir / "configs" / "targets" / "targets_dict.yaml"
        )
        targets_dict = {}
        if targets_yaml.exists():
            with open(targets_yaml) as f:
                targets_dict = yaml.safe_load(f) or {}

        cfg_root = targets_dict.setdefault("target_dict_cfg", {})
        cfg_root[task_name] = {
            "source": "custom_targets",
            "target_filename": target_filename,
            "target_path": str(dest_pdb),
            "target_input": target_input,
            "hotspot_residues": [
                h if isinstance(h, str) else f"{chains[0]}{h}"
                for h in hotspots
            ],
            "binder_length": [lo, hi],
            "pdb_id": None,
        }

        with open(targets_yaml, "w") as f:
            yaml.dump(targets_dict, f, default_flow_style=False, sort_keys=False)

        # Pipeline mode: "design" (full) or "generate" (designs only)
        pipeline_mode = run_config.extra_settings.get("pipeline_mode", "design")
        config_yaml = str(
            self._install_dir / "configs" / "search_binder_local_pipeline.yaml"
        )

        return {
            "task_name": task_name,
            "config_yaml": config_yaml,
            "pipeline_mode": pipeline_mode,
            "num_samples": run_config.num_designs,
        }

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        (run_dir / "output").mkdir(exist_ok=True)

        mode = prepared["pipeline_mode"]
        task_name = prepared["task_name"]

        commands = (
            f"complexa {mode} {prepared['config_yaml']} "
            f"++run_name={task_name} "
            f"++generation.task_name={task_name} "
            f"++gen_njobs=1 ++eval_njobs=1 "
            f"++seed=42"
        )
        script = self._write_venv_launch_script(
            run_dir=run_dir,
            venv_path=str(self._install_dir / ".venv"),
            commands=commands,
            cwd=str(self._install_dir),
            log_file=str(run_dir / "complexa.log"),
        )
        return subprocess.Popen(["bash", str(script)])

    def is_complete(self, run_dir: Path) -> bool:
        # Complexa writes output relative to its install dir
        inference = self._install_dir / "inference"
        eval_results = self._install_dir / "evaluation_results"
        # Check for PDB/CIF in any subdirectory of inference/
        if inference.exists():
            for subdir in inference.iterdir():
                if subdir.is_dir() and (
                    any(subdir.glob("*.pdb")) or any(subdir.glob("*.cif"))
                ):
                    return True
        if eval_results.exists() and any(eval_results.rglob("*.csv")):
            return True
        return False

    def output_dir(self, run_dir: Path) -> Path:
        # Find the most recently modified inference subdirectory
        inference = self._install_dir / "inference"
        if inference.exists():
            subdirs = [d for d in inference.iterdir() if d.is_dir()]
            if subdirs:
                return max(subdirs, key=lambda d: d.stat().st_mtime)
        return run_dir / "output"

    def parser_name(self) -> str:
        return "complexa"
