"""BindCraft tool launcher.

BindCraft expects:
    - target JSON: starting_pdb, chains, target_hotspot_residues, lengths, etc.
    - filters JSON: default_filters.json (pLDDT, ipTM, Rosetta thresholds)
    - advanced JSON: optimization settings (4stage_multimer, etc.)

Output: <design_path>/final_designs/ with PDBs + scores/scores.csv
Conda env: BindCraft
Install dir: ~/BindCraft/
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class BindCraftLauncher(ToolLauncher):
    """Launch BindCraft in its conda env."""

    def __init__(
        self,
        install_dir: Path | None = None,
        conda_env: str = "BindCraft",
    ):
        self._install_dir = Path(
            install_dir or Path.home() / "BindMaster" / "BindCraft"
        )
        self._conda_env = conda_env

    @property
    def name(self) -> str:
        return "bindcraft"

    @property
    def env_spec(self) -> str:
        return f"conda:{self._conda_env}"

    def check_installed(self) -> bool:
        return (self._install_dir / "bindcraft.py").exists()

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "output"

        target = campaign.target
        hotspots = run_config.hotspot_residues or (
            target.hotspot_residues if target else []
        )
        lo, hi = run_config.binder_length_range

        target_json = {
            "design_path": str(output_dir),
            "binder_name": campaign.name,
            "starting_pdb": str(target.pdb_path) if target else "",
            "chains": ",".join(target.chains) if target else "A",
            "target_hotspot_residues": ",".join(hotspots),
            "lengths": list(range(lo, hi + 1)),
            "number_of_final_designs": run_config.num_designs,
        }
        target_json.update(run_config.extra_settings)

        target_path = run_dir / "target.json"
        with open(target_path, "w") as f:
            json.dump(target_json, f, indent=2)

        filters_src = (
            self._install_dir / "settings_filters" / "default_filters.json"
        )
        advanced_src = (
            self._install_dir
            / "settings_advanced"
            / "default_4stage_multimer.json"
        )

        return {
            "target_json": target_path,
            "filters_json": filters_src,
            "advanced_json": advanced_src,
        }

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        cmd = (
            f"conda run --no-banner -n {self._conda_env} "
            f"python {self._install_dir / 'bindcraft.py'} "
            f"--settings '{prepared['target_json']}' "
            f"--filters '{prepared['filters_json']}' "
            f"--advanced '{prepared['advanced_json']}'"
        )
        log_path = run_dir / "bindcraft.log"
        log_file = open(log_path, "w")
        return subprocess.Popen(
            cmd,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(self._install_dir),
        )

    def is_complete(self, run_dir: Path) -> bool:
        output = run_dir / "output"
        # BindCraft signals completion by populating Accepted/Ranked/
        # and writing final_design_stats.csv
        ranked = output / "Accepted" / "Ranked"
        stats = output / "final_design_stats.csv"
        return (ranked.exists() and any(ranked.glob("*.pdb"))) or stats.exists()

    def output_dir(self, run_dir: Path) -> Path:
        return run_dir / "output"

    def parser_name(self) -> str:
        return "bindcraft"
