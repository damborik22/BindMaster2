"""PXDesign (ByteDance Protenix) tool launcher.

PXDesign uses a Click CLI with YAML input configs.
CLI: pxdesign pipeline -i config.yaml -o output_dir --N_sample N --preset preview
Presets: preview (AF2-only), extended (AF2+Protenix), custom
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class PXDesignLauncher(ToolLauncher):
    """Launch PXDesign in its conda env."""

    def __init__(
        self,
        install_dir: Path | None = None,
        conda_env: str = "bindmaster_pxdesign",
    ):
        self._install_dir = Path(
            install_dir or Path.home() / "BindMaster" / "PXDesign"
        )
        self._conda_env = conda_env

    @property
    def name(self) -> str:
        return "pxdesign"

    @property
    def env_spec(self) -> str:
        return f"conda:{self._conda_env}"

    def check_installed(self) -> bool:
        if self._install_dir and (self._install_dir / "pxdesign").is_dir():
            return True
        env_path = Path.home() / "miniconda3" / "envs" / self._conda_env
        return env_path.is_dir()

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)

        target = campaign.target
        hotspots = run_config.hotspot_residues or (
            target.hotspot_residues if target else []
        )
        lo, hi = run_config.binder_length_range
        chains = target.chains if target else ["A"]

        # Build PXDesign YAML config
        chain_configs = {}
        for chain_id in chains:
            chain_cfg = {}
            if target and target.target_length:
                chain_cfg["crop"] = [f"1-{target.target_length}"]
            # PXDesign hotspots are integer residue numbers without chain prefix
            chain_hotspots = []
            for h in hotspots:
                if isinstance(h, int):
                    chain_hotspots.append(h)
                elif isinstance(h, str):
                    stripped = h.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    try:
                        chain_hotspots.append(int(stripped))
                    except ValueError:
                        pass
            if chain_hotspots:
                chain_cfg["hotspots"] = chain_hotspots
            chain_configs[chain_id] = chain_cfg

        pdb_path = str(target.pdb_path) if target else ""
        config = {
            "target": {
                "file": pdb_path,
                "chains": chain_configs,
            },
            "binder_length": (lo + hi) // 2,
        }

        config_path = run_dir / "pxdesign_input.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        preset = run_config.extra_settings.get("preset", "preview")

        return {
            "config_yaml": str(config_path),
            "output_dir": str(run_dir / "output"),
            "num_samples": run_config.num_designs,
            "preset": preset,
        }

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        (run_dir / "output").mkdir(exist_ok=True)

        commands = (
            f"python -m pxdesign.runner.cli pipeline "
            f"-i {prepared['config_yaml']} "
            f"-o {prepared['output_dir']} "
            f"--N_sample {prepared['num_samples']} "
            f"--preset {prepared['preset']} "
            f"--N_max_runs 1"
        )
        script = self._write_conda_launch_script(
            run_dir=run_dir,
            env_name=self._conda_env,
            commands=commands,
            cwd=str(self._install_dir),
            log_file=str(run_dir / "pxdesign.log"),
        )
        return subprocess.Popen(["bash", str(script)])

    def is_complete(self, run_dir: Path) -> bool:
        output = run_dir / "output"
        return output.exists() and any(output.rglob("*.pdb"))

    def output_dir(self, run_dir: Path) -> Path:
        return run_dir / "output"

    def parser_name(self) -> str:
        return "pxdesign"
