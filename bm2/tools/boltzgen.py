"""BoltzGen tool launcher.

BoltzGen uses a YAML spec to define the design task.
Output: CIF files + metrics CSVs in output directory.
Conda env: BoltzGen
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class BoltzGenLauncher(ToolLauncher):
    """Launch BoltzGen in its conda env."""

    def __init__(
        self,
        install_dir: Path | None = None,
        conda_env: str = "BoltzGen",
    ):
        self._install_dir = Path(
            install_dir or Path.home() / "BindMaster" / "BoltzGen"
        )
        self._conda_env = conda_env

    @property
    def name(self) -> str:
        return "boltzgen"

    @property
    def env_spec(self) -> str:
        return f"conda:{self._conda_env}"

    def check_installed(self) -> bool:
        return self._install_dir.exists() and (
            self._install_dir / "src" / "boltzgen"
        ).exists()

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)

        target = campaign.target
        lo, hi = run_config.binder_length_range
        protocol = run_config.extra_settings.get("protocol", "protein-anything")

        # Build YAML spec
        spec = {
            "target": {
                "pdb": str(target.pdb_path) if target else "",
                "chains": target.chains if target else ["A"],
            },
            "binder": {
                "length_range": [lo, hi],
            },
            "protocol": protocol,
            "num_designs": run_config.num_designs,
            "budget": run_config.extra_settings.get("budget", run_config.num_designs),
        }

        if run_config.hotspot_residues or (
            target and target.hotspot_residues
        ):
            spec["hotspots"] = run_config.hotspot_residues or target.hotspot_residues

        spec_path = run_dir / "spec.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

        return {"spec_yaml": spec_path}

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        output_dir = run_dir / "output"
        output_dir.mkdir(exist_ok=True)

        cmd = (
            f"conda run --no-banner -n {self._conda_env} "
            f"boltzgen run {prepared['spec_yaml']} "
            f"--output {output_dir}"
        )
        log_path = run_dir / "boltzgen.log"
        log_file = open(log_path, "w")
        return subprocess.Popen(
            cmd,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    def is_complete(self, run_dir: Path) -> bool:
        output = run_dir / "output"
        return output.exists() and (
            any(output.glob("*.cif"))
            or (output / "final_ranked_designs").exists()
        )

    def output_dir(self, run_dir: Path) -> Path:
        return run_dir / "output"

    def parser_name(self) -> str:
        return "boltzgen"
