"""PXDesign (ByteDance Protenix) tool launcher.

PXDesign may be a local install or web server.
Conda env: bindmaster_pxdesign
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class PXDesignLauncher(ToolLauncher):
    """Launch PXDesign."""

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
        # Check install dir exists with pxdesign package
        if self._install_dir and (self._install_dir / "pxdesign").is_dir():
            return True
        # Fallback: check conda env exists
        env_path = Path.home() / "miniconda3" / "envs" / self._conda_env
        return env_path.is_dir()

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)
        # PXDesign config — TBD based on actual installation
        return {"run_dir": run_dir}

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        raise NotImplementedError(
            "PXDesign launch not yet implemented. "
            "Provide designs manually via bm2-evaluator."
        )

    def is_complete(self, run_dir: Path) -> bool:
        output = run_dir / "output"
        return output.exists() and any(output.glob("*.pdb"))

    def output_dir(self, run_dir: Path) -> Path:
        return run_dir / "output"

    def parser_name(self) -> str:
        return "pxdesign"
