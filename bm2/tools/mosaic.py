"""Mosaic tool launcher.

Mosaic is a library, not a CLI tool. Users provide their own
design scripts. BM2 launches the script in the Mosaic env.
Output: generic PDB directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class MosaicLauncher(ToolLauncher):
    """Launch user-provided Mosaic design script."""

    def __init__(
        self,
        install_dir: Path | None = None,
        venv_path: str | None = None,
    ):
        self._install_dir = Path(
            install_dir or Path.home() / "BindMaster" / "Mosaic"
        )
        self._venv_path = venv_path or str(self._install_dir / ".venv")

    @property
    def name(self) -> str:
        return "mosaic"

    @property
    def env_spec(self) -> str:
        return f"venv:{self._venv_path}"

    def check_installed(self) -> bool:
        venv = Path(self._venv_path)
        return (venv / "bin" / "python").exists()

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)

        # User must provide a design script path in extra_settings
        script_path = run_config.extra_settings.get("script")
        if not script_path:
            raise ValueError(
                "Mosaic requires 'script' in extra_settings: "
                "path to your design script"
            )

        return {"script": Path(script_path)}

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        python = Path(self._venv_path) / "bin" / "python"
        output_dir = run_dir / "output"
        output_dir.mkdir(exist_ok=True)

        cmd = f"{python} {prepared['script']} --output_dir {output_dir}"
        log_path = run_dir / "mosaic.log"
        log_file = open(log_path, "w")
        return subprocess.Popen(
            cmd,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    def is_complete(self, run_dir: Path) -> bool:
        output = run_dir / "output"
        return output.exists() and any(output.glob("*.pdb"))

    def output_dir(self, run_dir: Path) -> Path:
        return run_dir / "output"

    def parser_name(self) -> str:
        return "mosaic"
