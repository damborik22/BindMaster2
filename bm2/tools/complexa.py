"""Proteina-Complexa (NVIDIA) tool launcher.

CLI: complexa generate --target PDB --chains A --binder-length N --num-samples N
Search strategies: best_of_n, beam_search, mcts, fk_steering
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class ComplexaLauncher(ToolLauncher):
    """Launch Proteina-Complexa."""

    def __init__(
        self,
        install_dir: Path | None = None,
        conda_env: str = "complexa",
    ):
        self._install_dir = install_dir
        self._conda_env = conda_env

    @property
    def name(self) -> str:
        return "complexa"

    @property
    def env_spec(self) -> str:
        return f"conda:{self._conda_env}"

    def check_installed(self) -> bool:
        try:
            result = subprocess.run(
                ["conda", "run", "--no-banner", "-n", self._conda_env,
                 "complexa", "--help"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)

        target = campaign.target
        lo, hi = run_config.binder_length_range
        strategy = run_config.extra_settings.get("search_strategy", "best_of_n")

        return {
            "target_pdb": str(target.pdb_path) if target else "",
            "chains": " ".join(target.chains) if target else "A",
            "binder_length": (lo + hi) // 2,
            "num_samples": run_config.num_designs,
            "search_strategy": strategy,
            "output_dir": str(run_dir / "output"),
        }

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        (run_dir / "output").mkdir(exist_ok=True)

        cmd = (
            f"conda run --no-banner -n {self._conda_env} "
            f"complexa generate "
            f"--target {prepared['target_pdb']} "
            f"--chains {prepared['chains']} "
            f"--binder-length {prepared['binder_length']} "
            f"--num-samples {prepared['num_samples']} "
            f"--search-strategy {prepared['search_strategy']} "
            f"--output-dir {prepared['output_dir']}"
        )
        log_path = run_dir / "complexa.log"
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
            any(output.glob("*.pdb")) or any(output.glob("*.cif"))
        )

    def output_dir(self, run_dir: Path) -> Path:
        return run_dir / "output"

    def parser_name(self) -> str:
        return "complexa"
