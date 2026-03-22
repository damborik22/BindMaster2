"""RFdiffusion + LigandMPNN tool launcher.

Two-stage pipeline:
    1. RFdiffusion: backbone generation
    2. LigandMPNN: sequence design

Conda env: bindmaster_rfaa (or rfdiffusion)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


class RFdiffusionLauncher(ToolLauncher):
    """Launch RFdiffusion + LigandMPNN pipeline."""

    def __init__(
        self,
        install_dir: Path | None = None,
        conda_env: str = "bindmaster_rfaa",
    ):
        self._install_dir = Path(
            install_dir
            or Path.home() / "BindMaster" / "rf_diffusion_all_atom"
        )
        self._conda_env = conda_env

    @property
    def name(self) -> str:
        return "rfdiffusion"

    @property
    def env_spec(self) -> str:
        return f"conda:{self._conda_env}"

    def check_installed(self) -> bool:
        # Check install dir exists with config
        if self._install_dir and (self._install_dir / "config").is_dir():
            return True
        # Fallback: check conda env
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

        # RFAA contig string: just the binder length range
        contig = f"{lo}-{hi}"
        hotspot_str = ",".join(str(h) for h in hotspots) if hotspots else ""

        return {
            "target_pdb": str(target.pdb_path) if target else "",
            "contig": contig,
            "hotspots": hotspot_str,
            "num_designs": run_config.num_designs,
            "output_prefix": str(run_dir / "output" / "sample"),
            "ckpt_path": str(self._install_dir / "weights" / "RFDiffusionAA_paper_weights.pt"),
            "diffusion_steps": run_config.extra_settings.get("diffusion_steps", 100),
        }

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        (run_dir / "output").mkdir(exist_ok=True)

        cmd_parts = [
            "python run_inference.py",
            f'inference.input_pdb="{prepared["target_pdb"]}"',
            f'inference.output_prefix="{prepared["output_prefix"]}"',
            f'inference.ckpt_path="{prepared["ckpt_path"]}"',
            f"inference.num_designs={prepared['num_designs']}",
            f"diffuser.T={prepared['diffusion_steps']}",
            f"contigmap.contigs=\"['{prepared['contig']}']\"",
        ]
        if prepared["hotspots"]:
            hotspot_val = prepared["hotspots"]
            cmd_parts.append(f"'ppi.hotspot_res=[{hotspot_val}]'")

        commands = " ".join(cmd_parts)
        script = self._write_conda_launch_script(
            run_dir=run_dir,
            env_name=self._conda_env,
            commands=commands,
            cwd=str(self._install_dir),
            log_file=str(run_dir / "rfdiffusion.log"),
            env_vars={
                "PYTHONPATH": (
                    f"{self._install_dir}:"
                    f"{Path.home() / 'BindMaster' / 'LigandMPNN'}"
                ),
            },
        )
        return subprocess.Popen(["bash", str(script)])

    def is_complete(self, run_dir: Path) -> bool:
        output = run_dir / "output"
        return output.exists() and any(output.glob("*.pdb"))

    def output_dir(self, run_dir: Path) -> Path:
        return run_dir / "output"

    def parser_name(self) -> str:
        return "rfdiffusion"
