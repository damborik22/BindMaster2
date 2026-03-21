"""Mosaic tool launcher.

Default mode: templates hallucinate_bindmaster.py from BM1's example dir
and injects campaign parameters. Alternatively, accepts a custom script
via extra_settings["script"].

Output: PDB files + designs.csv with native Boltz2 metrics.
Env: UV venv at ~/BindMaster/Mosaic/.venv
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bm2.core.models import Campaign, ToolRunConfig
from bm2.tools.base import ToolLauncher


# Template source locations (searched in order)
_HALLUCINATE_SEARCH_PATHS = [
    "examples/bindmaster_examples/hallucinate_bindmaster.py",
]


class MosaicLauncher(ToolLauncher):
    """Launch Mosaic binder design in its UV venv."""

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

    def _find_hallucinate_template(self) -> Path | None:
        """Find hallucinate_bindmaster.py in known locations."""
        for rel_path in _HALLUCINATE_SEARCH_PATHS:
            candidate = self._install_dir / rel_path
            if candidate.exists():
                return candidate
        return None

    def _inject_parameters(
        self,
        content: str,
        target_sequence: str,
        n_designs: int,
        top_k: int,
        min_length: int,
        max_length: int,
        length_step: int = 5,
    ) -> str:
        """Inject campaign parameters into the hallucination script.

        Replaces the BINDMASTER PARAMETERS block at the top of the file,
        matching BM1's configurator pattern.
        """
        old_block = (
            'TARGET_SEQUENCE = "REPLACE_ME"  # target protein sequence\n'
            "N_DESIGNS = 100  # Stage 1: how many designs to generate per length\n"
            "TOP_K = 5  # Stage 2: how many top designs to refold and export PDB\n"
            "MIN_LENGTH = 65  # minimum binder length (aa)\n"
            "MAX_LENGTH = 100  # maximum binder length (aa)\n"
            "LENGTH_STEP = 5  # step between scanned lengths; set MIN=MAX for a single length"
        )
        new_block = (
            f"TARGET_SEQUENCE = {target_sequence!r}  # target protein sequence\n"
            f"N_DESIGNS = {n_designs}  # Stage 1: how many designs to generate per length\n"
            f"TOP_K = {top_k}  # Stage 2: how many top designs to refold and export PDB\n"
            f"MIN_LENGTH = {min_length}  # minimum binder length (aa)\n"
            f"MAX_LENGTH = {max_length}  # maximum binder length (aa)\n"
            f"LENGTH_STEP = {length_step}  # step between scanned lengths; set MIN=MAX for a single length"
        )

        if old_block in content:
            return content.replace(old_block, new_block)
        import logging
        logging.getLogger(__name__).warning(
            "Could not find parameter block in hallucinate template — "
            "edit the script manually."
        )
        return content

    def prepare_config(
        self, campaign: Campaign, run_config: ToolRunConfig, run_dir: Path
    ) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)

        # Option 1: User provides custom script
        custom_script = run_config.extra_settings.get("script")
        if custom_script:
            return {"script": Path(custom_script)}

        # Option 2: Template hallucinate_bindmaster.py
        template = self._find_hallucinate_template()
        if template is None:
            raise ValueError(
                "Mosaic hallucination template not found. "
                f"Expected at: {self._install_dir}/examples/bindmaster_examples/"
                "hallucinate_bindmaster.py\n"
                "Alternatively, provide a script path via extra_settings['script']."
            )

        target = campaign.target
        target_seq = target.target_sequence if target else ""
        lo, hi = run_config.binder_length_range
        top_k = run_config.extra_settings.get("top_k", run_config.num_designs)
        length_step = run_config.extra_settings.get("length_step", 5)

        content = template.read_text()
        content = self._inject_parameters(
            content=content,
            target_sequence=target_seq,
            n_designs=run_config.num_designs,
            top_k=top_k,
            min_length=lo,
            max_length=hi,
            length_step=length_step,
        )

        script_path = run_dir / "hallucinate.py"
        script_path.write_text(content)

        return {"script": script_path}

    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        # Use bare 'python' since venv activation puts it on PATH
        commands = f"python {prepared['script']}"
        script = self._write_venv_launch_script(
            run_dir=run_dir,
            venv_path=self._venv_path,
            commands=commands,
            cwd=str(run_dir),
            log_file=str(run_dir / "mosaic.log"),
        )
        return subprocess.Popen(["bash", str(script)])

    def is_complete(self, run_dir: Path) -> bool:
        # Check for PDB output in structures_*/ directories or output/
        if any(run_dir.glob("structures_*/*.pdb")):
            return True
        output = run_dir / "output"
        return output.exists() and any(output.glob("*.pdb"))

    def output_dir(self, run_dir: Path) -> Path:
        # Mosaic hallucination writes to run_dir directly
        return run_dir

    def parser_name(self) -> str:
        return "mosaic"
