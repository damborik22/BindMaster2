"""BM2 global configuration.

Loaded from ~/.bm2/config.toml or env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BM2Config:
    """Global BM2 configuration."""

    base_dir: Path = field(default_factory=lambda: Path.home() / ".bm2")

    # Tool paths (auto-discovered or manual override)
    tool_paths: dict[str, str] = field(default_factory=dict)
    tool_envs: dict[str, str] = field(default_factory=dict)

    # Evaluator settings
    eval_engines: list[str] = field(
        default_factory=lambda: ["boltz2", "af2"]
    )
    eval_rosetta: bool = False
    pae_cutoff: float = 15.0  # Dunbrack convention
    ipsae_threshold: float = 0.61  # BM2 default from Overath 2025

    # GPU settings
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    max_parallel_tools: int = 1

    @classmethod
    def load(cls, path: Optional[Path] = None) -> BM2Config:
        """Load from TOML file. Falls back to defaults."""
        config = cls()

        if path is None:
            path = config.base_dir / "config.toml"

        if path.exists():
            config = _load_from_toml(path, config)

        # Env var overrides
        if v := os.environ.get("BM2_BASE_DIR"):
            config.base_dir = Path(v)
        if v := os.environ.get("BM2_GPU_IDS"):
            config.gpu_ids = [int(x) for x in v.split(",")]

        return config

    def save(self, path: Optional[Path] = None) -> None:
        """Save to TOML."""
        if path is None:
            path = self.base_dir / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "[bm2]",
            f'base_dir = "{self.base_dir}"',
            f"eval_engines = {self.eval_engines}",
            f"eval_rosetta = {str(self.eval_rosetta).lower()}",
            f"pae_cutoff = {self.pae_cutoff}",
            f"ipsae_threshold = {self.ipsae_threshold}",
            f"gpu_ids = {self.gpu_ids}",
            f"max_parallel_tools = {self.max_parallel_tools}",
            "",
            "[tools.paths]",
        ]
        for tool, p in self.tool_paths.items():
            lines.append(f'{tool} = "{p}"')
        lines.append("")
        lines.append("[tools.envs]")
        for tool, env in self.tool_envs.items():
            lines.append(f'{tool} = "{env}"')

        path.write_text("\n".join(lines) + "\n")


def _load_from_toml(path: Path, config: BM2Config) -> BM2Config:
    """Load config values from TOML file."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)

    bm2 = data.get("bm2", {})
    if "base_dir" in bm2:
        config.base_dir = Path(bm2["base_dir"])
    if "eval_engines" in bm2:
        config.eval_engines = list(bm2["eval_engines"])
    if "eval_rosetta" in bm2:
        config.eval_rosetta = bool(bm2["eval_rosetta"])
    if "pae_cutoff" in bm2:
        config.pae_cutoff = float(bm2["pae_cutoff"])
    if "ipsae_threshold" in bm2:
        config.ipsae_threshold = float(bm2["ipsae_threshold"])
    if "gpu_ids" in bm2:
        config.gpu_ids = list(bm2["gpu_ids"])
    if "max_parallel_tools" in bm2:
        config.max_parallel_tools = int(bm2["max_parallel_tools"])

    tools = data.get("tools", {})
    if "paths" in tools:
        config.tool_paths = dict(tools["paths"])
    if "envs" in tools:
        config.tool_envs = dict(tools["envs"])

    return config
