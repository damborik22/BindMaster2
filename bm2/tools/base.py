"""Tool launcher interface.

BM2 launches design tools in their own envs as subprocesses.
It never modifies tool code — only reads their output.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from bm2.core.models import Campaign, ToolRunConfig


class ToolLauncher(ABC):
    """Interface for launching an external design tool."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier matching DesignSource enum value."""

    @property
    @abstractmethod
    def env_spec(self) -> str:
        """How to activate: conda env name, venv path, or module path."""

    @abstractmethod
    def check_installed(self) -> bool:
        """Verify the tool is accessible."""

    @abstractmethod
    def prepare_config(
        self,
        campaign: Campaign,
        run_config: ToolRunConfig,
        run_dir: Path,
    ) -> dict:
        """Create tool-native config files in run_dir.

        Returns dict with paths to created config files.
        """

    @abstractmethod
    def launch(self, prepared: dict, run_dir: Path) -> subprocess.Popen:
        """Start the tool as a subprocess.

        Returns Popen for monitoring. Does NOT wait for completion.
        """

    @abstractmethod
    def is_complete(self, run_dir: Path) -> bool:
        """Check if the tool has finished by examining output files."""

    @abstractmethod
    def output_dir(self, run_dir: Path) -> Path:
        """Path to design outputs (fed to bm2-evaluator ingestion)."""

    @abstractmethod
    def parser_name(self) -> str:
        """Which bm2-evaluator ingestor to use (e.g. 'bindcraft')."""
