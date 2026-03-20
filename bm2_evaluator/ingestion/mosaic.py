"""Mosaic output ingestor.

Mosaic is a library (not a CLI tool) that outputs generic PDB directories.
This ingestor delegates to GenericIngestor, overriding source_tool for
provenance tracking.
"""

from __future__ import annotations

from pathlib import Path

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor
from bm2_evaluator.ingestion.generic import GenericIngestor


class MosaicIngestor(DesignIngestor):
    """Parse Mosaic output (delegates to GenericIngestor)."""

    @property
    def tool_name(self) -> str:
        return "mosaic"

    def ingest(
        self,
        output_dir: Path,
        target_chain: str = "A",
        binder_chain: str = "B",
        **kwargs,
    ) -> list[IngestedDesign]:
        generic = GenericIngestor()
        designs = generic.ingest(
            output_dir, target_chain, binder_chain, **kwargs
        )
        for d in designs:
            d.source_tool = SourceTool.MOSAIC
        return designs
