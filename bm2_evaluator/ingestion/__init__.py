"""Ingestion layer: parse outputs from any design tool into IngestedDesign."""

from pathlib import Path

from bm2_evaluator.ingestion.base import DesignIngestor
from bm2_evaluator.ingestion.generic import GenericIngestor
from bm2_evaluator.ingestion.bindcraft import BindCraftIngestor
from bm2_evaluator.ingestion.boltzgen import BoltzGenIngestor
from bm2_evaluator.ingestion.mosaic import MosaicIngestor
from bm2_evaluator.ingestion.pxdesign import PXDesignIngestor
from bm2_evaluator.ingestion.rfdiffusion import RFdiffusionIngestor
from bm2_evaluator.ingestion.complexa import ComplexaIngestor

INGESTORS: dict[str, type[DesignIngestor]] = {
    "generic": GenericIngestor,
    "bindcraft": BindCraftIngestor,
    "boltzgen": BoltzGenIngestor,
    "mosaic": MosaicIngestor,
    "pxdesign": PXDesignIngestor,
    "rfdiffusion": RFdiffusionIngestor,
    "complexa": ComplexaIngestor,
}


def get_ingestor(tool_name: str) -> DesignIngestor:
    """Get an ingestor instance by tool name."""
    cls = INGESTORS.get(tool_name)
    if cls is None:
        raise ValueError(
            f"Unknown tool: {tool_name}. Available: {list(INGESTORS.keys())}"
        )
    return cls()


def auto_detect(output_dir: Path) -> str:
    """Attempt to auto-detect which tool produced the output directory.

    Heuristics:
    - scores/scores.csv + final_designs/ -> bindcraft
    - aggregate_metrics_analyze.csv + *.cif -> boltzgen
    - *.trb files -> rfdiffusion
    - ptx_iptm in CSV headers -> pxdesign
    - eval CSV with ipae column -> complexa
    - fallback -> generic
    """
    d = output_dir

    if (d / "final_designs").is_dir() and (
        (d / "scores" / "scores.csv").is_file() or (d / "scores.csv").is_file()
    ):
        return "bindcraft"

    if (d / "aggregate_metrics_analyze.csv").is_file():
        return "boltzgen"

    trb_files = list(d.glob("*.trb"))
    if trb_files:
        return "rfdiffusion"

    for csv_path in d.glob("*.csv"):
        try:
            with open(csv_path) as f:
                header = f.readline()
            if "ptx_iptm" in header:
                return "pxdesign"
            if "ipae" in header and "iptm" in header:
                return "complexa"
        except (OSError, UnicodeDecodeError):
            continue

    return "generic"
