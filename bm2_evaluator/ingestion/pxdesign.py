"""PXDesign (ByteDance Protenix) output ingestor.

Expected structure:
    output_dir/
    +-- *.pdb                        PDB files for passing designs
    +-- sample_level_output.csv      Primary metrics CSV (real pipeline output)
    +-- metrics.csv                  Alternative metrics CSV

Metrics CSV columns (real PXDesign pipeline output):
    - i_pTM: 0-1 scale (Protenix ipTM)
    - pLDDT: 0-1 scale (Protenix pLDDT)
    - af2_iptm: AF2 in-silico validation ipTM
    - af2_plddt: 0-100 scale (AF2 in-silico validation pLDDT)

Legacy column names (also supported):
    - ptx_iptm, ptx_plddt, af2ig_plddt, af2ig_rmsd

Note: PXDesign has TWO pLDDT values on DIFFERENT scales.
Both stored in tool_metrics with original names.
Values may be bracket-wrapped like [0.85] — stripped before conversion.

Filter methodology (from paper bioRxiv 2025.08.15.670450):
    1. Protenix filter: ptx_iptm >= 0.85 (strict) or >= 0.80 (basic)
    2. AF2-IG filter: af2ig_plddt > 80, RMSD < 2.0 (strict)
    3. Dual-filter intersection captures complementary true positives
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor

logger = logging.getLogger(__name__)


# Column mapping: real PXDesign pipeline names → standardised tool_metrics keys
_PXDESIGN_COL_MAP = {
    # Real pipeline column names (from sample_level_output.csv)
    "i_pTM": "pxdesign_iptm",
    "af2_iptm": "pxdesign_af2_iptm",
    "pLDDT": "pxdesign_plddt",
    "af2_plddt": "pxdesign_af2_plddt",
    # Legacy / fallback column names
    "ptx_iptm": "pxdesign_iptm",
    "ptx_plddt": "pxdesign_plddt",
    "af2ig_plddt": "pxdesign_af2_plddt",
    "af2ig_rmsd": "pxdesign_af2_rmsd",
}

# Column names that identify a CSV as PXDesign output
_PXDESIGN_MARKER_COLS = {
    "ptx_iptm", "ptx_plddt", "af2ig_plddt",
    "i_pTM", "pLDDT", "af2_iptm", "af2_plddt",
}


class PXDesignIngestor(DesignIngestor):
    """Parse PXDesign output directory."""

    @property
    def tool_name(self) -> str:
        return "pxdesign"

    @staticmethod
    def _safe_float(v) -> float | None:
        """Convert to float, stripping bracket-wrapped values like [0.85]."""
        if v is None:
            return None
        try:
            s = str(v).strip().strip("[]")
            f = float(s)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (ValueError, TypeError):
            return None

    def ingest(
        self,
        output_dir: Path,
        target_chain: str = "A",
        binder_chain: str = "B",
        **kwargs,
    ) -> list[IngestedDesign]:
        output_dir = Path(output_dir)

        # Find metrics CSV
        metrics_by_id = self._load_metrics(output_dir)

        # Find PDB files
        pdb_files = sorted(output_dir.glob("*.pdb"))
        if not pdb_files:
            raise FileNotFoundError(f"No PDB files found in {output_dir}")

        designs = []
        for pdb_path in pdb_files:
            design_id = pdb_path.stem

            try:
                seqs = self._extract_sequences_from_pdb(
                    pdb_path, [target_chain, binder_chain]
                )
            except Exception as e:
                logger.warning(f"Failed to parse {pdb_path}: {e}")
                continue

            target_seq = seqs.get(target_chain, "")
            binder_seq = seqs.get(binder_chain, "")

            if not target_seq or not binder_seq:
                logger.warning(f"Missing chains in {pdb_path}")
                continue

            tool_metrics = metrics_by_id.get(design_id, {})

            design = IngestedDesign(
                design_id=design_id,
                source_tool=SourceTool.PXDESIGN,
                binder_sequence=binder_seq,
                binder_chain=binder_chain,
                target_sequence=target_seq,
                target_chain=target_chain,
                binder_length=len(binder_seq),
                target_length=len(target_seq),
                complex_structure_path=pdb_path,
                tool_metrics=tool_metrics,
            )

            warnings = self._validate_design(design)
            for w in warnings:
                logger.warning(w)

            designs.append(design)

        logger.info(
            f"Ingested {len(designs)} PXDesign designs from {output_dir}"
        )
        return designs

    def _load_metrics(self, output_dir: Path) -> dict[str, dict[str, float]]:
        """Load metrics from available CSV files.

        Search order:
        1. sample_level_output.csv (real PXDesign pipeline output)
        2. metrics.csv / results.csv / scores.csv (legacy names)
        3. Recursive glob for sample_level_output.csv in subdirectories
        4. Any CSV with recognised PXDesign column names
        """
        metrics: dict[str, dict[str, float]] = {}

        csv_candidates: list[Path] = [
            output_dir / "sample_level_output.csv",
            output_dir / "metrics.csv",
            output_dir / "results.csv",
            output_dir / "scores.csv",
        ]
        # Recursive search for sample_level_output.csv in subdirectories
        csv_candidates.extend(
            sorted(output_dir.rglob("sample_level_output.csv"))
        )
        # Also check for any other CSV files in the output directory
        csv_candidates.extend(sorted(output_dir.glob("*.csv")))

        # De-duplicate while preserving order
        seen: set[Path] = set()
        unique_candidates: list[Path] = []
        for p in csv_candidates:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_candidates.append(p)

        known_csv_names = {
            "sample_level_output.csv",
            "metrics.csv",
            "results.csv",
            "scores.csv",
        }

        for csv_path in unique_candidates:
            if not csv_path.is_file():
                continue
            try:
                rows = self._read_csv(csv_path)
                if not rows:
                    continue

                # Check if this looks like a PXDesign CSV
                headers = set(rows[0].keys())
                has_pxdesign_cols = bool(headers & _PXDESIGN_MARKER_COLS)
                if (
                    not has_pxdesign_cols
                    and csv_path.name not in known_csv_names
                ):
                    continue

                for row in rows:
                    design_id = (
                        row.get("design_id")
                        or row.get("name")
                        or row.get("id")
                        or ""
                    )
                    if not design_id:
                        continue
                    entry: dict[str, float] = {}
                    for k, v in row.items():
                        val = self._safe_float(v)
                        if val is not None:
                            # Map known columns to standardised names
                            mapped = _PXDESIGN_COL_MAP.get(k, k)
                            entry[mapped] = val
                    metrics[design_id] = entry
                break  # Use first valid CSV found

            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")

        return metrics
