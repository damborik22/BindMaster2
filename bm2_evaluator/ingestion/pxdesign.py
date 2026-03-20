"""PXDesign (ByteDance Protenix) output ingestor.

Expected structure:
    output_dir/
    +-- *.pdb                PDB files for passing designs
    +-- metrics.csv          Metrics CSV (or results.csv, etc.)

Metrics CSV columns:
    - ptx_iptm: 0-1 scale (Protenix ipTM)
    - ptx_plddt: 0-1 scale (Protenix pLDDT)
    - af2ig_plddt: 0-100 scale (AF2 in-silico validation pLDDT)
    - af2ig_rmsd: RMSD from AF2 validation (Angstroms)

Note: PXDesign has TWO pLDDT values on DIFFERENT scales.
Both stored in tool_metrics with original names.

Filter methodology (from paper bioRxiv 2025.08.15.670450):
    1. Protenix filter: ptx_iptm >= 0.85 (strict) or >= 0.80 (basic)
    2. AF2-IG filter: af2ig_plddt > 80, RMSD < 2.0 (strict)
    3. Dual-filter intersection captures complementary true positives
"""

from __future__ import annotations

import logging
from pathlib import Path

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor

logger = logging.getLogger(__name__)


class PXDesignIngestor(DesignIngestor):
    """Parse PXDesign output directory."""

    @property
    def tool_name(self) -> str:
        return "pxdesign"

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
        """Load metrics from available CSV files."""
        metrics: dict[str, dict[str, float]] = {}

        csv_candidates = [
            output_dir / "metrics.csv",
            output_dir / "results.csv",
            output_dir / "scores.csv",
        ]
        # Also check for any CSV with ptx_ columns
        csv_candidates.extend(sorted(output_dir.glob("*.csv")))

        for csv_path in csv_candidates:
            if not csv_path.is_file():
                continue
            try:
                rows = self._read_csv(csv_path)
                if not rows:
                    continue

                # Check if this looks like a PXDesign CSV
                headers = set(rows[0].keys())
                has_pxdesign_cols = bool(
                    headers & {"ptx_iptm", "ptx_plddt", "af2ig_plddt"}
                )
                if not has_pxdesign_cols and csv_path.name not in (
                    "metrics.csv",
                    "results.csv",
                    "scores.csv",
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
                        try:
                            entry[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                    metrics[design_id] = entry
                break  # Use first valid CSV found

            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")

        return metrics
