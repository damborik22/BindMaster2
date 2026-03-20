"""BoltzGen output ingestor.

Expected structure:
    output_dir/
    +-- *.cif                           Structure files (CIF, not PDB)
    +-- aggregate_metrics_analyze.csv   Summary metrics
    +-- per_target_metrics_analyze.csv  Per-target metrics
    +-- final_ranked_designs/
        +-- all_designs_metrics.csv
        +-- final_designs_metrics_*.csv

Structure format: CIF (uses gemmi to parse).
pLDDT scale: 0-1 (Boltz2 convention).
Chain ordering: follows input YAML entity order.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor

logger = logging.getLogger(__name__)


class BoltzGenIngestor(DesignIngestor):
    """Parse BoltzGen output directory."""

    @property
    def tool_name(self) -> str:
        return "boltzgen"

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

        # Find CIF structure files
        cif_files = sorted(output_dir.glob("*.cif"))

        # Also check subdirectories (inverse_folded/, refold_cif/)
        for subdir_name in ("inverse_folded", "refold_cif", "final_ranked_designs"):
            subdir = output_dir / subdir_name
            if subdir.is_dir():
                cif_files.extend(sorted(subdir.glob("*.cif")))
                # Check nested subdirs
                for nested in subdir.iterdir():
                    if nested.is_dir():
                        cif_files.extend(sorted(nested.glob("*.cif")))

        # Deduplicate by stem, preferring top-level files
        seen = set()
        unique_cifs = []
        for f in cif_files:
            if f.stem not in seen:
                seen.add(f.stem)
                unique_cifs.append(f)

        if not unique_cifs:
            raise FileNotFoundError(f"No CIF files found in {output_dir}")

        designs = []
        for cif_path in unique_cifs:
            design_id = cif_path.stem

            try:
                seqs = self._extract_sequences_from_cif(
                    cif_path, [target_chain, binder_chain]
                )
            except Exception as e:
                logger.warning(f"Failed to parse {cif_path}: {e}")
                continue

            target_seq = seqs.get(target_chain, "")
            binder_seq = seqs.get(binder_chain, "")

            if not target_seq or not binder_seq:
                logger.warning(f"Missing chains in {cif_path}")
                continue

            # Get tool metrics (note: pLDDT is 0-1 scale for BoltzGen)
            tool_metrics = metrics_by_id.get(design_id, {})

            design = IngestedDesign(
                design_id=design_id,
                source_tool=SourceTool.BOLTZGEN,
                binder_sequence=binder_seq,
                binder_chain=binder_chain,
                target_sequence=target_seq,
                target_chain=target_chain,
                binder_length=len(binder_seq),
                target_length=len(target_seq),
                complex_structure_path=cif_path,
                tool_metrics=tool_metrics,
            )

            warnings = self._validate_design(design)
            for w in warnings:
                logger.warning(w)

            designs.append(design)

        logger.info(
            f"Ingested {len(designs)} BoltzGen designs from {output_dir}"
        )
        return designs

    def _load_metrics(self, output_dir: Path) -> dict[str, dict[str, float]]:
        """Load metrics from available CSVs."""
        metrics: dict[str, dict[str, float]] = {}

        csv_candidates = [
            output_dir / "aggregate_metrics_analyze.csv",
            output_dir / "per_target_metrics_analyze.csv",
        ]
        # Check final_ranked_designs/
        ranked_dir = output_dir / "final_ranked_designs"
        if ranked_dir.is_dir():
            csv_candidates.extend(sorted(ranked_dir.glob("*.csv")))

        for csv_path in csv_candidates:
            if not csv_path.is_file():
                continue
            try:
                rows = self._read_csv(csv_path)
                for row in rows:
                    design_id = (
                        row.get("design_id")
                        or row.get("name")
                        or row.get("id")
                        or ""
                    )
                    if not design_id:
                        continue
                    if design_id not in metrics:
                        metrics[design_id] = {}
                    for k, v in row.items():
                        try:
                            metrics[design_id][k] = float(v)
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")

        return metrics
