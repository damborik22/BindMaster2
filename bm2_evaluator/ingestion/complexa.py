"""Proteina-Complexa (NVIDIA) output ingestor.

Expected structure:
    output_dir/
    +-- *.pdb or *.cif      Design structures (co-generated sequences)
    +-- eval/
        +-- eval.csv         Evaluation CSV from Complexa's pipeline

Eval CSV columns:
    - ipae: interaction PAE
    - iptm: interaction pTM (0-1)
    - plddt: predicted LDDT
    - n_hbonds: number of hydrogen bonds
    - rmsd: backbone RMSD

Search strategies: best_of_n, beam_search, mcts, fk_steering,
                  generate_and_hallucinate
"""

from __future__ import annotations

import logging
from pathlib import Path

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor

logger = logging.getLogger(__name__)


class ComplexaIngestor(DesignIngestor):
    """Parse Proteina-Complexa output directory."""

    @property
    def tool_name(self) -> str:
        return "complexa"

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

        # Find structure files (PDB and CIF)
        structure_files = sorted(
            list(output_dir.glob("*.pdb"))
            + list(output_dir.glob("*.cif"))
        )

        if not structure_files:
            raise FileNotFoundError(
                f"No PDB/CIF files found in {output_dir}"
            )

        designs = []
        for struct_path in structure_files:
            design_id = struct_path.stem

            try:
                seqs = self._extract_sequences(
                    struct_path, [target_chain, binder_chain]
                )
            except Exception as e:
                logger.warning(f"Failed to parse {struct_path}: {e}")
                continue

            target_seq = seqs.get(target_chain, "")
            binder_seq = seqs.get(binder_chain, "")

            if not target_seq or not binder_seq:
                logger.warning(f"Missing chains in {struct_path}")
                continue

            tool_metrics = metrics_by_id.get(design_id, {})

            design = IngestedDesign(
                design_id=design_id,
                source_tool=SourceTool.COMPLEXA,
                binder_sequence=binder_seq,
                binder_chain=binder_chain,
                target_sequence=target_seq,
                target_chain=target_chain,
                binder_length=len(binder_seq),
                target_length=len(target_seq),
                complex_structure_path=struct_path,
                tool_metrics=tool_metrics,
            )

            warnings = self._validate_design(design)
            for w in warnings:
                logger.warning(w)

            designs.append(design)

        logger.info(
            f"Ingested {len(designs)} Complexa designs from {output_dir}"
        )
        return designs

    def _load_metrics(self, output_dir: Path) -> dict[str, dict[str, float]]:
        """Load metrics from Complexa evaluation CSV."""
        metrics: dict[str, dict[str, float]] = {}

        csv_candidates = [
            output_dir / "eval" / "eval.csv",
            output_dir / "eval.csv",
            output_dir / "metrics.csv",
            output_dir / "results.csv",
        ]

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
                    entry: dict[str, float] = {}
                    for k, v in row.items():
                        try:
                            entry[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                    metrics[design_id] = entry
                break  # Use first valid CSV

            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")

        return metrics
