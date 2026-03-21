"""Mosaic output ingestor.

Handles hallucinate_bindmaster.py output:
- PDB files in structures_*aa_*/ directories
- Metrics CSV: designs.csv with native Boltz2 metrics
  (bt_ipsae, tb_ipsae, ipsae_min, iptm, plddt_binder_mean, etc.)

Falls back to GenericIngestor for non-hallucination Mosaic output.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor
from bm2_evaluator.ingestion.generic import GenericIngestor

logger = logging.getLogger(__name__)


class MosaicIngestor(DesignIngestor):
    """Parse Mosaic hallucination output."""

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
        output_dir = Path(output_dir)

        # Check for hallucination output (designs.csv + structures_*/ dirs)
        designs_csv = output_dir / "designs.csv"
        if designs_csv.exists():
            return self._ingest_hallucination(
                output_dir, designs_csv, target_chain, binder_chain
            )

        # Fallback to generic ingestor
        generic = GenericIngestor()
        designs = generic.ingest(
            output_dir, target_chain, binder_chain, **kwargs
        )
        for d in designs:
            d.source_tool = SourceTool.MOSAIC
        return designs

    def _ingest_hallucination(
        self,
        output_dir: Path,
        designs_csv: Path,
        target_chain: str,
        binder_chain: str,
    ) -> list[IngestedDesign]:
        """Parse hallucinate_bindmaster.py output."""
        rows = self._read_csv(designs_csv)
        if not rows:
            logger.warning(f"Empty designs.csv at {designs_csv}")
            return []

        # Build metrics lookup by PDB path
        metrics_by_pdb: dict[str, dict[str, float]] = {}
        for row in rows:
            pdb_path = row.get("pdb", "")
            if not pdb_path:
                continue
            entry: dict[str, float] = {}
            for k in [
                "ranking_loss", "iptm_aux", "bt_ipsae", "tb_ipsae",
                "ipsae_min", "bt_iptm", "binder_ptm", "plddt_aux",
                "bb_pae", "bt_pae_aux", "tb_pae", "intra_contact",
                "target_contact", "pTMEnergy", "iptm",
                "plddt_binder_mean", "plddt_binder_min",
                "pae_bt_mean", "pae_tb_mean", "pae_overall_mean",
            ]:
                try:
                    entry[f"mosaic_{k}"] = float(row[k])
                except (KeyError, ValueError, TypeError):
                    pass
            metrics_by_pdb[pdb_path] = entry

        # Find PDB files in structures_*/ directories and output_dir
        pdb_files = sorted(output_dir.glob("structures_*/*.pdb"))
        if not pdb_files:
            pdb_files = sorted(output_dir.glob("*.pdb"))
        if not pdb_files:
            pdb_files = sorted(output_dir.glob("output/*.pdb"))

        if not pdb_files:
            raise FileNotFoundError(
                f"No PDB files found in {output_dir}/structures_*/"
            )

        designs = []
        for pdb_path in pdb_files:
            design_id = pdb_path.stem

            try:
                seqs = self._extract_sequences(
                    pdb_path, [binder_chain, target_chain]
                )
            except Exception as e:
                logger.warning(f"Failed to parse {pdb_path}: {e}")
                continue

            target_seq = seqs.get(target_chain, "")
            binder_seq = seqs.get(binder_chain, "")

            if not binder_seq:
                # Mosaic hallucination: binder is first chain (A), target is second (B)
                # Try swapping if standard assignment failed
                binder_seq = seqs.get(target_chain, "")
                target_seq = seqs.get(binder_chain, "")
                if not binder_seq:
                    logger.warning(f"No binder sequence in {pdb_path}")
                    continue

            # Match metrics by relative path
            rel_path = str(pdb_path.relative_to(output_dir))
            tool_metrics = metrics_by_pdb.get(rel_path, {})
            # Also try just the path as stored in CSV
            if not tool_metrics:
                for csv_path, m in metrics_by_pdb.items():
                    if pdb_path.name in csv_path:
                        tool_metrics = m
                        break

            design = IngestedDesign(
                design_id=design_id,
                source_tool=SourceTool.MOSAIC,
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
            f"Ingested {len(designs)} Mosaic designs from {output_dir}"
        )
        return designs
