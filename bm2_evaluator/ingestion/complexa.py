"""Proteina-Complexa (NVIDIA) output ingestor.

Handles two output formats:
1. Real Complexa pipeline output:
   - Structures in inference/{run_name}/ (PDB files)
   - Metrics in evaluation_results/binder_results_*.csv
   - Columns: self_complex_i_pTM, self_complex_pLDDT, self_complex_i_pAE,
              self_binder_scRMSD
   - Legacy columns also supported: _pAE_complex, _pLDDT_complex

2. Manual/generic input (fallback):
   - Structures as PDB/CIF in output_dir/
   - Metrics in eval/eval.csv with columns: ipae, iptm, plddt, n_hbonds, rmsd
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

        # Find structure files — search output_dir and subdirectories
        structure_files = sorted(
            list(output_dir.glob("*.pdb"))
            + list(output_dir.glob("*.cif"))
        )
        # Also search one level of subdirectories (Complexa inference output)
        if not structure_files:
            for subdir in sorted(output_dir.iterdir()):
                if subdir.is_dir():
                    structure_files.extend(sorted(
                        list(subdir.glob("*.pdb"))
                        + list(subdir.glob("*.cif"))
                    ))

        if not structure_files:
            raise FileNotFoundError(
                f"No PDB/CIF files found in {output_dir} or subdirectories"
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
        """Load metrics from Complexa evaluation CSV.

        Searches for:
        1. Real Complexa output: evaluation_results/binder_results_*.csv
        2. Fallback static paths: eval/eval.csv, eval.csv, metrics.csv
        """
        metrics: dict[str, dict[str, float]] = {}

        # Search 1: Glob for real Complexa binder_results CSVs
        eval_dirs = [
            output_dir / "evaluation_results",
            output_dir.parent / "evaluation_results",  # one level up
        ]
        for eval_dir in eval_dirs:
            if not eval_dir.is_dir():
                continue
            candidates = sorted(eval_dir.glob("binder_results_*.csv"))
            if candidates:
                metrics = self._parse_complexa_csv(candidates[0])
                if metrics:
                    return metrics

        # Search 2: Fallback static paths (manual/generic input)
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
                break

            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")

        return metrics

    def _parse_complexa_csv(
        self, csv_path: Path
    ) -> dict[str, dict[str, float]]:
        """Parse a real Complexa binder_results CSV with native column names."""
        metrics: dict[str, dict[str, float]] = {}

        # Column mapping: Complexa native → standardized tool_metrics keys
        # Real Complexa evaluation CSV uses self_complex_* and self_binder_* prefixes
        col_map = {
            "self_complex_i_pTM": "complexa_iptm",
            "self_complex_pLDDT": "complexa_plddt",
            "self_complex_i_pAE": "complexa_ipae",
            "self_binder_scRMSD": "complexa_scrmsd",
            # Legacy / fallback column names
            "_pAE_complex": "complexa_ipae",
            "_pLDDT_complex": "complexa_plddt",
        }

        try:
            rows = self._read_csv(csv_path)
            for row in rows:
                design_id = (
                    row.get("design_id")
                    or row.get("name")
                    or row.get("sample_name")
                    or row.get("id")
                    or ""
                )
                if not design_id:
                    continue
                entry: dict[str, float] = {}
                for k, v in row.items():
                    try:
                        val = float(v)
                    except (ValueError, TypeError):
                        continue
                    # Map known columns to standard names
                    mapped = col_map.get(k, k)
                    entry[mapped] = val
                    # Also capture scRMSD columns
                    if "scRMSD" in k:
                        entry[f"complexa_{k.lstrip('_')}"] = val
                metrics[design_id] = entry

        except Exception as e:
            logger.warning(f"Failed to parse Complexa CSV {csv_path}: {e}")

        return metrics
