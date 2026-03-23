"""BindCraft output ingestor.

Real BindCraft output structure (verified against actual runs):
    output_dir/
    +-- Accepted/                PDB files passing all filters
    |   +-- Ranked/              Final ranked PDBs (if run completed)
    |   +-- <name>.pdb
    +-- Rejected/
    +-- Trajectory/
    +-- MPNN/
    +-- final_design_stats.csv   Ranked design metrics
    +-- trajectory_stats.csv     All trajectory metrics
    +-- mpnn_design_stats.csv    MPNN stage metrics

Also supports older/alternative layouts:
    +-- final_designs/           Alternative name for accepted designs
    +-- scores/scores.csv        Alternative scores location

Chain convention: target-first, binder-last in PDB files.
Default: target=chain A, binder=chain B.
pLDDT scale: 0-100 (AF2 convention).

CSV column mapping (real BindCraft uses Average_ prefix):
    Design -> design_name
    Average_pLDDT -> plddt
    Average_i_pTM -> i_ptm
    Average_dG -> dG
    Sequence -> sequence
"""

from __future__ import annotations

import logging
from pathlib import Path

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor

logger = logging.getLogger(__name__)

# BindCraft uses mixed-case column names across versions.
# Map all known variants to a single canonical lowercase form.
_BINDCRAFT_COL_MAP: dict[str, str] = {
    "i_pTM": "i_ptm",
    "Average_i_pTM": "i_ptm",
    "i_pAE": "i_pae",
    "Average_i_pAE": "i_pae",
    "Sequence": "sequence",
}


class BindCraftIngestor(DesignIngestor):
    """Parse BindCraft output directory."""

    @property
    def tool_name(self) -> str:
        return "bindcraft"

    def ingest(
        self,
        output_dir: Path,
        target_chain: str = "A",
        binder_chain: str = "B",
        include_filtered: bool = False,
        **kwargs,
    ) -> list[IngestedDesign]:
        output_dir = Path(output_dir)

        # Find design PDBs directory
        designs_dir = self._find_designs_dir(output_dir)
        scores_csv = self._find_scores_csv(output_dir)

        # Load scores CSV
        scores_by_name: dict[str, dict[str, float]] = {}
        if scores_csv:
            for row in self._read_csv(scores_csv):
                # BindCraft uses "Design" column, not "design_name"
                name = (
                    row.get("Design")
                    or row.get("design_name")
                    or row.get("name")
                    or ""
                )
                if not name:
                    continue
                metrics = {}
                for k, v in row.items():
                    if k in ("Design", "design_name", "name"):
                        continue
                    # Normalise mixed-case BindCraft column names
                    norm_k = _BINDCRAFT_COL_MAP.get(k, k)
                    try:
                        metrics[norm_k] = float(v)
                    except (ValueError, TypeError):
                        pass
                scores_by_name[name] = metrics
        else:
            logger.warning(f"No scores CSV found in {output_dir}")

        # Find PDB files
        if designs_dir:
            pdb_files = sorted(designs_dir.glob("*.pdb"))
            # Exclude binder-only, Animation subdirs, etc.
            pdb_files = [
                p
                for p in pdb_files
                if not p.stem.endswith("_binder")
                and p.parent.name != "Animation"
            ]
        elif include_filtered:
            pdb_files = sorted(output_dir.rglob("*.pdb"))
            pdb_files = [
                p
                for p in pdb_files
                if not p.stem.endswith("_binder")
                and "Trajectory" not in str(p)
                and "MPNN" not in str(p)
                and "Rejected" not in str(p)
                and "Animation" not in str(p)
            ]
        else:
            raise FileNotFoundError(
                f"No Accepted/ or final_designs/ directory found in {output_dir}. "
                f"Contents: {[p.name for p in output_dir.iterdir()]}"
            )

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

            # Look for binder-only PDB
            binder_pdb = pdb_path.parent / f"{design_id}_binder.pdb"
            binder_path = binder_pdb if binder_pdb.exists() else None

            # Match to scores CSV by design name
            # BindCraft CSV "Design" column may not include model suffix
            tool_metrics = self._match_scores(design_id, scores_by_name)

            design = IngestedDesign(
                design_id=design_id,
                source_tool=SourceTool.BINDCRAFT,
                binder_sequence=binder_seq,
                binder_chain=binder_chain,
                target_sequence=target_seq,
                target_chain=target_chain,
                binder_length=len(binder_seq),
                target_length=len(target_seq),
                complex_structure_path=pdb_path,
                binder_structure_path=binder_path,
                tool_metrics=tool_metrics,
            )

            warnings = self._validate_design(design)
            for w in warnings:
                logger.warning(w)

            designs.append(design)

        logger.info(
            f"Ingested {len(designs)} BindCraft designs from {output_dir}"
        )
        return designs

    def _find_designs_dir(self, output_dir: Path) -> Path | None:
        """Find the directory containing accepted/final design PDBs.

        Checks (in order):
        1. Accepted/Ranked/ (if BindCraft completed ranking)
        2. Accepted/ (standard BindCraft output)
        3. final_designs/ (alternative name)
        4. One level nested versions of all above
        """
        candidates = [
            output_dir / "Accepted" / "Ranked",
            output_dir / "Accepted",
            output_dir / "final_designs",
        ]
        # Also check one level nested (target name subdirectory)
        for subdir in output_dir.iterdir():
            if subdir.is_dir() and subdir.name not in (
                "Rejected",
                "Trajectory",
                "MPNN",
            ):
                candidates.extend(
                    [
                        subdir / "Accepted" / "Ranked",
                        subdir / "Accepted",
                        subdir / "final_designs",
                    ]
                )

        for c in candidates:
            if c.is_dir() and any(c.glob("*.pdb")):
                return c

        return None

    def _find_scores_csv(self, output_dir: Path) -> Path | None:
        """Find the scores/metrics CSV file.

        Checks (in order):
        1. final_design_stats.csv (real BindCraft output)
        2. mpnn_design_stats.csv (has all MPNN-stage metrics)
        3. scores/scores.csv (alternative)
        4. scores.csv
        """
        candidates = [
            output_dir / "final_design_stats.csv",
            output_dir / "mpnn_design_stats.csv",
            output_dir / "scores" / "scores.csv",
            output_dir / "scores.csv",
        ]
        # Nested
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                candidates.extend(
                    [
                        subdir / "final_design_stats.csv",
                        subdir / "scores" / "scores.csv",
                    ]
                )

        for c in candidates:
            if c.is_file():
                return c

        return None

    def _match_scores(
        self, design_id: str, scores_by_name: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Match a PDB filename to scores CSV entry.

        Real BindCraft Ranked PDB names have rank prefix + model suffix:
            PDB:  1_BindMaster_test_02_l80_s351468_mpnn2_model2
            CSV:  BindMaster_test_02_l80_s351468_mpnn2

        We try progressively stripping prefixes and suffixes.
        """
        if design_id in scores_by_name:
            return scores_by_name[design_id]

        # Try stripping _modelN suffix
        name = design_id
        if "_model" in name:
            name = name[: name.rfind("_model")]
            if name in scores_by_name:
                return scores_by_name[name]

        # Try stripping rank prefix (N_)
        stripped = design_id
        if stripped and stripped[0].isdigit():
            idx = stripped.find("_")
            if idx > 0:
                stripped = stripped[idx + 1 :]
                if stripped in scores_by_name:
                    return scores_by_name[stripped]
                # Also strip _modelN from rank-stripped version
                if "_model" in stripped:
                    stripped2 = stripped[: stripped.rfind("_model")]
                    if stripped2 in scores_by_name:
                        return scores_by_name[stripped2]

        return {}
