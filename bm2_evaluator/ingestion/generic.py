"""Generic ingestor for any directory of PDB/CIF files.

Also used as the base for Mosaic, which outputs generic PDB directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bm2_evaluator.core.models import IngestedDesign, SourceTool
from bm2_evaluator.ingestion.base import DesignIngestor

logger = logging.getLogger(__name__)


class GenericIngestor(DesignIngestor):
    """Parse a directory of PDB/CIF files with no tool-specific metadata.

    Chain assignment:
        - Uses provided target_chain and binder_chain IDs.
        - If auto_chains=True and chains not found, uses heuristic:
          longest chain = target, next chain = binder.
    """

    @property
    def tool_name(self) -> str:
        return "generic"

    def ingest(
        self,
        output_dir: Path,
        target_chain: str = "A",
        binder_chain: str = "B",
        metrics_csv: Optional[Path] = None,
        auto_chains: bool = False,
        **kwargs,
    ) -> list[IngestedDesign]:
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        # Find all structure files
        structure_files = sorted(
            list(output_dir.glob("*.pdb"))
            + list(output_dir.glob("*.cif"))
            + list(output_dir.glob("*.mmcif"))
        )

        if not structure_files:
            raise FileNotFoundError(
                f"No PDB/CIF files found in {output_dir}"
            )

        # Load optional metrics CSV
        metrics_by_id: dict[str, dict[str, str]] = {}
        if metrics_csv and metrics_csv.is_file():
            rows = self._read_csv(metrics_csv)
            for row in rows:
                design_id = row.get("design_id") or row.get("name") or row.get("id")
                if design_id:
                    metrics_by_id[design_id] = row

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

            # If specified chains not found, try auto-detection
            if auto_chains and (
                target_chain not in seqs or binder_chain not in seqs
            ):
                seqs, target_chain, binder_chain = self._auto_detect_chains(
                    struct_path
                )

            target_seq = seqs.get(target_chain, "")
            binder_seq = seqs.get(binder_chain, "")

            if not target_seq or not binder_seq:
                logger.warning(
                    f"Missing chain(s) in {struct_path}: "
                    f"target={target_chain}({'found' if target_seq else 'missing'}), "
                    f"binder={binder_chain}({'found' if binder_seq else 'missing'})"
                )
                continue

            # Extract tool metrics from CSV if available
            tool_metrics = {}
            if design_id in metrics_by_id:
                for k, v in metrics_by_id[design_id].items():
                    try:
                        tool_metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass

            design = IngestedDesign(
                design_id=design_id,
                source_tool=SourceTool.GENERIC,
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

        logger.info(f"Ingested {len(designs)} designs from {output_dir}")
        return designs

    def _auto_detect_chains(
        self, struct_path: Path
    ) -> tuple[dict[str, str], str, str]:
        """Auto-detect target and binder chains.

        Heuristic: longest chain = target, next longest = binder.
        """
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import protein_letters_3to1

        ext = struct_path.suffix.lower()
        if ext in (".cif", ".mmcif"):
            import gemmi

            doc = gemmi.cif.read(str(struct_path))
            st = gemmi.make_structure_from_block(doc[0])
            model = st[0]
            chain_seqs = {}
            for chain in model:
                seq = []
                for residue in chain:
                    if residue.is_amino_acid():
                        code = gemmi.find_tabulated_residue(
                            residue.name
                        ).one_letter_code
                        seq.append(code if code != "?" else "X")
                if seq:
                    chain_seqs[chain.name] = "".join(seq)
        else:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("s", str(struct_path))
            model = structure[0]
            chain_seqs = {}
            for chain in model.get_chains():
                seq = []
                for residue in chain.get_residues():
                    if residue.id[0] != " ":
                        continue
                    resname = residue.get_resname().strip()
                    code = protein_letters_3to1.get(resname, "X")
                    seq.append(code)
                if seq:
                    chain_seqs[chain.id] = "".join(seq)

        if len(chain_seqs) < 2:
            raise ValueError(
                f"Need at least 2 chains for auto-detection, "
                f"found {len(chain_seqs)} in {struct_path}"
            )

        sorted_chains = sorted(
            chain_seqs.items(), key=lambda x: len(x[1]), reverse=True
        )
        target_chain = sorted_chains[0][0]
        binder_chain = sorted_chains[1][0]

        return chain_seqs, target_chain, binder_chain
