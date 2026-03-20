"""Abstract base class for tool-specific design parsers."""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from bm2_evaluator.core.models import IngestedDesign

logger = logging.getLogger(__name__)


class DesignIngestor(ABC):
    """Base class for tool-specific design parsers.

    Each ingestor knows how to parse a specific tool's output directory
    and produce standardized IngestedDesign objects.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name identifier for this tool."""

    @abstractmethod
    def ingest(
        self,
        output_dir: Path,
        target_chain: str = "A",
        binder_chain: str = "B",
        **kwargs,
    ) -> list[IngestedDesign]:
        """Parse all designs from a tool's output directory.

        Args:
            output_dir: Root directory of the tool's output.
            target_chain: Chain ID for the target in the structure.
            binder_chain: Chain ID for the binder in the structure.
            **kwargs: Tool-specific options.

        Returns:
            List of IngestedDesign objects.
        """

    def _extract_sequences_from_pdb(
        self, pdb_path: Path, chain_ids: list[str]
    ) -> dict[str, str]:
        """Extract amino acid sequences from a PDB file by chain ID.

        Uses BioPython PDBParser. Returns {chain_id: sequence}.
        """
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import protein_letters_3to1

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(pdb_path))
        model = structure[0]

        result = {}
        for chain_id in chain_ids:
            if chain_id not in [c.id for c in model.get_chains()]:
                logger.warning(
                    f"Chain {chain_id} not found in {pdb_path}. "
                    f"Available: {[c.id for c in model.get_chains()]}"
                )
                continue
            chain = model[chain_id]
            seq = []
            for residue in chain.get_residues():
                resname = residue.get_resname().strip()
                if residue.id[0] != " ":
                    continue  # skip heteroatoms
                code = protein_letters_3to1.get(resname, "X")
                seq.append(code)
            result[chain_id] = "".join(seq)

        return result

    def _extract_sequences_from_cif(
        self, cif_path: Path, chain_ids: list[str]
    ) -> dict[str, str]:
        """Extract amino acid sequences from a CIF file by chain ID.

        Uses gemmi for CIF parsing.
        """
        import gemmi

        doc = gemmi.cif.read(str(cif_path))
        st = gemmi.make_structure_from_block(doc[0])

        result = {}
        model = st[0]
        for chain_id in chain_ids:
            chain = model.find_chain(chain_id)
            if chain is None:
                # Try auth chain ID
                for c in model:
                    if c.subchain and c.name == chain_id:
                        chain = c
                        break
                if chain is None:
                    logger.warning(
                        f"Chain {chain_id} not found in {cif_path}. "
                        f"Available: {[c.name for c in model]}"
                    )
                    continue

            seq = []
            for residue in chain:
                if not residue.is_amino_acid():
                    continue
                one_letter = gemmi.find_tabulated_residue(residue.name).one_letter_code
                seq.append(one_letter if one_letter != "?" else "X")
            result[chain_id] = "".join(seq)

        return result

    def _extract_sequences(
        self, structure_path: Path, chain_ids: list[str]
    ) -> dict[str, str]:
        """Extract sequences from PDB or CIF based on file extension."""
        ext = structure_path.suffix.lower()
        if ext in (".cif", ".mmcif"):
            return self._extract_sequences_from_cif(structure_path, chain_ids)
        else:
            return self._extract_sequences_from_pdb(structure_path, chain_ids)

    def _read_csv(self, csv_path: Path) -> list[dict[str, str]]:
        """Read a CSV file into a list of dicts."""
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _validate_design(self, design: IngestedDesign) -> list[str]:
        """Validate an IngestedDesign. Returns list of warnings."""
        warnings = []
        if not design.complex_structure_path.exists():
            warnings.append(
                f"Structure file does not exist: {design.complex_structure_path}"
            )
        if not design.binder_sequence:
            warnings.append(f"Empty binder sequence for {design.design_id}")
        if not design.target_sequence:
            warnings.append(f"Empty target sequence for {design.design_id}")
        if design.binder_length != len(design.binder_sequence):
            warnings.append(
                f"binder_length ({design.binder_length}) != "
                f"len(binder_sequence) ({len(design.binder_sequence)}) "
                f"for {design.design_id}"
            )
        return warnings
