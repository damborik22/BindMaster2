"""Monomer validation: refold binder alone to check independent folding.

If the binder only folds correctly in the presence of the target
(high monomer RMSD), it is likely a "target-dependent folder" —
intrinsically disordered or misfolded when alone. Such designs are
prone to aggregation and unlikely to function as practical binders.

Threshold: RMSD <= 3.0 Angstroms (BindCraft convention).
Source: Pacesa et al., Nature 2024.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from bm2_evaluator.refolding.base import RefoldingEngine

logger = logging.getLogger(__name__)


@dataclass
class MonomerValidationResult:
    """Result of monomer validation."""

    monomer_rmsd: float  # Ca RMSD between monomer and complex binder (Angstroms)
    monomer_plddt_mean: float  # Normalized 0-1
    passes_validation: bool  # True if RMSD <= threshold
    threshold: float  # RMSD threshold used (default 3.0)
    error: Optional[str] = None


class MonomerValidator:
    """Validate that binder folds independently of target.

    Uses whichever refolding engine is provided to predict the binder
    structure alone, then compares to the binder as it appears in the
    complex prediction via Ca RMSD.
    """

    def __init__(
        self,
        engine: RefoldingEngine,
        rmsd_threshold: float = 3.0,
    ):
        """Initialize validator.

        Args:
            engine: Refolding engine to use for monomer prediction.
            rmsd_threshold: Max Ca RMSD for passing (default 3.0 A).
                            Source: BindCraft Nature 2024 monomer filter.
        """
        self.engine = engine
        self.rmsd_threshold = rmsd_threshold

    def validate(
        self,
        binder_seq: str,
        complex_structure_path: Path,
        binder_chain: str,
        output_dir: Path,
    ) -> MonomerValidationResult:
        """Run monomer validation.

        1. Refold binder sequence alone
        2. Extract binder Ca atoms from complex structure
        3. Extract Ca atoms from monomer prediction
        4. Compute Ca RMSD after optimal superposition
        5. Compare to threshold

        Args:
            binder_seq: Binder amino acid sequence.
            complex_structure_path: Path to complex structure (PDB/CIF).
            binder_chain: Binder chain ID in the complex structure.
            output_dir: Directory for monomer output files.

        Returns:
            MonomerValidationResult with RMSD and pass/fail.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Refold binder alone
        try:
            monomer_result = self.engine.refold_monomer(binder_seq, output_dir)
        except Exception as e:
            logger.error(f"Monomer refolding failed: {e}")
            return MonomerValidationResult(
                monomer_rmsd=float("inf"),
                monomer_plddt_mean=0.0,
                passes_validation=False,
                threshold=self.rmsd_threshold,
                error=f"Monomer refolding failed: {e}",
            )

        if not monomer_result.success:
            return MonomerValidationResult(
                monomer_rmsd=float("inf"),
                monomer_plddt_mean=0.0,
                passes_validation=False,
                threshold=self.rmsd_threshold,
                error=monomer_result.error,
            )

        # Normalize pLDDT to 0-1
        plddt_norm = monomer_result.plddt_mean
        if monomer_result.plddt_scale_max > 1.0:
            plddt_norm = monomer_result.plddt_mean / monomer_result.plddt_scale_max

        # Step 2-4: Compute Ca RMSD
        try:
            rmsd = compute_ca_rmsd(
                structure1=complex_structure_path,
                chain1=binder_chain,
                structure2=monomer_result.structure_path,
                chain2=None,  # Monomer has one chain, auto-detect
            )
        except Exception as e:
            logger.error(f"RMSD computation failed: {e}")
            return MonomerValidationResult(
                monomer_rmsd=float("inf"),
                monomer_plddt_mean=plddt_norm,
                passes_validation=False,
                threshold=self.rmsd_threshold,
                error=f"RMSD computation failed: {e}",
            )

        passes = rmsd <= self.rmsd_threshold
        logger.info(
            f"Monomer validation: RMSD={rmsd:.2f}A "
            f"(threshold={self.rmsd_threshold}A) -> "
            f"{'PASS' if passes else 'FAIL'}"
        )

        return MonomerValidationResult(
            monomer_rmsd=rmsd,
            monomer_plddt_mean=plddt_norm,
            passes_validation=passes,
            threshold=self.rmsd_threshold,
        )


def compute_ca_rmsd(
    structure1: Path,
    chain1: str,
    structure2: Path,
    chain2: Optional[str] = None,
) -> float:
    """Compute Ca RMSD between two structures after optimal superposition.

    Uses BioPython's SVDSuperimposer for Kabsch alignment.

    Args:
        structure1: Path to first structure (PDB/CIF).
        chain1: Chain ID in first structure.
        structure2: Path to second structure (PDB/CIF).
        chain2: Chain ID in second structure (None = first chain found).

    Returns:
        Ca RMSD in Angstroms after optimal superposition.
    """
    coords1 = _extract_ca_coords(structure1, chain1)
    coords2 = _extract_ca_coords(structure2, chain2)

    # Use the shorter length (structures may differ slightly)
    n = min(len(coords1), len(coords2))
    if n == 0:
        raise ValueError("No Ca atoms found for RMSD computation")

    coords1 = coords1[:n]
    coords2 = coords2[:n]

    # Kabsch superposition
    from Bio.SVDSuperimposer import SVDSuperimposer

    sup = SVDSuperimposer()
    sup.set(coords1, coords2)
    sup.run()

    return float(sup.get_rms())


def _extract_ca_coords(
    structure_path: Path, chain_id: Optional[str]
) -> np.ndarray:
    """Extract Ca atom coordinates from a structure file.

    Args:
        structure_path: Path to PDB or CIF file.
        chain_id: Chain to extract. None = first protein chain.

    Returns:
        Nx3 numpy array of Ca coordinates.
    """
    structure_path = Path(structure_path)
    ext = structure_path.suffix.lower()

    if ext in (".cif", ".mmcif"):
        return _extract_ca_from_cif(structure_path, chain_id)
    else:
        return _extract_ca_from_pdb(structure_path, chain_id)


def _extract_ca_from_pdb(path: Path, chain_id: Optional[str]) -> np.ndarray:
    """Extract Ca coords from PDB."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(path))
    model = structure[0]

    if chain_id is None:
        chain = next(model.get_chains())
    else:
        chain = model[chain_id]

    coords = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        if "CA" in residue:
            coords.append(residue["CA"].get_vector().get_array())

    return np.array(coords)


def _extract_ca_from_cif(path: Path, chain_id: Optional[str]) -> np.ndarray:
    """Extract Ca coords from CIF."""
    import gemmi

    doc = gemmi.cif.read(str(path))
    st = gemmi.make_structure_from_block(doc[0])
    model = st[0]

    if chain_id is None:
        chain = model[0]
    else:
        chain = model.find_chain(chain_id)
        if chain is None:
            chain = model[0]

    coords = []
    for residue in chain:
        if not residue.is_amino_acid():
            continue
        ca = residue.find_atom("CA", "*")
        if ca is not None:
            coords.append([ca.pos.x, ca.pos.y, ca.pos.z])

    return np.array(coords)
