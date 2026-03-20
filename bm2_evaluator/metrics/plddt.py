"""pLDDT extraction and normalization.

Scale conventions:
    - AF2/ColabFold:  0-100 scale
    - Boltz2:         0-1 scale
    - BindCraft:      0-100 (uses AF2 internally)
    - PXDesign:       ptx_plddt 0-1 (Protenix), af2ig_plddt 0-100 (AF2)

BM2 normalized scale: 0-1 (always).
Both raw and normalized values are stored.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# pLDDT scale lookup by tool/engine name
_PLDDT_SCALES: dict[str, str] = {
    "bindcraft": "0-100",
    "boltzgen": "0-1",
    "pxdesign_ptx": "0-1",
    "pxdesign_af2": "0-100",
    "complexa": "0-100",
    "rfdiffusion": "0-100",
    "generic": "0-100",
    "mosaic": "0-100",
    "af2": "0-100",
    "boltz2": "0-1",
}


def normalize_plddt(
    raw_value: float,
    source_scale: str,
) -> tuple[float, float]:
    """Normalize pLDDT to 0-1 scale.

    Args:
        raw_value: The raw pLDDT value.
        source_scale: One of "0-100" or "0-1".

    Returns:
        (raw_value, normalized_value) tuple.
        Both values always returned for transparency.

    Raises:
        ValueError: If source_scale is unknown or value out of range.
    """
    if source_scale == "0-100":
        if raw_value < 0 or raw_value > 100:
            raise ValueError(f"pLDDT {raw_value} outside 0-100 range")
        return raw_value, raw_value / 100.0
    elif source_scale == "0-1":
        if raw_value < 0 or raw_value > 1.0:
            raise ValueError(f"pLDDT {raw_value} outside 0-1 range")
        return raw_value, raw_value
    else:
        raise ValueError(f"Unknown pLDDT scale: {source_scale}")


def plddt_scale_for_tool(tool_name: str) -> str:
    """Return the pLDDT scale used by a given tool/engine.

    Returns "0-100" or "0-1". Defaults to "0-100" for unknown tools.
    """
    return _PLDDT_SCALES.get(tool_name, "0-100")


def detect_plddt_scale(values: list[float] | np.ndarray) -> str:
    """Auto-detect whether pLDDT values are on 0-1 or 0-100 scale.

    Heuristic: if max(values) <= 1.0, assume 0-1 scale.

    Returns "0-1" or "0-100".
    """
    if len(values) == 0:
        return "0-100"
    max_val = float(np.max(values))
    return "0-1" if max_val <= 1.0 else "0-100"


def extract_plddt_per_chain(
    structure_path: Path,
    chain_ids: list[str],
) -> dict[str, dict[str, float | list[float]]]:
    """Extract per-chain pLDDT statistics from a structure file.

    Reads B-factor column (PDB) or _atom_site.B_iso_or_equiv (CIF),
    which AF2/Boltz2 use to store pLDDT values.

    Args:
        structure_path: Path to PDB or CIF file.
        chain_ids: List of chain IDs to extract.

    Returns:
        {chain_id: {"mean": float, "min": float, "median": float,
                     "per_residue": list[float]}}

    Note: Scale depends on source tool. Caller must normalize.
    """
    structure_path = Path(structure_path)
    ext = structure_path.suffix.lower()

    if ext in (".cif", ".mmcif"):
        return _extract_plddt_from_cif(structure_path, chain_ids)
    else:
        return _extract_plddt_from_pdb(structure_path, chain_ids)


def _extract_plddt_from_pdb(
    pdb_path: Path, chain_ids: list[str]
) -> dict[str, dict[str, float | list[float]]]:
    """Extract pLDDT from PDB B-factor column."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = structure[0]

    result = {}
    for chain_id in chain_ids:
        if chain_id not in [c.id for c in model.get_chains()]:
            logger.warning(f"Chain {chain_id} not found in {pdb_path}")
            continue

        chain = model[chain_id]
        per_residue = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue  # skip heteroatoms
            # Use CA atom B-factor as residue pLDDT
            if "CA" in residue:
                per_residue.append(residue["CA"].get_bfactor())

        if per_residue:
            arr = np.array(per_residue)
            result[chain_id] = {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "median": float(np.median(arr)),
                "per_residue": per_residue,
            }

    return result


def _extract_plddt_from_cif(
    cif_path: Path, chain_ids: list[str]
) -> dict[str, dict[str, float | list[float]]]:
    """Extract pLDDT from CIF B-factor equivalent."""
    import gemmi

    doc = gemmi.cif.read(str(cif_path))
    st = gemmi.make_structure_from_block(doc[0])
    model = st[0]

    result = {}
    for chain_id in chain_ids:
        chain = model.find_chain(chain_id)
        if chain is None:
            logger.warning(f"Chain {chain_id} not found in {cif_path}")
            continue

        per_residue = []
        for residue in chain:
            if not residue.is_amino_acid():
                continue
            # Use CA atom B-factor
            ca = residue.find_atom("CA", "*")
            if ca is not None:
                per_residue.append(ca.b_iso)

        if per_residue:
            arr = np.array(per_residue)
            result[chain_id] = {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "median": float(np.median(arr)),
                "per_residue": per_residue,
            }

    return result
