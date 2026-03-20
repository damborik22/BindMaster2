"""Target analysis utilities.

Parse target PDB, compute SASA, detect hotspots, assess difficulty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bm2.core.models import TargetProfile

logger = logging.getLogger(__name__)


def parse_target_pdb(pdb_path: Path, chains: list[str]) -> dict[str, str]:
    """Extract sequences from target PDB by chain ID.

    Uses BioPython PDBParser.

    Returns:
        {chain_id: amino_acid_sequence}
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    model = structure[0]

    sequences = {}
    for chain_id in chains:
        if chain_id not in [c.id for c in model.get_chains()]:
            logger.warning(f"Chain {chain_id} not found in {pdb_path}")
            continue
        chain = model[chain_id]
        seq = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            resname = residue.get_resname().strip()
            code = protein_letters_3to1.get(resname, "X")
            seq.append(code)
        sequences[chain_id] = "".join(seq)

    return sequences


def compute_sasa(
    pdb_path: Path, chain: str
) -> dict[int, float]:
    """Compute per-residue SASA.

    Tries freesasa first (fast C library), falls back to BioPython.

    Returns:
        {residue_number: sasa_value_in_A2}
    """
    try:
        return _compute_sasa_freesasa(pdb_path, chain)
    except ImportError:
        return _compute_sasa_biopython(pdb_path, chain)


def _compute_sasa_freesasa(pdb_path: Path, chain: str) -> dict[int, float]:
    """Compute SASA using freesasa."""
    import freesasa
    from Bio.PDB import PDBParser

    struct = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(struct)

    parser = PDBParser(QUIET=True)
    bio_struct = parser.get_structure("s", str(pdb_path))
    model = bio_struct[0]

    sasa = {}
    if chain not in [c.id for c in model.get_chains()]:
        return sasa

    for residue in model[chain].get_residues():
        if residue.id[0] != " ":
            continue
        resnum = residue.id[1]
        total = 0.0
        for atom in residue.get_atoms():
            selection = freesasa.selectArea(
                {f"a": f"chain {chain} and resi {resnum} and name {atom.name}"},
                struct,
                result,
            )
            total += selection.get("a", 0.0)
        sasa[resnum] = total

    return sasa


def _compute_sasa_biopython(pdb_path: Path, chain: str) -> dict[int, float]:
    """Compute SASA using BioPython ShrakeRupley."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = structure[0]

    sr = ShrakeRupley()
    sr.compute(model, level="R")

    sasa = {}
    if chain not in [c.id for c in model.get_chains()]:
        return sasa

    for residue in model[chain].get_residues():
        if residue.id[0] != " ":
            continue
        sasa[residue.id[1]] = residue.sasa

    return sasa


def auto_detect_hotspots(
    pdb_path: Path,
    chain: str,
    sasa: dict[int, float] | None = None,
    sasa_threshold: float = 40.0,
    min_patch_size: int = 3,
) -> list[str]:
    """Auto-detect potential binding hotspots from SASA.

    Surface-exposed residues (SASA > threshold) in contiguous patches.

    Args:
        pdb_path: Path to PDB file.
        chain: Chain ID.
        sasa: Pre-computed SASA dict. If None, computes it.
        sasa_threshold: Min SASA for a residue to be "surface exposed" (A^2).
        min_patch_size: Min number of contiguous surface residues.

    Returns:
        List of residue IDs like ["A10", "A11", "A25"].
    """
    if sasa is None:
        sasa = compute_sasa(pdb_path, chain)

    # Find surface-exposed residues
    surface = sorted(
        resnum for resnum, val in sasa.items() if val > sasa_threshold
    )

    if not surface:
        return []

    # Group into contiguous patches
    patches: list[list[int]] = []
    current_patch = [surface[0]]

    for i in range(1, len(surface)):
        if surface[i] - surface[i - 1] <= 2:  # allow 1 gap
            current_patch.append(surface[i])
        else:
            if len(current_patch) >= min_patch_size:
                patches.append(current_patch)
            current_patch = [surface[i]]

    if len(current_patch) >= min_patch_size:
        patches.append(current_patch)

    # Select the largest patch
    if not patches:
        # No patches large enough — return top N individual residues
        top_n = min(5, len(surface))
        top_residues = sorted(sasa.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        return [f"{chain}{resnum}" for resnum, _ in top_residues]

    largest = max(patches, key=len)
    return [f"{chain}{resnum}" for resnum in largest]


def assess_difficulty(target: TargetProfile) -> float:
    """Estimate target difficulty on a 0-1 scale.

    Factors:
    - Target length (longer = harder)
    - Binding site accessibility (fewer surface residues = harder)
    - Hotspot coverage

    This is a rough heuristic for tool selection / resource allocation.
    """
    score = 0.0

    # Length factor: 0.0 for < 100aa, 0.3 for 100-300, 0.5 for 300+
    length = target.target_length
    if length > 300:
        score += 0.3
    elif length > 100:
        score += 0.15

    # No hotspots provided → harder (less guidance)
    if not target.hotspot_residues:
        score += 0.2

    # Small binding site → harder
    if target.sasa_per_residue:
        high_sasa = sum(
            1 for v in target.sasa_per_residue.values() if float(v) > 40.0
        )
        frac_exposed = high_sasa / max(len(target.sasa_per_residue), 1)
        if frac_exposed < 0.2:
            score += 0.2  # Mostly buried target

    return min(score, 1.0)
