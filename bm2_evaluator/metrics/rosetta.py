"""PyRosetta interface scoring. OPTIONAL module.

Returns None for all metrics if PyRosetta not installed.

Reference thresholds (BindCraft defaults):
    dG <= -10 REU, dSASA >= 800 A^2, SC >= 0.55, unsat_hbonds <= 4
    Source: Pacesa et al., BindCraft, Nature 2024
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if PyRosetta is importable."""
    try:
        import pyrosetta  # noqa: F401

        return True
    except ImportError:
        return False


def score_interface(
    pdb_path: str,
    binder_chain: str = "B",
    target_chain: str = "A",
    relax_first: bool = True,
) -> Optional[dict[str, float]]:
    """Score protein-protein interface using PyRosetta.

    Args:
        pdb_path: Path to complex PDB file.
        binder_chain: Binder chain ID.
        target_chain: Target chain ID.
        relax_first: Run FastRelax before scoring (recommended).

    Returns:
        Dict with dG, dSASA, shape_complementarity, n_hbonds, n_clashes,
        dG_per_dSASA. Returns None if PyRosetta not available.

    Metrics:
        dG: Interface binding energy (REU, more negative = better).
            Source: InterfaceAnalyzerMover, REF2015 scorefunction.
        dSASA: Buried surface area (A^2).
        shape_complementarity: Lawrence & Colman SC (0-1, >0.62 good).
        n_hbonds: Interface hydrogen bond count.
        n_clashes: Interface clashes after relaxation.
        dG_per_dSASA: Energy density (REU/A^2).
    """
    if not is_available():
        return None

    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        pyrosetta.init("-mute all -ignore_unrecognized_res")

        pose = pyrosetta.pose_from_pdb(pdb_path)

        if relax_first:
            from pyrosetta.rosetta.protocols.relax import FastRelax

            sfxn = pyrosetta.create_score_function("ref2015")
            relax = FastRelax()
            relax.set_scorefxn(sfxn)
            relax.constrain_relax_to_start_coords(True)
            relax.apply(pose)

        interface_str = f"{target_chain}_{binder_chain}"
        iam = InterfaceAnalyzerMover(interface_str)
        iam.set_pack_separated(True)
        iam.set_compute_interface_sc(True)
        iam.apply(pose)

        dG = iam.get_interface_dG()
        dSASA = iam.get_interface_delta_sasa()

        return {
            "dG": dG,
            "dSASA": dSASA,
            "shape_complementarity": iam.get_interface_sc(),
            "n_hbonds": iam.get_interface_hbonds(),
            "n_clashes": iam.get_clashes(),
            "dG_per_dSASA": dG / max(dSASA, 1.0),
        }

    except Exception as e:
        logger.error(f"PyRosetta scoring failed: {e}")
        return None
