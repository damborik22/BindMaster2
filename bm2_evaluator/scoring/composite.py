"""Composite scoring functions.

Every formula cites its source. Every threshold is configurable.
"""

from __future__ import annotations


def composite_with_rosetta(
    ipsae_min: float,
    dG: float,
    dSASA: float,
) -> float:
    """Best single composite per Overath 2025 meta-analysis.

    Formula: ipsae_min * |dG / dSASA|

    Source: Overath et al. 2025, bioRxiv 2025.08.14.670059
            "combining [ipSAE] metrics with orthogonal physicochemical
             interface descriptors, including Rosetta dG/dSASA and
             interface shape complementarity, improves predictive performance"

    Args:
        ipsae_min: min(bt_ipsae, tb_ipsae), 0-1 scale.
        dG: Rosetta interface dG (REU, typically negative for binders).
        dSASA: Buried surface area (A^2, typically > 0).

    Returns:
        Composite score (higher = better binder candidate).
    """
    if dSASA == 0:
        return 0.0
    return ipsae_min * abs(dG / dSASA)


def composite_basic(
    ipsae_min: float,
    iptm: float,
    agreement: float,
    plddt_binder: float,
    pae_interaction_mean: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Fallback composite when Rosetta is not available.

    Default weights calibrated from Overath 2025 feature importance:
    ipSAE_min is the dominant predictor, others are supplementary.

    Source: BM2 default weights, informed by Overath 2025 feature ranking.
    These are NOT published Overath weights — they are our defaults.
    Users should tune per campaign.

    Args:
        ipsae_min: min(bt_ipsae, tb_ipsae), 0-1 normalized.
        iptm: Best engine ipTM, 0-1.
        agreement: Multi-model agreement fraction, 0-1.
        plddt_binder: Binder mean pLDDT, 0-1 normalized.
        pae_interaction_mean: Mean interaction PAE in Angstroms.
        weights: Override default weights dict.

    Returns:
        Composite score (higher = better).
    """
    w = weights or {
        "ipsae_min": 0.40,
        "iptm": 0.25,
        "agreement": 0.15,
        "plddt": 0.10,
        "pae_inv": 0.10,
    }
    pae_score = 1.0 - min(pae_interaction_mean / 30.0, 1.0)

    return (
        w["ipsae_min"] * ipsae_min
        + w["iptm"] * iptm
        + w["agreement"] * agreement
        + w["plddt"] * plddt_binder
        + w["pae_inv"] * pae_score
    )
