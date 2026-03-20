"""Multi-engine ranking logic."""

from __future__ import annotations


def compute_multi_model_agreement(
    engine_results: dict[str, dict[str, float]],
    threshold: float = 0.61,
) -> float:
    """Fraction of engines where ipsae_min exceeds threshold.

    1.0 = all engines agree this is a hit.
    0.5 = half agree.
    0.0 = none agree.

    Source: BM2 consensus metric. Threshold default calibrated from
    Overath 2025 (bioRxiv 2025.08.14.670059).
    """
    if not engine_results:
        return 0.0
    passing = sum(
        1 for r in engine_results.values() if r["ipsae_min"] > threshold
    )
    return passing / len(engine_results)


def compute_ensemble_metrics(
    engine_results: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate metrics across engines.

    Rules:
    - ipsae_min: take MINIMUM across engines (conservative = trustworthy)
    - iptm: take MINIMUM across engines (same logic)
    - plddt: take MEAN across engines
    - For ranking: use ensemble_ipsae_min as primary sort key

    Rationale: if one engine says it binds but another doesn't,
    the conservative estimate is more likely correct.
    Cross-model agreement is the whole point of BM2.
    """
    if not engine_results:
        return {
            "ensemble_ipsae_min": 0.0,
            "ensemble_iptm": 0.0,
            "ensemble_plddt": 0.0,
            "best_ipsae_min": 0.0,
            "best_iptm": 0.0,
        }

    ipsae_vals = [r["ipsae_min"] for r in engine_results.values()]
    iptm_vals = [r["iptm"] for r in engine_results.values()]
    plddt_vals = [r["plddt_binder_norm"] for r in engine_results.values()]

    return {
        "ensemble_ipsae_min": min(ipsae_vals),
        "ensemble_iptm": min(iptm_vals),
        "ensemble_plddt": sum(plddt_vals) / len(plddt_vals),
        "best_ipsae_min": max(ipsae_vals),
        "best_iptm": max(iptm_vals),
    }


def rank_designs(
    scored_designs: list[dict],
    sort_key: str = "composite_score",
) -> list[dict]:
    """Rank designs by composite score, break ties by ensemble_ipsae_min.

    Assigns 1-indexed rank. Higher score = better = lower rank number.
    """
    sorted_designs = sorted(
        scored_designs,
        key=lambda d: (d.get(sort_key, 0), d.get("ensemble_ipsae_min", 0)),
        reverse=True,
    )
    for i, d in enumerate(sorted_designs):
        d["rank"] = i + 1
    return sorted_designs
