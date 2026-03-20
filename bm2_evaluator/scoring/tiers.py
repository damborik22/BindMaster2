"""Quality tier classification.

Thresholds are BM2 defaults calibrated from Overath 2025 dataset.
They are NOT hardcoded Overath recommendations — the paper shows
performance varies by target. All thresholds are configurable.

Tier definitions:
    consensus_hit: ALL engines ipsae_min > consensus threshold
    strong:        AT LEAST ONE > consensus, ALL > strong threshold
    moderate:      best ipsae_min > strong threshold AND best iptm > moderate
    weak:          at least one engine passes basic filters (plddt + iptm)
    fail:          nothing passes
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TierThresholds:
    """All thresholds in one place, all configurable, all documented."""

    # BM2 default, calibrated from Overath 2025 (bioRxiv 2025.08.14.670059)
    ipsae_consensus: float = 0.61
    # BM2 default for "strong" tier
    ipsae_strong: float = 0.40
    # Widely used in literature for interface confidence
    iptm_moderate: float = 0.6
    # 0-1 normalized; = 70 on AF2 0-100 scale
    plddt_basic: float = 0.7
    # BindCraft post-hallucination filter
    iptm_basic: float = 0.5


def classify_tier(
    engine_results: dict[str, dict[str, float]],
    thresholds: TierThresholds | None = None,
) -> str:
    """Classify a design into quality tiers based on cross-model refolding.

    Args:
        engine_results: {engine_name: {"ipsae_min": float, "iptm": float,
                         "plddt_binder_norm": float}}
        thresholds: Override default thresholds.

    Returns:
        One of: "consensus_hit", "strong", "moderate", "weak", "fail"
    """
    t = thresholds or TierThresholds()

    if not engine_results:
        return "fail"

    ipsae_values = [r["ipsae_min"] for r in engine_results.values()]
    iptm_values = [r["iptm"] for r in engine_results.values()]
    plddt_values = [r["plddt_binder_norm"] for r in engine_results.values()]

    all_above_consensus = all(v > t.ipsae_consensus for v in ipsae_values)
    any_above_consensus = any(v > t.ipsae_consensus for v in ipsae_values)
    all_above_strong = all(v > t.ipsae_strong for v in ipsae_values)
    best_ipsae = max(ipsae_values)
    best_iptm = max(iptm_values)
    best_plddt = max(plddt_values)

    if all_above_consensus:
        return "consensus_hit"

    if any_above_consensus and all_above_strong:
        return "strong"

    if best_ipsae > t.ipsae_strong and best_iptm > t.iptm_moderate:
        return "moderate"

    if best_plddt > t.plddt_basic and best_iptm > t.iptm_basic:
        return "weak"

    return "fail"
