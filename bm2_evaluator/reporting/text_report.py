"""Human-readable evaluation report."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_report(
    scored_designs: list[dict],
    target_info: dict,
    eval_config: dict,
    output_path: Path,
    top_n: int = 20,
) -> str:
    """Generate text report and save to file.

    Args:
        scored_designs: Ranked list of design dicts.
        target_info: {"name": str, "chain": str, "n_residues": int}
        eval_config: Config dict for documenting parameters.
        output_path: Path to save report.
        top_n: Number of top designs to show.

    Returns:
        Report text (also saved to file).
    """
    lines = []
    sep = "=" * 55

    lines.append(sep)
    lines.append("BM2 Evaluator — Campaign Report")
    lines.append(sep)
    lines.append("")

    # Target info
    name = target_info.get("name", "unknown")
    chain = target_info.get("chain", "?")
    n_res = target_info.get("n_residues", "?")
    lines.append(f"Target: {name} (chain {chain}, {n_res} residues)")
    lines.append(f"Designs evaluated: {len(scored_designs)}")

    # Count tools
    tool_counts = Counter(d.get("source_tool", "unknown") for d in scored_designs)
    lines.append(f"Source tools: {len(tool_counts)}")

    # Config
    engines = eval_config.get("engines", ["boltz2", "af2"])
    lines.append(f"Refolding engines: {', '.join(engines)}")
    lines.append(
        f"PAE cutoff: {eval_config.get('pae_cutoff', 15.0)} A "
        f"(Dunbrack 2025 default)"
    )
    lines.append(
        f"ipSAE threshold: {eval_config.get('ipsae_consensus_threshold', 0.61)} "
        f"(BM2 default, calibrated from Overath 2025)"
    )
    rosetta = eval_config.get("use_rosetta", False)
    lines.append(f"Rosetta scoring: {'yes' if rosetta else 'no'}")
    lines.append("")

    # Tier distribution
    tier_counts = Counter(d.get("tier", "fail") for d in scored_designs)
    tier_order = ["consensus_hit", "strong", "moderate", "weak", "fail"]
    lines.append("Tier Distribution:")
    for tier in tier_order:
        count = tier_counts.get(tier, 0)
        label = f"  {tier}:"
        suffix = "  <-  TEST THESE FIRST" if tier == "consensus_hit" and count > 0 else ""
        lines.append(f"{label:<22}{count:>4}{suffix}")
    lines.append("")

    # Tier & metric legend
    lines.append("Legend:")
    lines.append(
        "  agreement_count: number of engines with ipSAE_min > 0.61 "
        "(0 = none, 1 = Boltz-2 only, 2 = both Boltz-2 and AF2)"
    )
    lines.append("")

    # Source tool breakdown
    lines.append("Source Tool Breakdown:")
    for tool, total in sorted(tool_counts.items()):
        tool_designs = [d for d in scored_designs if d.get("source_tool") == tool]
        n_consensus = sum(1 for d in tool_designs if d.get("tier") == "consensus_hit")
        n_strong = sum(1 for d in tool_designs if d.get("tier") == "strong")
        lines.append(
            f"  {tool:<20}{total:>4} designs "
            f"({n_consensus} consensus, {n_strong} strong)"
        )
    lines.append("")

    # Top N designs
    top = scored_designs[:top_n]
    if top:
        lines.append(f"Top {min(top_n, len(scored_designs))} Designs:")
        header = f"{'Rank':>4}  {'ID':<16}{'Source':<14}{'Score':>7}  {'ipSAE':>7}  {'ipTM':>6}  {'Tier':<15}"
        lines.append(header)
        for d in top:
            rank = d.get("rank", "")
            did = str(d.get("design_id", ""))[:15]
            src = str(d.get("source_tool", ""))[:13]
            score = d.get("composite_score", 0)
            ipsae = d.get("ensemble_ipsae_min", 0)
            iptm = d.get("ensemble_iptm", 0)
            tier = d.get("tier", "")
            lines.append(
                f"{rank:>4}  {did:<16}{src:<14}{score:>7.4f}  {ipsae:>7.4f}  {iptm:>6.3f}  {tier:<15}"
            )
    lines.append("")

    # Metric distributions
    if scored_designs:
        ipsae_vals = [
            d.get("ensemble_ipsae_min", 0) for d in scored_designs
        ]
        iptm_vals = [d.get("ensemble_iptm", 0) for d in scored_designs]
        agreement_vals = [
            d.get("multi_model_agreement", 0) for d in scored_designs
        ]

        lines.append("Metric Distributions (all designs):")
        lines.append(
            f"  ipSAE_min:  mean={np.mean(ipsae_vals):.3f}  "
            f"median={np.median(ipsae_vals):.3f}  "
            f"std={np.std(ipsae_vals):.3f}"
        )
        lines.append(
            f"  ipTM:       mean={np.mean(iptm_vals):.3f}  "
            f"median={np.median(iptm_vals):.3f}  "
            f"std={np.std(iptm_vals):.3f}"
        )
        lines.append(f"  Agreement:  mean={np.mean(agreement_vals):.2f}")
    lines.append("")

    # Files
    lines.append("Files:")
    lines.append("  evaluation_summary.csv — one row per design, raw + normalized")
    lines.append("  evaluation_detail.csv  — one row per engine per design")
    lines.append("  pae/                   — all PAE .npy files")
    lines.append("  structures/            — all refolded structures")
    lines.append("")

    # References
    lines.append("References:")
    lines.append(
        "  ipSAE: Dunbrack 2025, bioRxiv 2025.02.10.637595, Eq 14-16"
    )
    lines.append(
        "  Thresholds: calibrated from Overath 2025, "
        "bioRxiv 2025.08.14.670059"
    )
    cutoff = eval_config.get("pae_cutoff", 15.0)
    lines.append(
        f"  d0 variant: d0_res (per-residue count with PAE < {cutoff} A)"
    )
    lines.append(sep)

    report = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    logger.info(f"Report saved: {output_path}")
    return report
