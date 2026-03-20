"""Cross-tool comparison tables."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def compare_tools(scored_designs: list[dict]) -> str:
    """Generate tool-vs-tool comparison table.

    Returns formatted string:

    Tool         | Designs | Consensus | Strong | Hit Rate | Best ipSAE | Med ipSAE
    BindCraft    |      97 |         5 |     12 |    17.5% |      0.74  |     0.45
    ...

    Hit Rate = (consensus + strong) / total designs * 100
    """
    if not scored_designs:
        return "No designs to compare."

    tools: dict[str, list[dict]] = defaultdict(list)
    for d in scored_designs:
        tool = d.get("source_tool", "unknown")
        tools[tool].append(d)

    lines = []
    header = (
        f"{'Tool':<16}| {'Designs':>7} | {'Consensus':>9} | {'Strong':>6} "
        f"| {'Hit Rate':>8} | {'Best ipSAE':>10} | {'Med ipSAE':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for tool_name in sorted(tools.keys()):
        designs = tools[tool_name]
        total = len(designs)
        n_consensus = sum(
            1 for d in designs if d.get("tier") == "consensus_hit"
        )
        n_strong = sum(1 for d in designs if d.get("tier") == "strong")
        hit_rate = (n_consensus + n_strong) / total * 100 if total > 0 else 0

        ipsae_vals = [
            d.get("ensemble_ipsae_min", 0) for d in designs
        ]
        best_ipsae = max(ipsae_vals) if ipsae_vals else 0
        med_ipsae = float(np.median(ipsae_vals)) if ipsae_vals else 0

        lines.append(
            f"{tool_name:<16}| {total:>7} | {n_consensus:>9} | {n_strong:>6} "
            f"| {hit_rate:>7.1f}% | {best_ipsae:>10.4f} | {med_ipsae:>9.4f}"
        )

    return "\n".join(lines)
