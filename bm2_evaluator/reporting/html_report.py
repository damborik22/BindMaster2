"""Standalone HTML report generator — no external dependencies.

Produces a self-contained report.html with tier-colored tables,
tool breakdown, and top-20 candidates. Falls back to this when
BM1's binder-compare is not available.

No pandas, no matplotlib — stdlib only.
"""

from __future__ import annotations

import html
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Tier colors
_TIER_COLORS = {
    "consensus_hit": "#2e7d32",  # green
    "strong": "#1565c0",         # blue
    "moderate": "#f57f17",       # amber
    "weak": "#9e9e9e",           # grey
    "fail": "#c62828",           # red
}

_TOOL_COLORS = {
    "bindcraft": "#1565C0",
    "boltzgen": "#E65100",
    "mosaic": "#2E7D32",
    "pxdesign": "#7B1FA2",
    "rfdiffusion": "#C62828",
    "complexa": "#00838F",
}


def generate_html_report(
    scored_designs: list[dict],
    target_info: dict,
    eval_config: dict,
    output_path: Path,
    top_n: int = 20,
) -> str:
    """Generate self-contained HTML report.

    Args:
        scored_designs: Ranked list of design dicts.
        target_info: {"name": str, "chain": str, "n_residues": int}
        eval_config: Config dict.
        output_path: Path to write report.html.
        top_n: Number of top designs to show in detail.

    Returns:
        Path to generated report.
    """
    target_name = target_info.get("name", "unknown")
    target_chain = target_info.get("chain", "?")
    target_nres = target_info.get("n_residues", "?")
    n_designs = len(scored_designs)
    engines = ", ".join(eval_config.get("engines", ["boltz2", "af2"]))

    # Tier counts
    tier_counts = Counter(d.get("tier", "fail") for d in scored_designs)
    tier_order = ["consensus_hit", "strong", "moderate", "weak", "fail"]

    # Tool counts
    tool_counts = Counter(d.get("source_tool", "unknown") for d in scored_designs)

    # Tool x tier breakdown
    tool_tier: dict[str, Counter] = {}
    for d in scored_designs:
        tool = d.get("source_tool", "unknown")
        tier = d.get("tier", "fail")
        tool_tier.setdefault(tool, Counter())[tier] += 1

    # Metric stats
    def _safe_mean(vals: list) -> float:
        clean = [v for v in vals if v is not None and v == v]  # filter NaN
        return sum(clean) / len(clean) if clean else 0.0

    def _safe_float(val, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            f = float(val)
            return f if f == f else default  # NaN check
        except (ValueError, TypeError):
            return default

    ipsae_vals = [_safe_float(d.get("ensemble_ipsae_min")) for d in scored_designs]
    iptm_vals = [_safe_float(d.get("ensemble_iptm")) for d in scored_designs]
    plddt_vals = [_safe_float(d.get("ensemble_plddt")) for d in scored_designs]

    # Top N
    top = scored_designs[:top_n]

    # Build HTML
    parts: list[str] = []
    parts.append(_html_head(target_name))
    parts.append(f"""
<body>
<h1>BM2 Evaluation Report</h1>
<div class="meta">
  <b>Target:</b> {html.escape(str(target_name))} (chain {html.escape(str(target_chain))}, {target_nres} residues)<br>
  <b>Designs evaluated:</b> {n_designs}<br>
  <b>Refolding engines:</b> {html.escape(engines)}<br>
  <b>ipSAE threshold:</b> {eval_config.get('ipsae_consensus_threshold', 0.61)} (Overath 2025)
</div>
""")

    # Tier summary
    parts.append("<h2>Tier Distribution</h2>")
    parts.append(
        '<table class="tier-table"><tr>'
        "<th>Tier</th><th>Count</th><th>%</th><th></th>"
        "</tr>"
    )
    for tier in tier_order:
        count = tier_counts.get(tier, 0)
        pct = (count / n_designs * 100) if n_designs else 0
        color = _TIER_COLORS.get(tier, "#333")
        bar_width = min(pct * 3, 300)
        parts.append(
            f'<tr><td style="color:{color};font-weight:bold">{html.escape(tier)}</td>'
            f'<td class="num">{count}</td>'
            f'<td class="num">{pct:.0f}%</td>'
            f'<td><div style="background:{color};height:16px;'
            f"width:{bar_width}px;border-radius:3px\"></div></td></tr>"
        )
    parts.append("</table>")

    # Legend
    parts.append("""
<p class="legend">
  <b>agreement_count</b>: number of engines with ipSAE_min &gt; 0.61
  (0 = none, 1 = Boltz-2 only, 2 = both Boltz-2 and AF2)
</p>
""")

    # Tool breakdown
    parts.append("<h2>Source Tools</h2>")
    parts.append("<table><tr><th>Tool</th><th>Total</th>")
    for tier in tier_order:
        parts.append(
            f'<th style="color:{_TIER_COLORS.get(tier, "#333")}">'
            f"{html.escape(tier)}</th>"
        )
    parts.append("</tr>")
    for tool in sorted(tool_counts.keys()):
        color = _TOOL_COLORS.get(tool, "#333")
        parts.append(
            f'<tr><td style="color:{color};font-weight:bold">'
            f"{html.escape(tool)}</td>"
        )
        parts.append(f'<td class="num">{tool_counts[tool]}</td>')
        tc = tool_tier.get(tool, Counter())
        for tier in tier_order:
            parts.append(f'<td class="num">{tc.get(tier, 0)}</td>')
        parts.append("</tr>")
    parts.append("</table>")

    # Metric summary
    parts.append("<h2>Metric Summary</h2>")
    parts.append(
        "<table>\n"
        "<tr><th>Metric</th><th>Mean</th><th>Best</th><th>Worst</th></tr>\n"
        f'<tr><td>ipSAE_min</td><td class="num">{_safe_mean(ipsae_vals):.3f}</td>'
        f'<td class="num">{max(ipsae_vals) if ipsae_vals else 0:.3f}</td>'
        f'<td class="num">{min(ipsae_vals) if ipsae_vals else 0:.3f}</td></tr>\n'
        f'<tr><td>ipTM</td><td class="num">{_safe_mean(iptm_vals):.3f}</td>'
        f'<td class="num">{max(iptm_vals) if iptm_vals else 0:.3f}</td>'
        f'<td class="num">{min(iptm_vals) if iptm_vals else 0:.3f}</td></tr>\n'
        f'<tr><td>pLDDT (norm)</td><td class="num">{_safe_mean(plddt_vals):.3f}</td>'
        f'<td class="num">{max(plddt_vals) if plddt_vals else 0:.3f}</td>'
        f'<td class="num">{min(plddt_vals) if plddt_vals else 0:.3f}</td></tr>\n'
        "</table>"
    )

    # Top N table
    parts.append(f"<h2>Top {min(top_n, n_designs)} Designs</h2>")
    parts.append(
        "<table>\n"
        "<tr><th>Rank</th><th>Design ID</th><th>Tool</th><th>ipSAE_min</th>"
        "<th>ipTM</th><th>pLDDT</th><th>Length</th><th>Tier</th>"
        "<th>Sequence</th></tr>"
    )
    for d in top:
        rank = d.get("rank", "")
        did = html.escape(str(d.get("design_id", ""))[:20])
        tool = d.get("source_tool", "")
        tool_color = _TOOL_COLORS.get(tool, "#333")
        ipsae = _safe_float(d.get("ensemble_ipsae_min"))
        iptm = _safe_float(d.get("ensemble_iptm"))
        plddt = _safe_float(d.get("ensemble_plddt"))
        blen = d.get("binder_length", "")
        tier = d.get("tier", "")
        tier_color = _TIER_COLORS.get(tier, "#333")
        seq = d.get("binder_sequence", d.get("sequence", ""))
        seq_str = str(seq) if seq else ""
        seq_short = html.escape(seq_str[:40]) + ("..." if len(seq_str) > 40 else "")

        parts.append(
            f'<tr><td class="num">{rank}</td>'
            f"<td>{did}</td>"
            f'<td style="color:{tool_color};font-weight:bold">'
            f"{html.escape(str(tool))}</td>"
            f'<td class="num">{ipsae:.4f}</td>'
            f'<td class="num">{iptm:.3f}</td>'
            f'<td class="num">{plddt:.3f}</td>'
            f'<td class="num">{blen}</td>'
            f'<td style="color:{tier_color};font-weight:bold">'
            f"{html.escape(str(tier))}</td>"
            f'<td style="font-family:monospace;font-size:0.8em">'
            f"{seq_short}</td></tr>"
        )
    parts.append("</table>")

    # Footer
    parts.append("""
<div class="meta" style="margin-top:2em;font-size:0.85em;color:#666">
  Generated by BM2 Evaluator (standalone mode).<br>
  For full report with plots and PyMOL scripts, install BindMaster 1's
  binder-compare package.
</div>
</body></html>""")

    report_html = "\n".join(parts)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html)
    logger.info(f"HTML report saved: {output_path}")
    return str(output_path)


def _html_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BM2 Report — {html.escape(title)}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 2em; background: #f8f9fa; color: #333; }}
  h1 {{ color: #1a5276; }}
  h2 {{ color: #1a5276; margin-top: 2em; border-bottom: 2px solid #CFE6F6; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: auto; font-size: 0.85em; margin-bottom: 1em; }}
  th {{ background: #1a5276; color: white; padding: 6px 10px; text-align: left; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:nth-child(even) {{ background: #EBF5FB; }}
  tr:hover {{ background: #CFE6F6; }}
  .meta {{ background: #fff; padding: 1em; border: 1px solid #ddd; border-radius: 6px; margin: 1em 0; }}
  .legend {{ font-size: 0.9em; color: #555; margin: 0.5em 0 1em; }}
  .tier-table td:first-child {{ min-width: 120px; }}
</style>
</head>"""
