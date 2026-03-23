#!/usr/bin/env python3
"""Inner evaluator for the Karpathy loss-tuning loop.

Reads Mosaic designs.csv output and computes a composite inner_score
for fast directional feedback. This is NOT the final Evaluator 2.0 —
it uses single-model (Boltz-2) metrics only.

Usage:
    python inner_evaluator.py --input designs.csv [--output summary.json]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _to_float(v) -> float:
    """Convert a value to float, returning NaN for None/empty/unparseable."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def load_designs(csv_path: Path) -> list[dict]:
    """Load designs from CSV, parsing numeric fields.

    Skips malformed rows where essential fields (rank, sequence) are None,
    which happens when semicolon-delimited data is concatenated into a
    comma-delimited CSV.
    """
    designs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip malformed rows (e.g., semicolon-delimited data in comma CSV)
            if row.get("rank") is None or row.get("sequence") is None:
                continue
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            designs.append(parsed)
    return designs


def compute_sequence_diversity(sequences: list[str]) -> float:
    """Compute average pairwise sequence identity, then invert.

    Returns 0.0 if sequences are nearly identical (identity > 0.95),
    1.0 if very diverse (identity < 0.5), linear between.
    """
    if len(sequences) < 2:
        return 0.5  # can't compute diversity from 1 sequence

    # Sample up to 100 pairs for efficiency
    import random
    n = len(sequences)
    if n * (n - 1) // 2 <= 100:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = []
        seen = set()
        while len(pairs) < 100:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and (i, j) not in seen:
                pairs.append((i, j))
                seen.add((i, j))

    identities = []
    for i, j in pairs:
        s1, s2 = sequences[i], sequences[j]
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            continue
        matches = sum(a == b for a, b in zip(s1[:min_len], s2[:min_len]))
        identities.append(matches / min_len)

    if not identities:
        return 0.5

    avg_identity = sum(identities) / len(identities)

    # Linear interpolation: identity 0.95 → diversity 0.0, identity 0.5 → diversity 1.0
    diversity = max(0.0, min(1.0, (0.95 - avg_identity) / 0.45))
    return diversity


def evaluate(designs: list[dict], hit_threshold: float = 0.5) -> dict:
    """Compute inner evaluation metrics.

    Args:
        designs: List of design dicts from CSV.
        hit_threshold: ipSAE threshold for counting hits (default 0.5,
                       intentionally lower than Overath 0.61 for tuning).

    Returns:
        Dict with inner_score and breakdown.
    """
    if not designs:
        return {
            "inner_score": 0.0,
            "hit_rate": 0.0,
            "mean_ipsae_min": 0.0,
            "mean_plddt_binder": 0.0,
            "mean_iptm": 0.0,
            "sequence_diversity": 0.0,
            "n_designs": 0,
            "n_hits": 0,
            "best_ipsae_min": 0.0,
            "worst_ipsae_min": 0.0,
        }

    # Extract metrics — only use rows with is_top=1 (Stage 2 refolded)
    top_designs = [d for d in designs if d.get("is_top", 0) == 1]
    if not top_designs:
        # Fall back to all designs (use aux metrics from Stage 1)
        top_designs = designs

    ipsae_values = []
    plddt_values = []
    iptm_values = []
    sequences = []

    for d in top_designs:
        ipsae = _to_float(d.get("ipsae_min"))
        plddt = _to_float(d.get("plddt_binder_mean"))
        iptm = _to_float(d.get("iptm"))

        # Fall back to aux metrics if prediction metrics are NaN
        if math.isnan(plddt):
            plddt = _to_float(d.get("plddt_aux"))
        if math.isnan(iptm):
            iptm = _to_float(d.get("iptm_aux"))

        if not math.isnan(ipsae):
            ipsae_values.append(ipsae)
        if not math.isnan(plddt):
            plddt_values.append(plddt)
        if not math.isnan(iptm):
            iptm_values.append(iptm)

        seq = d.get("sequence", "")
        if seq:
            sequences.append(seq)

    n_designs = len(top_designs)
    n_hits = sum(1 for v in ipsae_values if v > hit_threshold)
    hit_rate = n_hits / max(n_designs, 1)

    mean_ipsae = sum(ipsae_values) / max(len(ipsae_values), 1) if ipsae_values else 0.0
    mean_plddt = sum(plddt_values) / max(len(plddt_values), 1) if plddt_values else 0.0
    mean_iptm = sum(iptm_values) / max(len(iptm_values), 1) if iptm_values else 0.0
    seq_diversity = compute_sequence_diversity(sequences)

    # Composite inner score (from BM2_AGENT_INSTRUCTIONS Part 2.3)
    inner_score = (
        3.0 * hit_rate
        + 2.0 * mean_ipsae
        + 1.0 * mean_plddt
        + 1.0 * mean_iptm
        + 0.5 * seq_diversity
    )

    best_ipsae = max(ipsae_values) if ipsae_values else 0.0
    worst_ipsae = min(ipsae_values) if ipsae_values else 0.0

    return {
        "inner_score": round(inner_score, 4),
        "hit_rate": round(hit_rate, 4),
        "mean_ipsae_min": round(mean_ipsae, 4),
        "mean_plddt_binder": round(mean_plddt, 4),
        "mean_iptm": round(mean_iptm, 4),
        "sequence_diversity": round(seq_diversity, 4),
        "n_designs": n_designs,
        "n_hits": n_hits,
        "best_ipsae_min": round(best_ipsae, 4),
        "worst_ipsae_min": round(worst_ipsae, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="BM2 Inner Evaluator")
    parser.add_argument("--input", required=True, help="Path to designs.csv")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--hit-threshold", type=float, default=0.5,
                       help="ipSAE threshold for hits (default: 0.5)")
    args = parser.parse_args()

    designs = load_designs(Path(args.input))
    result = evaluate(designs, hit_threshold=args.hit_threshold)

    output = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Summary written to {args.output}")
    else:
        print(output)

    return result


if __name__ == "__main__":
    main()
