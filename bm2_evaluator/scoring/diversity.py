"""Structural diversity clustering.

Ensures the final selection has diverse scaffolds,
not 20 variants of the same fold.
"""

from __future__ import annotations


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute fractional sequence identity between two sequences.

    Uses the shorter sequence length as denominator.
    Only counts exact matches at aligned positions (no gaps).
    """
    n = min(len(seq1), len(seq2))
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq1[:n], seq2[:n]) if a == b)
    return matches / n


def cluster_by_sequence(
    designs: list[dict],
    identity_threshold: float = 0.7,
    sequence_key: str = "binder_sequence",
    id_key: str = "design_id",
) -> list[list[str]]:
    """Simple sequence-based clustering using pairwise identity.

    Greedy clustering: each design is assigned to the first cluster
    where identity > threshold. If no match, starts a new cluster.

    Args:
        designs: List of design dicts with sequence and id fields.
        identity_threshold: Min identity to join a cluster (0-1).
        sequence_key: Key for the sequence in each dict.
        id_key: Key for the design ID in each dict.

    Returns:
        List of clusters, each cluster is a list of design_ids.
    """
    clusters: list[list[str]] = []
    cluster_reps: list[str] = []  # representative sequence per cluster

    for design in designs:
        seq = design[sequence_key]
        design_id = design[id_key]

        assigned = False
        for i, rep_seq in enumerate(cluster_reps):
            if compute_sequence_identity(seq, rep_seq) > identity_threshold:
                clusters[i].append(design_id)
                assigned = True
                break

        if not assigned:
            clusters.append([design_id])
            cluster_reps.append(seq)

    return clusters


def select_diverse_representatives(
    ranked_designs: list[dict],
    clusters: list[list[str]],
    max_per_cluster: int = 3,
    id_key: str = "design_id",
) -> list[dict]:
    """From each cluster, take up to max_per_cluster designs (by rank).

    Ensures scaffold diversity in final selection.

    Args:
        ranked_designs: Designs sorted by rank (best first).
        clusters: Output from cluster_by_sequence.
        max_per_cluster: Max designs to select per cluster.
        id_key: Key for design ID.

    Returns:
        Filtered list of designs with diverse scaffolds.
    """
    # Build cluster membership lookup
    cluster_map: dict[str, int] = {}
    for i, cluster in enumerate(clusters):
        for design_id in cluster:
            cluster_map[design_id] = i

    # Count selections per cluster
    cluster_counts: dict[int, int] = {}
    selected = []

    for design in ranked_designs:
        design_id = design[id_key]
        cluster_idx = cluster_map.get(design_id)

        if cluster_idx is None:
            # Design not in any cluster — include it
            selected.append(design)
            continue

        count = cluster_counts.get(cluster_idx, 0)
        if count < max_per_cluster:
            selected.append(design)
            cluster_counts[cluster_idx] = count + 1

    return selected
