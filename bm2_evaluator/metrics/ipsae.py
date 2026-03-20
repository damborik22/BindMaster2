"""ipSAE computation following Dunbrack 2025.

Source: Dunbrack Lab IPSAE paper and reference implementation
        github.com/DunbrackLab/IPSAE

d0 variant: d0_res (per-source-residue d0 based on qualifying count PER ROW)

Algorithm (d0_res variant, cross-validated against BM1 which matched
DunbrackLab ipsae package v1.0.1):
    For each source residue i:
        1. Find qualifying scored residues j where PAE_ij < cutoff
        2. N_qualifying_i = count of qualifying j for THIS source residue
        3. d0_i = compute_d0_res(N_qualifying_i)
        4. pSAE_ij = 1 / (1 + (PAE_ij / d0_i)^2) for qualifying j only
        5. score_i = mean of pSAE_ij over qualifying j
    ipSAE(source->scored) = max over i of score_i

Conventions:
    ipSAE = max(A->B, B->A)  [Dunbrack convention]
    ipsae_min = min(A->B, B->A)  [Overath "weakest link" for ranking]

PAE cutoff: 10 Angstroms default (Adaptyv convention, uniform for Boltz2/AF2).
    Dunbrack default is 15A. Configurable via EvalConfig.
Threshold: 0.61 BM2 default, calibrated from Overath 2025 dataset.

d0 floor: 1.0 with N floored at 27 (Adaptyv/BM1 convention).
    Cross-validated against DunbrackLab ipsae package v1.0.1.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class IpSAEResult(NamedTuple):
    """Result of ipSAE computation."""

    bt_ipsae: float  # binder->target direction
    tb_ipsae: float  # target->binder direction
    ipsae_max: float  # max(bt, tb) -- Dunbrack convention
    ipsae_min: float  # min(bt, tb) -- Overath "weakest link"
    n_qualifying_bt: int  # max qualifying residues across bt source residues
    n_qualifying_tb: int  # max qualifying residues across tb source residues


def compute_d0_res(n_qualifying: int) -> float:
    """Compute d0 for ipSAE using the d0_res variant.

    For L > 27: d0 = 1.24 * (L - 15)^(1/3) - 1.8
    For L <= 27: d0 = 1.0

    Source: Adaptyv Bio nipah_ipsae_pipeline (github.com/adaptyvbio/nipah_ipsae_pipeline)
    and DunbrackLab IPSAE reference implementation.
    N is floored at 27 (matching Adaptyv calc_d0: if L<27: L=27).
    d0 is floored at 1.0.

    Args:
        n_qualifying: Number of qualifying residues for this source residue.

    Returns:
        d0 value (minimum 1.0).
    """
    n = max(n_qualifying, 27)  # Floor at 27, matching Adaptyv calc_d0
    if n > 27:
        d0 = 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8
    else:
        d0 = 1.0
    return max(1.0, d0)


def _psae_row(pae_row: np.ndarray, cutoff: float) -> tuple[float, int]:
    """Compute per-source-residue ipSAE score.

    For one source residue i, computes mean pSAE over qualifying scored
    residues j where PAE_ij < cutoff.

    This is the per-row algorithm matching BM1 (cross-validated against
    DunbrackLab ipsae v1.0.1).

    Returns:
        (score, n_qualifying)
    """
    qualifying = pae_row[pae_row < cutoff]
    n = len(qualifying)
    if n == 0:
        return 0.0, 0
    d0 = compute_d0_res(n)
    score = float(np.mean(1.0 / (1.0 + (qualifying / d0) ** 2)))
    return score, n


def _compute_directional_ipsae(
    pae_submatrix: np.ndarray,
    cutoff: float = 10.0,
) -> tuple[float, int]:
    """Compute ipSAE in one direction: source -> scored.

    Args:
        pae_submatrix: 2D array of shape (n_source, n_scored).
                       pae_submatrix[i, j] = PAE of residue j when
                       aligned on residue i.
        cutoff: PAE cutoff in Angstroms for qualifying residues.

    Returns:
        (ipsae_directional, max_n_qualifying)

    Algorithm (Dunbrack d0_res, per-row qualifying):
        For each source residue i:
            qualifying_j = {j : PAE_ij < cutoff}
            N_i = |qualifying_j|
            d0_i = compute_d0_res(N_i)
            score_i = mean over qualifying_j of [1/(1 + (PAE_ij/d0_i)^2)]
        ipSAE = max over i of score_i
    """
    n_source, n_scored = pae_submatrix.shape

    if n_source == 0 or n_scored == 0:
        return 0.0, 0

    scores = []
    max_n_qual = 0
    for i in range(n_source):
        score, n_qual = _psae_row(pae_submatrix[i], cutoff)
        scores.append(score)
        max_n_qual = max(max_n_qual, n_qual)

    if not scores:
        return 0.0, 0

    ipsae = float(np.max(scores))
    return ipsae, max_n_qual


def compute_ipsae(
    pae_matrix: np.ndarray,
    binder_slice: slice,
    target_slice: slice,
    cutoff: float = 10.0,
) -> IpSAEResult:
    """Compute ipSAE between binder and target chains.

    Args:
        pae_matrix: Full PAE matrix of shape (N_total, N_total).
        binder_slice: Slice for binder residues in the PAE matrix.
        target_slice: Slice for target residues in the PAE matrix.
        cutoff: PAE cutoff in Angstroms. Default 15.0 (Dunbrack convention).

    Returns:
        IpSAEResult with both directions and aggregate values.
    """
    # Binder->Target: source=binder rows, scored=target cols
    bt_pae = pae_matrix[binder_slice, target_slice]
    bt_ipsae, n_qual_bt = _compute_directional_ipsae(bt_pae, cutoff)

    # Target->Binder: source=target rows, scored=binder cols
    tb_pae = pae_matrix[target_slice, binder_slice]
    tb_ipsae, n_qual_tb = _compute_directional_ipsae(tb_pae, cutoff)

    return IpSAEResult(
        bt_ipsae=bt_ipsae,
        tb_ipsae=tb_ipsae,
        ipsae_max=max(bt_ipsae, tb_ipsae),
        ipsae_min=min(bt_ipsae, tb_ipsae),
        n_qualifying_bt=n_qual_bt,
        n_qualifying_tb=n_qual_tb,
    )
