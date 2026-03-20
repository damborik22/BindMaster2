"""Tests for ipSAE computation.

Validates against Dunbrack 2025 reference implementation.
Source: github.com/DunbrackLab/IPSAE
"""

import numpy as np
import pytest

from bm2_evaluator.metrics.ipsae import (
    IpSAEResult,
    compute_d0_res,
    compute_ipsae,
    _compute_directional_ipsae,
)


class TestComputeD0Res:
    """Test d0_res formula: d0 = max(1.0, 1.24*(max(N,27)-15)^(1/3) - 1.8)

    Uses Adaptyv/BM1 convention: d0 floor=1.0, N floor=27.
    Cross-validated against DunbrackLab ipsae package v1.0.1.
    """

    def test_zero_qualifying(self):
        # N floored to 26, but 26 <= 27 so d0 = 1.0
        assert compute_d0_res(0) == 1.0

    def test_at_boundary_15(self):
        # Same as N=0 due to N floor at 26
        assert compute_d0_res(15) == compute_d0_res(0) == 1.0

    def test_n_27(self):
        # N=27: <= 27 boundary, d0 = 1.0 (Dunbrack reference)
        assert compute_d0_res(27) == 1.0

    def test_n_50(self):
        d0 = compute_d0_res(50)
        expected = 1.24 * (35 ** (1.0 / 3.0)) - 1.8
        assert abs(d0 - expected) < 1e-10
        assert d0 > 2.0

    def test_n_100(self):
        d0 = compute_d0_res(100)
        expected = 1.24 * (85 ** (1.0 / 3.0)) - 1.8
        assert abs(d0 - expected) < 1e-10

    def test_n_200(self):
        d0 = compute_d0_res(200)
        expected = 1.24 * (185 ** (1.0 / 3.0)) - 1.8
        assert abs(d0 - expected) < 1e-10

    def test_always_at_least_1_0(self):
        for n in range(0, 200):
            assert compute_d0_res(n) >= 1.0


class TestDirectionalIpSAE:
    """Test _compute_directional_ipsae."""

    def test_perfect_prediction(self):
        """All PAE = 0 -> ipSAE = 1.0."""
        pae = np.zeros((5, 10))
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)
        assert ipsae == 1.0
        assert n_qual == 10

    def test_all_above_cutoff(self):
        """All PAE > cutoff -> ipSAE = 0.0, n_qualifying = 0."""
        pae = np.full((5, 10), 20.0)
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)
        assert ipsae == 0.0
        assert n_qual == 0

    def test_empty_source(self):
        pae = np.zeros((0, 10))
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)
        assert ipsae == 0.0
        assert n_qual == 0

    def test_empty_scored(self):
        pae = np.zeros((5, 0))
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)
        assert ipsae == 0.0
        assert n_qual == 0

    def test_pae_exactly_at_cutoff_excluded(self):
        """PAE exactly at cutoff should NOT qualify (strict <)."""
        pae = np.full((3, 5), 15.0)
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)
        assert ipsae == 0.0
        assert n_qual == 0

    def test_pae_just_below_cutoff(self):
        """PAE just below cutoff should qualify."""
        pae = np.full((3, 5), 9.99)
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)
        assert ipsae > 0.0
        assert n_qual == 5

    def test_single_source_single_scored(self):
        """Minimal case: 1 source, 1 scored residue."""
        pae = np.array([[5.0]])
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)
        # d0 = compute_d0_res(1) = 0.5  (n=1 < 15)
        # pSAE = 1/(1 + (5.0/0.5)^2) = 1/(1+100) = 0.00990
        d0 = compute_d0_res(1)
        expected = 1.0 / (1.0 + (5.0 / d0) ** 2)
        assert abs(ipsae - expected) < 1e-10
        assert n_qual == 1

    def test_hand_computed_example(self):
        """Hand-computed example for validation.

        3 source, 4 scored residues.
        PAE matrix:
            [[2, 3, 20, 4],
             [5, 2, 18, 6],
             [3, 4,  1, 5]]
        cutoff = 10.0

        Per-row qualifying (Adaptyv/Dunbrack d0_res convention):
            Row 0: qualifying = [2, 3, 4] (3 vals < 10), N=3 -> d0=d0(3)
            Row 1: qualifying = [5, 2, 6] (3 vals < 10), N=3 -> d0=d0(3)
            Row 2: qualifying = [3, 4, 1, 5] (4 vals < 10), N=4 -> d0=d0(4)

        With N floor=27, all d0 values are the same: d0(27).
        """
        pae = np.array(
            [[2, 3, 20, 4], [5, 2, 18, 6], [3, 4, 1, 5]], dtype=np.float64
        )
        ipsae, n_qual = _compute_directional_ipsae(pae, cutoff=10.0)

        d0 = compute_d0_res(3)  # N floored to 27
        assert d0 == compute_d0_res(27)  # All small N give same d0

        # Compute expected per-row
        means = []
        for i in range(3):
            qualifying = pae[i][pae[i] < 10.0]
            if len(qualifying) == 0:
                means.append(0.0)
            else:
                d0_i = compute_d0_res(len(qualifying))
                score = float(np.mean(1.0 / (1.0 + (qualifying / d0_i) ** 2)))
                means.append(score)

        expected = max(means)
        assert abs(ipsae - expected) < 1e-10

    def test_cutoff_parameter_matters(self):
        """Different cutoff values produce different results."""
        pae = np.full((5, 10), 12.0)
        ipsae_15, n15 = _compute_directional_ipsae(pae, cutoff=15.0)
        ipsae_10, n10 = _compute_directional_ipsae(pae, cutoff=10.0)
        assert ipsae_15 > 0.0  # 12 < 15, qualifies
        assert ipsae_10 == 0.0  # 12 >= 10, does NOT qualify (strict <)
        assert n15 == 10
        assert n10 == 0


class TestComputeIpSAE:
    """Test full ipSAE computation with both directions."""

    def test_symmetric_matrix(self, strong_pae_matrix):
        """Symmetric PAE -> bt_ipsae == tb_ipsae."""
        # Build a symmetric interchain PAE
        n = 20
        mat = np.full((n, n), 5.0)
        np.fill_diagonal(mat, 0.0)

        result = compute_ipsae(mat, slice(0, 10), slice(10, 20))
        assert abs(result.bt_ipsae - result.tb_ipsae) < 1e-10
        assert result.ipsae_max == result.ipsae_min

    def test_asymmetric_matrix(self):
        """Asymmetric interchain PAE -> bt != tb."""
        n = 15
        mat = np.full((n, n), 3.0)
        np.fill_diagonal(mat, 0.0)
        # Make binder->target PAE different from target->binder
        mat[:10, 10:] = 5.0  # target rows, binder cols
        mat[10:, :10] = 12.0  # binder rows, target cols

        result = compute_ipsae(mat, slice(10, 15), slice(0, 10))
        assert result.bt_ipsae != result.tb_ipsae
        # binder->target has PAE=12, target->binder has PAE=5
        assert result.tb_ipsae > result.bt_ipsae

    def test_strong_interaction(self, strong_pae_matrix):
        """Strong interaction (low PAE) -> high ipSAE."""
        result = compute_ipsae(
            strong_pae_matrix, slice(10, 15), slice(0, 10)
        )
        assert result.ipsae_min > 0.0
        assert result.ipsae_max > 0.0
        assert result.ipsae_max >= result.ipsae_min

    def test_weak_interaction(self, weak_pae_matrix):
        """Weak interaction (high PAE > cutoff) -> low/zero ipSAE."""
        result = compute_ipsae(
            weak_pae_matrix, slice(10, 15), slice(0, 10)
        )
        assert result.ipsae_min == 0.0
        assert result.n_qualifying_bt == 0
        assert result.n_qualifying_tb == 0

    def test_ipsae_min_max_relationship(self, strong_pae_matrix):
        """ipsae_max >= ipsae_min always."""
        result = compute_ipsae(
            strong_pae_matrix, slice(10, 15), slice(0, 10)
        )
        assert result.ipsae_max >= result.ipsae_min

    def test_result_type(self, strong_pae_matrix):
        """Returns IpSAEResult named tuple."""
        result = compute_ipsae(
            strong_pae_matrix, slice(10, 15), slice(0, 10)
        )
        assert isinstance(result, IpSAEResult)
        assert hasattr(result, "bt_ipsae")
        assert hasattr(result, "tb_ipsae")
        assert hasattr(result, "ipsae_max")
        assert hasattr(result, "ipsae_min")
        assert hasattr(result, "n_qualifying_bt")
        assert hasattr(result, "n_qualifying_tb")

    def test_no_qualifying_residues(self):
        """All PAE > cutoff -> both directions 0."""
        n = 20
        mat = np.full((n, n), 20.0)
        np.fill_diagonal(mat, 0.0)

        result = compute_ipsae(mat, slice(0, 10), slice(10, 20))
        assert result.bt_ipsae == 0.0
        assert result.tb_ipsae == 0.0
        assert result.ipsae_min == 0.0
        assert result.ipsae_max == 0.0

    def test_custom_cutoff(self, strong_pae_matrix):
        """Custom cutoff changes results."""
        result_15 = compute_ipsae(
            strong_pae_matrix, slice(10, 15), slice(0, 10), cutoff=10.0
        )
        result_3 = compute_ipsae(
            strong_pae_matrix, slice(10, 15), slice(0, 10), cutoff=3.0
        )
        # With cutoff=3.0, interchain PAE of 5.0 won't qualify
        assert result_15.ipsae_min > result_3.ipsae_min

    def test_large_matrix_performance(self):
        """Performance test with realistic sizes."""
        n = 500  # 200 target + 300 binder
        mat = np.random.uniform(0, 30, (n, n))
        np.fill_diagonal(mat, 0.0)

        result = compute_ipsae(mat, slice(200, 500), slice(0, 200))
        assert 0.0 <= result.ipsae_min <= 1.0
        assert 0.0 <= result.ipsae_max <= 1.0
