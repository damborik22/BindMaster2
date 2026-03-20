"""Tests for scoring, tiers, ranking, diversity, rosetta, reporting, CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bm2_evaluator.scoring.composite import composite_basic, composite_with_rosetta
from bm2_evaluator.scoring.tiers import TierThresholds, classify_tier
from bm2_evaluator.scoring.ranking import (
    compute_ensemble_metrics,
    compute_multi_model_agreement,
    rank_designs,
)
from bm2_evaluator.scoring.diversity import (
    cluster_by_sequence,
    compute_sequence_identity,
    select_diverse_representatives,
)
from bm2_evaluator.metrics.rosetta import is_available as rosetta_available
from bm2_evaluator.reporting.csv_export import export_detail_csv, export_summary_csv
from bm2_evaluator.reporting.text_report import generate_report
from bm2_evaluator.reporting.comparison import compare_tools


# ---------------------------------------------------------------------------
# Composite scoring tests
# ---------------------------------------------------------------------------


class TestCompositeWithRosetta:
    def test_known_values(self):
        # 0.7 * |(-15) / 1000| = 0.7 * 0.015 = 0.0105
        score = composite_with_rosetta(0.7, -15.0, 1000.0)
        assert abs(score - 0.0105) < 1e-10

    def test_zero_dsasa_guard(self):
        score = composite_with_rosetta(0.7, -15.0, 0.0)
        assert score == 0.0

    def test_positive_dg(self):
        # Even with positive dG (unfavorable), formula still works
        score = composite_with_rosetta(0.5, 5.0, 1000.0)
        assert score == 0.5 * abs(5.0 / 1000.0)

    def test_high_ipsae_high_dg_ratio(self):
        score = composite_with_rosetta(0.9, -30.0, 500.0)
        assert score == 0.9 * abs(-30.0 / 500.0)


class TestCompositeBasic:
    def test_default_weights(self):
        score = composite_basic(
            ipsae_min=0.7,
            iptm=0.8,
            agreement=1.0,
            plddt_binder=0.85,
            pae_interaction_mean=5.0,
        )
        # 0.40*0.7 + 0.25*0.8 + 0.15*1.0 + 0.10*0.85 + 0.10*(1 - 5/30)
        pae_score = 1.0 - 5.0 / 30.0
        expected = 0.40 * 0.7 + 0.25 * 0.8 + 0.15 * 1.0 + 0.10 * 0.85 + 0.10 * pae_score
        assert abs(score - expected) < 1e-10

    def test_custom_weights(self):
        weights = {
            "ipsae_min": 1.0,
            "iptm": 0.0,
            "agreement": 0.0,
            "plddt": 0.0,
            "pae_inv": 0.0,
        }
        score = composite_basic(0.65, 0.8, 1.0, 0.9, 5.0, weights=weights)
        assert abs(score - 0.65) < 1e-10

    def test_high_pae_caps_at_1(self):
        # PAE >= 30 should give pae_score = 0
        score = composite_basic(0.5, 0.5, 0.5, 0.5, 35.0)
        pae_score = 0.0  # min(35/30, 1) = 1.0, so 1 - 1 = 0
        expected = 0.40 * 0.5 + 0.25 * 0.5 + 0.15 * 0.5 + 0.10 * 0.5 + 0.10 * 0.0
        assert abs(score - expected) < 1e-10


# ---------------------------------------------------------------------------
# Tier classification tests
# ---------------------------------------------------------------------------


class TestClassifyTier:
    def _make_er(self, ipsae_min, iptm, plddt_norm):
        return {"ipsae_min": ipsae_min, "iptm": iptm, "plddt_binder_norm": plddt_norm}

    def test_consensus_hit(self):
        er = {
            "boltz2": self._make_er(0.65, 0.8, 0.9),
            "af2": self._make_er(0.70, 0.85, 0.88),
        }
        assert classify_tier(er) == "consensus_hit"

    def test_strong(self):
        er = {
            "boltz2": self._make_er(0.65, 0.8, 0.9),
            "af2": self._make_er(0.45, 0.7, 0.85),
        }
        assert classify_tier(er) == "strong"

    def test_moderate(self):
        er = {
            "boltz2": self._make_er(0.50, 0.7, 0.85),
            "af2": self._make_er(0.35, 0.6, 0.80),
        }
        assert classify_tier(er) == "moderate"

    def test_weak(self):
        er = {
            "boltz2": self._make_er(0.30, 0.55, 0.80),
            "af2": self._make_er(0.20, 0.52, 0.75),
        }
        assert classify_tier(er) == "weak"

    def test_fail(self):
        er = {
            "boltz2": self._make_er(0.10, 0.3, 0.5),
            "af2": self._make_er(0.05, 0.2, 0.4),
        }
        assert classify_tier(er) == "fail"

    def test_empty_results(self):
        assert classify_tier({}) == "fail"

    def test_single_engine(self):
        er = {"boltz2": self._make_er(0.65, 0.8, 0.9)}
        assert classify_tier(er) == "consensus_hit"

    def test_custom_thresholds(self):
        t = TierThresholds(ipsae_consensus=0.80)
        er = {
            "boltz2": self._make_er(0.70, 0.8, 0.9),
            "af2": self._make_er(0.65, 0.7, 0.85),
        }
        # With higher threshold, 0.70 < 0.80, so not consensus
        assert classify_tier(er, t) != "consensus_hit"


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------


class TestMultiModelAgreement:
    def test_all_passing(self):
        er = {
            "boltz2": {"ipsae_min": 0.70},
            "af2": {"ipsae_min": 0.65},
        }
        assert compute_multi_model_agreement(er, 0.61) == 1.0

    def test_half_passing(self):
        er = {
            "boltz2": {"ipsae_min": 0.70},
            "af2": {"ipsae_min": 0.50},
        }
        assert compute_multi_model_agreement(er, 0.61) == 0.5

    def test_none_passing(self):
        er = {
            "boltz2": {"ipsae_min": 0.30},
            "af2": {"ipsae_min": 0.20},
        }
        assert compute_multi_model_agreement(er, 0.61) == 0.0

    def test_empty(self):
        assert compute_multi_model_agreement({}, 0.61) == 0.0


class TestEnsembleMetrics:
    def test_takes_min_ipsae_min_iptm(self):
        er = {
            "boltz2": {"ipsae_min": 0.70, "iptm": 0.80, "plddt_binder_norm": 0.90},
            "af2": {"ipsae_min": 0.50, "iptm": 0.75, "plddt_binder_norm": 0.85},
        }
        result = compute_ensemble_metrics(er)
        assert result["ensemble_ipsae_min"] == 0.50  # min
        assert result["ensemble_iptm"] == 0.75  # min
        assert abs(result["ensemble_plddt"] - 0.875) < 1e-10  # mean

    def test_best_values(self):
        er = {
            "boltz2": {"ipsae_min": 0.70, "iptm": 0.80, "plddt_binder_norm": 0.90},
            "af2": {"ipsae_min": 0.50, "iptm": 0.75, "plddt_binder_norm": 0.85},
        }
        result = compute_ensemble_metrics(er)
        assert result["best_ipsae_min"] == 0.70
        assert result["best_iptm"] == 0.80

    def test_empty(self):
        result = compute_ensemble_metrics({})
        assert result["ensemble_ipsae_min"] == 0.0


class TestRankDesigns:
    def test_sorts_descending(self):
        designs = [
            {"design_id": "a", "composite_score": 0.5},
            {"design_id": "b", "composite_score": 0.8},
            {"design_id": "c", "composite_score": 0.3},
        ]
        ranked = rank_designs(designs)
        assert ranked[0]["design_id"] == "b"
        assert ranked[1]["design_id"] == "a"
        assert ranked[2]["design_id"] == "c"

    def test_assigns_rank(self):
        designs = [
            {"design_id": "x", "composite_score": 0.9},
            {"design_id": "y", "composite_score": 0.1},
        ]
        ranked = rank_designs(designs)
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2


# ---------------------------------------------------------------------------
# Diversity tests
# ---------------------------------------------------------------------------


class TestSequenceIdentity:
    def test_identical(self):
        assert compute_sequence_identity("MKWAS", "MKWAS") == 1.0

    def test_completely_different(self):
        assert compute_sequence_identity("AAAAA", "LLLLL") == 0.0

    def test_partial_match(self):
        assert compute_sequence_identity("MKWAS", "MKXYZ") == 0.4

    def test_empty(self):
        assert compute_sequence_identity("", "MKWAS") == 0.0


class TestClusterBySequence:
    def test_identical_sequences_same_cluster(self):
        designs = [
            {"design_id": "a", "binder_sequence": "MKWAS"},
            {"design_id": "b", "binder_sequence": "MKWAS"},
        ]
        clusters = cluster_by_sequence(designs, identity_threshold=0.7)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a", "b"}

    def test_different_sequences_different_clusters(self):
        designs = [
            {"design_id": "a", "binder_sequence": "AAAAAAAAAA"},
            {"design_id": "b", "binder_sequence": "LLLLLLLLLL"},
        ]
        clusters = cluster_by_sequence(designs, identity_threshold=0.7)
        assert len(clusters) == 2


class TestSelectDiverseRepresentatives:
    def test_limits_per_cluster(self):
        ranked = [
            {"design_id": "a", "rank": 1},
            {"design_id": "b", "rank": 2},
            {"design_id": "c", "rank": 3},
            {"design_id": "d", "rank": 4},
        ]
        clusters = [["a", "b", "c"], ["d"]]
        selected = select_diverse_representatives(
            ranked, clusters, max_per_cluster=2
        )
        # From cluster 0: a and b (top 2 by rank)
        # From cluster 1: d
        ids = [d["design_id"] for d in selected]
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids
        assert "d" in ids


# ---------------------------------------------------------------------------
# Rosetta tests
# ---------------------------------------------------------------------------


class TestRosetta:
    def test_is_available(self):
        # Just check it doesn't crash
        result = rosetta_available()
        assert isinstance(result, bool)

    def test_score_interface_when_unavailable(self):
        from bm2_evaluator.metrics.rosetta import score_interface

        if not rosetta_available():
            result = score_interface("/nonexistent.pdb")
            assert result is None


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------


def _sample_scored_designs():
    return [
        {
            "rank": 1,
            "design_id": "design_001",
            "source_tool": "bindcraft",
            "binder_length": 80,
            "tier": "consensus_hit",
            "composite_score": 0.85,
            "ensemble_ipsae_min": 0.72,
            "ensemble_iptm": 0.81,
            "ensemble_plddt": 0.88,
            "multi_model_agreement": 1.0,
            "binder_sequence": "MKWASDEFGH" * 8,
            "monomer_rmsd": 1.2,
            "monomer_passes": True,
            "rosetta": {"dG": -15.3, "dSASA": 1200, "shape_complementarity": 0.68, "n_hbonds": 5},
            "engine_results": {
                "boltz2": {
                    "bt_ipsae": 0.74, "tb_ipsae": 0.72, "ipsae_min": 0.72,
                    "ipsae_max": 0.74, "iptm": 0.82, "ptm": 0.78,
                    "plddt_binder_mean_raw": 0.87, "plddt_binder_norm": 0.87,
                    "plddt_target_mean_raw": 0.91, "plddt_scale_max": 1.0,
                    "pae_interaction_mean": 4.5, "pae_binder_mean": 2.1,
                },
                "af2": {
                    "bt_ipsae": 0.71, "tb_ipsae": 0.73, "ipsae_min": 0.71,
                    "ipsae_max": 0.73, "iptm": 0.81, "ptm": 0.76,
                    "plddt_binder_mean_raw": 85.0, "plddt_binder_norm": 0.85,
                    "plddt_target_mean_raw": 90.0, "plddt_scale_max": 100.0,
                    "pae_interaction_mean": 5.1, "pae_binder_mean": 2.5,
                },
            },
        },
        {
            "rank": 2,
            "design_id": "design_002",
            "source_tool": "boltzgen",
            "binder_length": 65,
            "tier": "strong",
            "composite_score": 0.72,
            "ensemble_ipsae_min": 0.55,
            "ensemble_iptm": 0.70,
            "ensemble_plddt": 0.82,
            "multi_model_agreement": 0.5,
            "binder_sequence": "ACDEFGHIKL" * 6 + "ACDEF",
            "monomer_rmsd": 2.1,
            "monomer_passes": True,
            "rosetta": None,
            "engine_results": {
                "boltz2": {
                    "ipsae_min": 0.65, "iptm": 0.75, "plddt_binder_norm": 0.84,
                },
                "af2": {
                    "ipsae_min": 0.55, "iptm": 0.70, "plddt_binder_norm": 0.80,
                },
            },
        },
    ]


class TestSummaryCSV:
    def test_export(self, tmp_path):
        scored = _sample_scored_designs()
        path = tmp_path / "summary.csv"
        export_summary_csv(scored, path)
        assert path.exists()

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["design_id"] == "design_001"
        assert rows[0]["tier"] == "consensus_hit"
        assert rows[0]["boltz2_ipsae_min"] != ""
        assert rows[0]["rosetta_dG"] != ""

    def test_null_rosetta(self, tmp_path):
        scored = _sample_scored_designs()
        path = tmp_path / "summary.csv"
        export_summary_csv(scored, path)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        # design_002 has no Rosetta
        assert rows[1]["rosetta_dG"] == ""


class TestDetailCSV:
    def test_export(self, tmp_path):
        scored = _sample_scored_designs()
        path = tmp_path / "detail.csv"
        export_detail_csv(scored, path)
        assert path.exists()

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        # 2 designs * 2 engines = 4 rows
        assert len(rows) == 4
        engines = {r["engine"] for r in rows}
        assert "boltz2" in engines
        assert "af2" in engines


class TestTextReport:
    def test_generate(self, tmp_path):
        scored = _sample_scored_designs()
        report = generate_report(
            scored,
            {"name": "PD-L1", "chain": "A", "n_residues": 290},
            {"engines": ["boltz2", "af2"], "pae_cutoff": 15.0,
             "ipsae_consensus_threshold": 0.61, "use_rosetta": False},
            tmp_path / "report.txt",
        )
        assert "BM2 Evaluator" in report
        assert "consensus_hit" in report
        assert "PD-L1" in report
        assert "design_001" in report
        assert (tmp_path / "report.txt").exists()

    def test_tier_counts(self, tmp_path):
        scored = _sample_scored_designs()
        report = generate_report(
            scored,
            {"name": "test", "chain": "A", "n_residues": 100},
            {},
            tmp_path / "report.txt",
        )
        assert "1" in report  # At least 1 consensus_hit


class TestCompareTools:
    def test_comparison_table(self):
        scored = _sample_scored_designs()
        table = compare_tools(scored)
        assert "bindcraft" in table
        assert "boltzgen" in table
        assert "Designs" in table

    def test_empty(self):
        table = compare_tools([])
        assert "No designs" in table


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_score_help(self):
        import subprocess

        result = subprocess.run(
            ["python", "-m", "bm2_evaluator.cli", "score", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--designs" in result.stdout

    def test_report_help(self):
        import subprocess

        result = subprocess.run(
            ["python", "-m", "bm2_evaluator.cli", "report", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--eval-dir" in result.stdout

    def test_version(self):
        import subprocess

        result = subprocess.run(
            ["python", "-m", "bm2_evaluator.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
