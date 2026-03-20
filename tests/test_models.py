"""Tests for core data models."""

from pathlib import Path

from bm2_evaluator.core.models import (
    EvalConfig,
    EvaluationResult,
    IngestedDesign,
    ScoredDesign,
    SourceTool,
)


class TestSourceTool:
    def test_enum_values(self):
        assert SourceTool.BINDCRAFT == "bindcraft"
        assert SourceTool.BOLTZGEN == "boltzgen"
        assert SourceTool.MOSAIC == "mosaic"
        assert SourceTool.PXDESIGN == "pxdesign"
        assert SourceTool.RFDIFFUSION == "rfdiffusion"
        assert SourceTool.COMPLEXA == "complexa"
        assert SourceTool.GENERIC == "generic"

    def test_string_comparison(self):
        assert SourceTool.BINDCRAFT == "bindcraft"
        assert SourceTool("boltzgen") == SourceTool.BOLTZGEN


class TestIngestedDesign:
    def test_construction(self, sample_ingested_design):
        d = sample_ingested_design
        assert d.design_id == "test_001"
        assert d.source_tool == SourceTool.GENERIC
        assert d.binder_sequence == "MKWAS"
        assert d.binder_chain == "B"
        assert d.target_sequence == "AGLVIFDKER"
        assert d.target_chain == "A"
        assert d.binder_length == 5
        assert d.target_length == 10

    def test_default_fields(self, minimal_pdb):
        d = IngestedDesign(
            design_id="test",
            source_tool=SourceTool.GENERIC,
            binder_sequence="MK",
            binder_chain="B",
            target_sequence="AGL",
            target_chain="A",
            binder_length=2,
            target_length=3,
            complex_structure_path=minimal_pdb,
        )
        assert d.tool_metrics == {}
        assert d.metadata == {}
        assert d.binder_structure_path is None


class TestEvalConfig:
    def test_defaults(self):
        config = EvalConfig()
        # PAE cutoff: 10A (Adaptyv convention, uniform for Boltz2/AF2)
        assert config.pae_cutoff == 10.0
        # ipSAE threshold: 0.61 (BM2 default from Overath dataset)
        assert config.ipsae_consensus_threshold == 0.61
        assert config.ipsae_strong_threshold == 0.40
        assert config.monomer_rmsd_threshold == 3.0
        assert config.engines == ["boltz2", "af2"]
        assert config.use_rosetta is False
        assert config.plddt_min_norm == 0.70
        assert config.iptm_min_moderate == 0.6
        assert config.n_workers == 1

    def test_custom_values(self):
        config = EvalConfig(
            pae_cutoff=10.0,
            ipsae_consensus_threshold=0.55,
            engines=["boltz2"],
        )
        assert config.pae_cutoff == 10.0
        assert config.ipsae_consensus_threshold == 0.55
        assert config.engines == ["boltz2"]


class TestEvaluationResult:
    def test_construction(self, sample_ingested_design):
        result = EvaluationResult(design=sample_ingested_design)
        assert result.engine_results == {}
        assert result.rosetta_dG is None
        assert result.monomer_rmsd is None


class TestScoredDesign:
    def test_construction(self, sample_ingested_design):
        result = EvaluationResult(design=sample_ingested_design)
        scored = ScoredDesign(evaluation=result)
        assert scored.composite_score == 0.0
        assert scored.tier == "unscored"
        assert scored.rank == 0
