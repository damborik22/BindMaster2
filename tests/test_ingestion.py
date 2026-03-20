"""Tests for ingestion layer."""

import csv
from pathlib import Path

import pytest

from bm2_evaluator.core.models import SourceTool
from bm2_evaluator.ingestion import get_ingestor, auto_detect, INGESTORS
from bm2_evaluator.ingestion.generic import GenericIngestor
from bm2_evaluator.ingestion.bindcraft import BindCraftIngestor
from bm2_evaluator.ingestion.mosaic import MosaicIngestor
from bm2_evaluator.ingestion.rfdiffusion import RFdiffusionIngestor


class TestRegistry:
    def test_all_ingestors_registered(self):
        assert "generic" in INGESTORS
        assert "bindcraft" in INGESTORS
        assert "boltzgen" in INGESTORS
        assert "mosaic" in INGESTORS
        assert "pxdesign" in INGESTORS
        assert "rfdiffusion" in INGESTORS
        assert "complexa" in INGESTORS

    def test_get_ingestor(self):
        ingestor = get_ingestor("generic")
        assert isinstance(ingestor, GenericIngestor)

    def test_get_unknown_ingestor(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            get_ingestor("nonexistent")


class TestAutoDetect:
    def test_detect_bindcraft(self, bindcraft_output_dir):
        assert auto_detect(bindcraft_output_dir) == "bindcraft"

    def test_detect_rfdiffusion(self, rfdiffusion_output_dir):
        assert auto_detect(rfdiffusion_output_dir) == "rfdiffusion"

    def test_detect_generic_fallback(self, generic_output_dir):
        assert auto_detect(generic_output_dir) == "generic"


class TestGenericIngestor:
    def test_ingest_pdb_files(self, generic_output_dir):
        ingestor = GenericIngestor()
        designs = ingestor.ingest(generic_output_dir)
        assert len(designs) == 2
        assert all(d.source_tool == SourceTool.GENERIC for d in designs)

    def test_ingest_with_metrics_csv(self, generic_output_dir):
        # Create a metrics CSV
        csv_path = generic_output_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["design_id", "plddt", "iptm"]
            )
            writer.writeheader()
            writer.writerow({
                "design_id": "complex_001", "plddt": "85.0", "iptm": "0.7"
            })

        ingestor = GenericIngestor()
        designs = ingestor.ingest(
            generic_output_dir, metrics_csv=csv_path
        )
        assert len(designs) == 2
        d1 = [d for d in designs if d.design_id == "complex_001"][0]
        assert d1.tool_metrics["plddt"] == 85.0
        assert d1.tool_metrics["iptm"] == 0.7

    def test_empty_dir_raises(self, tmp_path):
        ingestor = GenericIngestor()
        with pytest.raises(FileNotFoundError, match="No PDB/CIF"):
            ingestor.ingest(tmp_path)

    def test_nonexistent_dir_raises(self, tmp_path):
        ingestor = GenericIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest(tmp_path / "nope")

    def test_sequences_extracted(self, generic_output_dir):
        ingestor = GenericIngestor()
        designs = ingestor.ingest(generic_output_dir)
        for d in designs:
            assert len(d.binder_sequence) > 0
            assert len(d.target_sequence) > 0
            assert d.binder_length == len(d.binder_sequence)
            assert d.target_length == len(d.target_sequence)


class TestBindCraftIngestor:
    def test_ingest(self, bindcraft_output_dir):
        ingestor = BindCraftIngestor()
        designs = ingestor.ingest(bindcraft_output_dir)
        assert len(designs) == 2
        assert all(d.source_tool == SourceTool.BINDCRAFT for d in designs)

    def test_metrics_loaded(self, bindcraft_output_dir):
        ingestor = BindCraftIngestor()
        designs = ingestor.ingest(bindcraft_output_dir)
        d1 = [d for d in designs if d.design_id == "design_001"][0]
        assert d1.tool_metrics["plddt"] == 87.5
        assert d1.tool_metrics["i_ptm"] == 0.75
        assert d1.tool_metrics["dG"] == -15.3

    def test_plddt_scale_is_0_100(self, bindcraft_output_dir):
        """BindCraft pLDDT is 0-100 (AF2 convention)."""
        ingestor = BindCraftIngestor()
        designs = ingestor.ingest(bindcraft_output_dir)
        for d in designs:
            if "plddt" in d.tool_metrics:
                assert d.tool_metrics["plddt"] > 1.0  # On 0-100 scale


class TestMosaicIngestor:
    def test_delegates_to_generic(self, generic_output_dir):
        ingestor = MosaicIngestor()
        designs = ingestor.ingest(generic_output_dir)
        assert len(designs) == 2
        assert all(d.source_tool == SourceTool.MOSAIC for d in designs)


class TestRFdiffusionIngestor:
    def test_ingest_with_fasta(self, rfdiffusion_output_dir):
        ingestor = RFdiffusionIngestor()
        designs = ingestor.ingest(rfdiffusion_output_dir)
        # Should get MPNN sequences (1 from FASTA, skipping native)
        assert len(designs) >= 1
        assert all(
            d.source_tool == SourceTool.RFDIFFUSION for d in designs
        )

    def test_chain_convention(self, rfdiffusion_output_dir):
        """RFdiffusion: binder=A, target=B (opposite of BindCraft)."""
        ingestor = RFdiffusionIngestor()
        designs = ingestor.ingest(rfdiffusion_output_dir)
        for d in designs:
            assert d.binder_chain == "A"
            assert d.target_chain == "B"

    def test_mpnn_scores_in_metrics(self, rfdiffusion_output_dir):
        ingestor = RFdiffusionIngestor()
        designs = ingestor.ingest(rfdiffusion_output_dir)
        for d in designs:
            if "score" in d.tool_metrics:
                assert d.tool_metrics["score"] > 0
