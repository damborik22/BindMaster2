"""Tests for the refolding layer.

Tests cover:
- WorkerOutput serialization/deserialization
- FASTA writing (target-first convention)
- Chain slice computation from chain_order
- Engine result building with ipSAE computation
- Orchestrator graceful failure handling
- Monomer RMSD computation
- pLDDT scale recording per engine
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bm2_evaluator.core.models import (
    EngineResult,
    EvalConfig,
    EvaluationResult,
    IngestedDesign,
    SourceTool,
)
from bm2_evaluator.refolding.base import MonomerResult, RefoldingEngine, WorkerOutput
from bm2_evaluator.refolding.boltz2 import Boltz2Engine
from bm2_evaluator.refolding.af2 import AF2Engine
from bm2_evaluator.refolding.monomer import MonomerValidator, compute_ca_rmsd
from bm2_evaluator.refolding.orchestrator import RefoldingOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BINDER_SEQ = "MKWAS"
TARGET_SEQ = "AGLVIFDKER"


@pytest.fixture
def worker_output_dict():
    """Sample worker output as dict (what metrics.json contains)."""
    return {
        "engine": "boltz2",
        "chain_order": "binder_first",
        "target_length": 10,
        "binder_length": 5,
        "iptm": 0.82,
        "ptm": 0.78,
        "plddt_binder_mean": 0.87,
        "plddt_binder_min": 0.65,
        "plddt_target_mean": 0.91,
        "plddt_complex_mean": 0.89,
        "plddt_scale_max": 1.0,
        "pae_matrix_file": "pae.npy",
        "structure_file": "structure.pdb",
        "pae_bt_mean": 5.2,
        "pae_tb_mean": 4.8,
        "success": True,
        "error": None,
    }


@pytest.fixture
def worker_output_json(tmp_path, worker_output_dict):
    """Write a metrics.json and return its path."""
    path = tmp_path / "metrics.json"
    with open(path, "w") as f:
        json.dump(worker_output_dict, f)
    return path


@pytest.fixture
def mock_pae_and_structure(tmp_path):
    """Create mock PAE .npy and structure .pdb in a directory."""
    # PAE: 15x15 (binder=5, target=10), binder-first ordering
    n = 15
    pae = np.full((n, n), 3.0)
    # Interchain blocks: moderate PAE
    pae[:5, 5:] = 6.0  # binder->target
    pae[5:, :5] = 6.0  # target->binder
    np.fill_diagonal(pae, 0.0)

    np.save(tmp_path / "pae.npy", pae)

    # Minimal PDB with binder and target chains
    from tests.conftest import MINIMAL_PDB

    (tmp_path / "structure.pdb").write_text(MINIMAL_PDB)

    return tmp_path


# ---------------------------------------------------------------------------
# WorkerOutput tests
# ---------------------------------------------------------------------------


class TestWorkerOutput:
    def test_from_json(self, worker_output_json):
        wo = WorkerOutput.from_json(worker_output_json)
        assert wo.engine == "boltz2"
        assert wo.chain_order == "binder_first"
        assert wo.iptm == 0.82
        assert wo.plddt_scale_max == 1.0
        assert wo.success is True
        assert wo.error is None

    def test_to_json_roundtrip(self, tmp_path):
        wo = WorkerOutput(
            engine="af2",
            chain_order="target_first",
            target_length=100,
            binder_length=50,
            iptm=0.75,
            ptm=0.70,
            plddt_binder_mean=85.0,
            plddt_binder_min=72.0,
            plddt_target_mean=90.0,
            plddt_complex_mean=88.0,
            plddt_scale_max=100.0,
            pae_matrix_file="pae.npy",
            structure_file="structure.pdb",
            success=True,
        )
        path = tmp_path / "metrics.json"
        wo.to_json(path)

        loaded = WorkerOutput.from_json(path)
        assert loaded.engine == "af2"
        assert loaded.plddt_scale_max == 100.0
        assert loaded.iptm == 0.75

    def test_extra_metrics_preserved(self, tmp_path):
        data = {
            "engine": "boltz2",
            "chain_order": "binder_first",
            "target_length": 10,
            "binder_length": 5,
            "iptm": 0.8,
            "ptm": 0.7,
            "plddt_binder_mean": 0.9,
            "plddt_binder_min": 0.7,
            "plddt_target_mean": 0.9,
            "plddt_complex_mean": 0.9,
            "plddt_scale_max": 1.0,
            "pae_matrix_file": "pae.npy",
            "structure_file": "structure.pdb",
            "success": True,
            "error": None,
            "aux_bt_ipsae": 0.65,
            "aux_tb_ipsae": 0.58,
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(data, f)

        wo = WorkerOutput.from_json(path)
        assert wo.extra["aux_bt_ipsae"] == 0.65
        assert wo.extra["aux_tb_ipsae"] == 0.58

    def test_binder_first_slices(self):
        wo = WorkerOutput(
            engine="boltz2",
            chain_order="binder_first",
            target_length=100,
            binder_length=50,
            iptm=0.0,
            ptm=0.0,
            plddt_binder_mean=0.0,
            plddt_binder_min=0.0,
            plddt_target_mean=0.0,
            plddt_complex_mean=0.0,
            plddt_scale_max=1.0,
            pae_matrix_file="pae.npy",
            structure_file="structure.pdb",
            success=True,
        )
        assert wo.get_binder_slice() == slice(0, 50)
        assert wo.get_target_slice() == slice(50, 150)

    def test_target_first_slices(self):
        wo = WorkerOutput(
            engine="af2",
            chain_order="target_first",
            target_length=100,
            binder_length=50,
            iptm=0.0,
            ptm=0.0,
            plddt_binder_mean=0.0,
            plddt_binder_min=0.0,
            plddt_target_mean=0.0,
            plddt_complex_mean=0.0,
            plddt_scale_max=100.0,
            pae_matrix_file="pae.npy",
            structure_file="structure.pdb",
            success=True,
        )
        assert wo.get_binder_slice() == slice(100, 150)
        assert wo.get_target_slice() == slice(0, 100)


# ---------------------------------------------------------------------------
# FASTA writing tests
# ---------------------------------------------------------------------------


class TestFastaWriting:
    def test_write_fasta(self, tmp_path):
        engine = Boltz2Engine()
        fasta_path = tmp_path / "test.fasta"
        engine._write_fasta(
            {"target": TARGET_SEQ, "binder": BINDER_SEQ}, fasta_path
        )
        content = fasta_path.read_text()
        assert ">target" in content
        assert ">binder" in content
        assert TARGET_SEQ in content
        assert BINDER_SEQ in content

        # Verify target comes first in the file
        target_pos = content.index(">target")
        binder_pos = content.index(">binder")
        assert target_pos < binder_pos

    def test_af2_write_fasta(self, tmp_path):
        """AF2 engine also writes standard FASTA."""
        engine = AF2Engine()
        fasta_path = tmp_path / "test.fasta"
        engine._write_fasta({"binder": BINDER_SEQ}, fasta_path)
        content = fasta_path.read_text()
        assert ">binder" in content
        assert BINDER_SEQ in content


# ---------------------------------------------------------------------------
# Engine pLDDT scale tests
# ---------------------------------------------------------------------------


class TestPlddtScales:
    def test_boltz2_scale(self, worker_output_json):
        wo = WorkerOutput.from_json(worker_output_json)
        assert wo.plddt_scale_max == 1.0  # Boltz2: 0-1

    def test_af2_scale(self, tmp_path):
        data = {
            "engine": "af2",
            "chain_order": "target_first",
            "target_length": 10,
            "binder_length": 5,
            "iptm": 0.7,
            "ptm": 0.7,
            "plddt_binder_mean": 85.0,
            "plddt_binder_min": 72.0,
            "plddt_target_mean": 90.0,
            "plddt_complex_mean": 88.0,
            "plddt_scale_max": 100.0,
            "pae_matrix_file": "pae.npy",
            "structure_file": "structure.pdb",
            "success": True,
            "error": None,
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as f:
            json.dump(data, f)
        wo = WorkerOutput.from_json(path)
        assert wo.plddt_scale_max == 100.0  # AF2: 0-100


# ---------------------------------------------------------------------------
# Orchestrator tests (with mocks)
# ---------------------------------------------------------------------------


class MockEngine(RefoldingEngine):
    """Mock engine for testing orchestrator."""

    def __init__(self, engine_name: str, should_fail: bool = False):
        self._name = engine_name
        self.should_fail = should_fail

    @property
    def name(self) -> str:
        return self._name

    def refold_complex(self, binder_seq, target_seq, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.should_fail:
            return WorkerOutput(
                engine=self._name,
                chain_order="binder_first",
                target_length=len(target_seq),
                binder_length=len(binder_seq),
                iptm=0.0,
                ptm=0.0,
                plddt_binder_mean=0.0,
                plddt_binder_min=0.0,
                plddt_target_mean=0.0,
                plddt_complex_mean=0.0,
                plddt_scale_max=1.0,
                pae_matrix_file="pae.npy",
                structure_file="structure.pdb",
                success=False,
                error="Mock failure",
            )

        # Create mock PAE and structure
        n = len(binder_seq) + len(target_seq)
        pae = np.full((n, n), 5.0)
        np.fill_diagonal(pae, 0.0)
        np.save(output_dir / "pae.npy", pae)

        from tests.conftest import MINIMAL_PDB
        (output_dir / "structure.pdb").write_text(MINIMAL_PDB)

        wo = WorkerOutput(
            engine=self._name,
            chain_order="binder_first",
            target_length=len(target_seq),
            binder_length=len(binder_seq),
            iptm=0.82,
            ptm=0.78,
            plddt_binder_mean=0.87,
            plddt_binder_min=0.65,
            plddt_target_mean=0.91,
            plddt_complex_mean=0.89,
            plddt_scale_max=1.0,
            pae_matrix_file="pae.npy",
            structure_file="structure.pdb",
            success=True,
        )
        wo.to_json(output_dir / "metrics.json")
        return wo

    def refold_monomer(self, binder_seq, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from tests.conftest import MINIMAL_PDB
        pdb_path = output_dir / "monomer.pdb"
        pdb_path.write_text(MINIMAL_PDB)

        return MonomerResult(
            plddt_mean=0.85,
            plddt_scale_max=1.0,
            structure_path=pdb_path,
            success=True,
        )


class TestOrchestrator:
    def test_single_engine_success(self, tmp_path, sample_ingested_design):
        engine = MockEngine("boltz2")
        config = EvalConfig()
        orch = RefoldingOrchestrator([engine], config)

        result = orch.evaluate_design(
            sample_ingested_design, tmp_path, run_monomer=False
        )

        assert isinstance(result, EvaluationResult)
        assert "boltz2" in result.engine_results
        er = result.engine_results["boltz2"]
        assert er.iptm == 0.82
        assert er.ipsae_min >= 0.0
        assert er.ipsae_max >= er.ipsae_min

    def test_two_engines_success(self, tmp_path, sample_ingested_design):
        engine1 = MockEngine("boltz2")
        engine2 = MockEngine("af2")
        config = EvalConfig()
        orch = RefoldingOrchestrator([engine1, engine2], config)

        result = orch.evaluate_design(
            sample_ingested_design, tmp_path, run_monomer=False
        )

        assert "boltz2" in result.engine_results
        assert "af2" in result.engine_results

    def test_one_engine_fails_other_succeeds(
        self, tmp_path, sample_ingested_design
    ):
        """Orchestrator continues when one engine fails."""
        engine_ok = MockEngine("boltz2")
        engine_fail = MockEngine("af2", should_fail=True)
        config = EvalConfig()
        orch = RefoldingOrchestrator([engine_ok, engine_fail], config)

        result = orch.evaluate_design(
            sample_ingested_design, tmp_path, run_monomer=False
        )

        assert "boltz2" in result.engine_results
        assert "af2" not in result.engine_results

    def test_all_engines_fail(self, tmp_path, sample_ingested_design):
        """Orchestrator returns empty engine_results when all fail."""
        engine1 = MockEngine("boltz2", should_fail=True)
        engine2 = MockEngine("af2", should_fail=True)
        config = EvalConfig()
        orch = RefoldingOrchestrator([engine1, engine2], config)

        result = orch.evaluate_design(
            sample_ingested_design, tmp_path, run_monomer=False
        )

        assert len(result.engine_results) == 0

    def test_ipsae_computed_from_pae(self, tmp_path, sample_ingested_design):
        """Verify ipSAE is computed from the PAE matrix."""
        engine = MockEngine("boltz2")
        config = EvalConfig(pae_cutoff=15.0)
        orch = RefoldingOrchestrator([engine], config)

        result = orch.evaluate_design(
            sample_ingested_design, tmp_path, run_monomer=False
        )

        er = result.engine_results["boltz2"]
        # With PAE=5.0 everywhere (< 15A cutoff), all residues qualify
        assert er.ipsae_min > 0.0
        assert er.bt_ipsae > 0.0
        assert er.tb_ipsae > 0.0
        assert er.pae_matrix_path.exists()

    def test_pae_matrix_saved(self, tmp_path, sample_ingested_design):
        """Verify PAE matrix is saved as .npy."""
        engine = MockEngine("boltz2")
        config = EvalConfig()
        orch = RefoldingOrchestrator([engine], config)

        result = orch.evaluate_design(
            sample_ingested_design, tmp_path, run_monomer=False
        )

        er = result.engine_results["boltz2"]
        assert er.pae_matrix_path.exists()
        pae = np.load(er.pae_matrix_path)
        expected_size = len(BINDER_SEQ) + len(TARGET_SEQ)
        assert pae.shape == (expected_size, expected_size)


# ---------------------------------------------------------------------------
# Monomer validation tests
# ---------------------------------------------------------------------------


class TestMonomerValidation:
    def test_identical_structures_zero_rmsd(self, minimal_pdb):
        """RMSD of a structure against itself should be ~0."""
        rmsd = compute_ca_rmsd(minimal_pdb, "B", minimal_pdb, "B")
        assert rmsd < 0.01

    def test_monomer_validator_passes(self, tmp_path, minimal_pdb):
        """Validator passes when structures are identical."""
        engine = MockEngine("boltz2")
        validator = MonomerValidator(engine, rmsd_threshold=3.0)

        result = validator.validate(
            binder_seq=BINDER_SEQ,
            complex_structure_path=minimal_pdb,
            binder_chain="B",
            output_dir=tmp_path / "monomer",
        )

        # Mock engine returns the same PDB, so RMSD ≈ 0
        assert result.passes_validation is True
        assert result.monomer_rmsd < 3.0
        assert result.threshold == 3.0

    def test_monomer_validator_threshold(self, tmp_path, minimal_pdb):
        """Test that the RMSD threshold is configurable."""
        engine = MockEngine("boltz2")
        validator = MonomerValidator(engine, rmsd_threshold=0.001)

        result = validator.validate(
            binder_seq=BINDER_SEQ,
            complex_structure_path=minimal_pdb,
            binder_chain="B",
            output_dir=tmp_path / "monomer",
        )

        # With an extremely tight threshold, even identical structures
        # might barely pass due to floating point
        assert result.threshold == 0.001


# ---------------------------------------------------------------------------
# Engine availability tests
# ---------------------------------------------------------------------------


class TestEngineAvailability:
    def test_boltz2_check_nonexistent_path(self):
        engine = Boltz2Engine(venv_path="/nonexistent/path")
        assert engine.check_available() is False

    def test_af2_target_pdb_required(self, tmp_path):
        """AF2 engine returns failure if target_pdb not set."""
        engine = AF2Engine(target_pdb=None)
        result = engine.refold_complex(BINDER_SEQ, TARGET_SEQ, tmp_path)
        assert result.success is False
        assert "target_pdb not set" in result.error
