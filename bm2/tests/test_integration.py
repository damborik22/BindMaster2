"""Integration tests for the full BM2 pipeline.

Uses mock tools — no actual GPU computation.
"""

from __future__ import annotations

import csv as csv_mod
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bm2.agents.design_runner import DesignRunner
from bm2.agents.strategy_planner import StrategyPlanner
from bm2.agents.target_analyst import TargetAnalyst
from bm2.core.campaign import CampaignManager
from bm2.core.config import BM2Config
from bm2.core.models import (
    Campaign,
    CampaignState,
    TargetProfile,
    ToolRunConfig,
)
from bm2.skills.manager import SkillsManager
from bm2.tools.base import ToolLauncher
from bm2.tools.registry import ToolRegistry

MINIMAL_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.325   1.490   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       3.957   2.800   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.466   2.734   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.076   1.677   0.000  1.00  0.00           O
TER
END
"""


class MockToolLauncher(ToolLauncher):
    """Fake tool that creates canned output."""

    def __init__(self, name_str="mock_tool"):
        self._name = name_str

    @property
    def name(self):
        return self._name

    @property
    def env_spec(self):
        return "mock"

    def check_installed(self):
        return True

    def prepare_config(self, campaign, run_config, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": run_dir}

    def launch(self, prepared, run_dir):
        output = run_dir / "output"
        output.mkdir(parents=True, exist_ok=True)
        # Write minimal PDB
        for i in range(3):
            (output / f"design_{i}.pdb").write_text(MINIMAL_PDB)
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        return mock_proc

    def is_complete(self, run_dir):
        output = run_dir / "output"
        return output.exists() and any(output.glob("*.pdb"))

    def output_dir(self, run_dir):
        return run_dir / "output"

    def parser_name(self):
        return "generic"


@pytest.fixture
def sample_pdb(tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text(MINIMAL_PDB)
    return pdb


class TestFullPipeline:
    def test_create_analyze_plan_design(self, tmp_path, sample_pdb):
        """End-to-end: create -> analyze -> plan -> design with mock tools."""
        config = BM2Config(base_dir=tmp_path / ".bm2")
        mgr = CampaignManager(tmp_path / ".bm2")

        registry = ToolRegistry()
        registry.register(MockToolLauncher("bindcraft"))
        registry.register(MockToolLauncher("boltzgen"))

        # Create campaign
        campaign = mgr.create("test", sample_pdb, ["A"])
        assert campaign.state == CampaignState.INIT

        # Analyze
        analyst = TargetAnalyst(registry)
        campaign = analyst.run(campaign)
        assert campaign.state == CampaignState.ANALYZING
        assert campaign.target.target_length > 0
        assert len(campaign.target.recommended_tools) > 0

        # Plan
        planner = StrategyPlanner(registry, config)
        campaign = planner.run(campaign, total_designs=30)
        assert campaign.state == CampaignState.PLANNING
        assert len(campaign.tool_runs) > 0

        # Design
        runner = DesignRunner(registry, mgr)
        campaign = runner.run(campaign)
        assert campaign.state == CampaignState.DESIGNING
        completed = [r for r in campaign.tool_runs if r.status == "completed"]
        assert len(completed) > 0


class TestCampaignPersistence:
    def test_roundtrip(self, tmp_path, sample_pdb):
        mgr = CampaignManager(tmp_path / ".bm2")
        campaign = mgr.create("persist_test", sample_pdb, ["A"], ["A1"])

        campaign.target.difficulty_score = 0.45
        campaign.target.recommended_tools = ["bindcraft", "boltzgen"]
        campaign.tool_runs.append(
            ToolRunConfig(tool_name="bindcraft", num_designs=100)
        )
        mgr.save(campaign)

        loaded = mgr.load(campaign.id)
        assert loaded.target.difficulty_score == 0.45
        assert loaded.target.recommended_tools == ["bindcraft", "boltzgen"]
        assert len(loaded.tool_runs) == 1
        assert loaded.tool_runs[0].tool_name == "bindcraft"


class TestStateMachine:
    def test_full_lifecycle(self):
        c = Campaign(id="test", name="test")
        c.transition_to(CampaignState.ANALYZING)
        c.transition_to(CampaignState.PLANNING)
        c.transition_to(CampaignState.DESIGNING)
        c.transition_to(CampaignState.EVALUATING)
        c.transition_to(CampaignState.RANKED)
        # Maturation loop
        c.transition_to(CampaignState.MATURING)
        c.transition_to(CampaignState.DESIGNING)
        c.transition_to(CampaignState.EVALUATING)
        c.transition_to(CampaignState.RANKED)
        # Wet lab
        c.transition_to(CampaignState.WET_LAB_PREP)
        c.transition_to(CampaignState.TESTING)
        c.transition_to(CampaignState.COMPLETED)
        assert c.state == CampaignState.COMPLETED

    def test_invalid_transition(self):
        c = Campaign(id="test", name="test")
        with pytest.raises(ValueError, match="Invalid transition"):
            c.transition_to(CampaignState.RANKED)


class TestSkillsIntegration:
    def test_all_skills_loaded(self):
        mgr = SkillsManager()
        assert len(mgr.list_names()) >= 8

    def test_tool_query(self):
        mgr = SkillsManager()
        results = mgr.query("which tool should I use for a hard target")
        assert results[0].name == "strategy-selector"

    def test_metrics_query(self):
        mgr = SkillsManager()
        results = mgr.query("what does consensus_hit tier mean")
        names = [r.name for r in results]
        assert "metrics-explainer" in names


class TestExperimentalImport:
    def test_import_results(self, tmp_path, sample_pdb):
        mgr = CampaignManager(tmp_path / ".bm2")
        campaign = mgr.create("import_test", sample_pdb, ["A"])

        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "design_id,binds,kd_nm,notes\n"
            "d001,yes,15.3,strong\n"
            "d002,no,,none\n"
            "d003,yes,250,weak\n"
        )

        with open(csv_path, newline="") as f:
            for row in csv_mod.DictReader(f):
                binds = row["binds"].lower() in ("yes", "true", "1")
                campaign.experimental_results.append(
                    {
                        "design_id": row["design_id"],
                        "binds": binds,
                        "kd_nm": float(row["kd_nm"]) if row.get("kd_nm") else None,
                    }
                )

        mgr.save(campaign)
        loaded = mgr.load(campaign.id)
        assert len(loaded.experimental_results) == 3
        assert loaded.experimental_results[0]["binds"] is True
        assert loaded.experimental_results[0]["kd_nm"] == 15.3
