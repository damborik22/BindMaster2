"""Tests for BM2 agents."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bm2.agents.base import Agent
from bm2.agents.target_analyst import TargetAnalyst
from bm2.agents.strategy_planner import StrategyPlanner
from bm2.agents.design_runner import DesignRunner
from bm2.agents.evaluator_agent import EvaluatorAgent
from bm2.agents.wetlab_advisor import WetLabAdvisor
from bm2.agents.maturation_agent import MaturationAgent
from bm2.agents.campaign_orchestrator import CampaignOrchestrator
from bm2.core.campaign import CampaignManager
from bm2.core.config import BM2Config
from bm2.core.models import (
    Campaign,
    CampaignState,
    TargetProfile,
    ToolRunConfig,
)
from bm2.tools.base import ToolLauncher
from bm2.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_PDB = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00 90.00           C
ATOM      3  C   ALA A   1       3.000   1.000   1.000  1.00 90.00           C
ATOM      4  O   ALA A   1       3.500   2.000   1.000  1.00 90.00           O
ATOM      5  N   GLY A   2       3.500   0.000   1.000  1.00 85.00           N
ATOM      6  CA  GLY A   2       4.500   0.000   1.000  1.00 85.00           C
ATOM      7  C   GLY A   2       5.500   0.000   1.000  1.00 85.00           C
ATOM      8  O   GLY A   2       6.000   1.000   1.000  1.00 85.00           O
TER
END
"""


class MockToolLauncher(ToolLauncher):
    """Mock tool for testing agents."""

    def __init__(self, tool_name="mock_tool", installed=True):
        self._name = tool_name
        self._installed = installed

    @property
    def name(self):
        return self._name

    @property
    def env_spec(self):
        return "mock"

    def check_installed(self):
        return self._installed

    def prepare_config(self, campaign, run_config, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"config": "mock"}

    def launch(self, prepared, run_dir):
        # Create mock output
        output = run_dir / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "design_001.pdb").write_text("MOCK PDB")
        # Return a mock Popen
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        return mock_proc

    def is_complete(self, run_dir):
        return (run_dir / "output").exists() and any(
            (run_dir / "output").glob("*.pdb")
        )

    def output_dir(self, run_dir):
        return run_dir / "output"

    def parser_name(self):
        return "generic"


@pytest.fixture
def target_pdb(tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text(MINIMAL_PDB)
    return pdb


@pytest.fixture
def campaign(target_pdb):
    return Campaign(
        id="test_campaign",
        name="test",
        target=TargetProfile(
            pdb_path=target_pdb,
            chains=["A"],
        ),
    )


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(MockToolLauncher("bindcraft"))
    reg.register(MockToolLauncher("boltzgen"))
    return reg


@pytest.fixture
def manager(tmp_path):
    mgr = CampaignManager(base_dir=tmp_path / ".bm2")
    # Create campaign dir
    campaign_dir = mgr.campaigns_dir / "test_campaign"
    (campaign_dir / "runs").mkdir(parents=True)
    (campaign_dir / "evaluation").mkdir(parents=True)
    return mgr


@pytest.fixture
def config():
    return BM2Config()


# ---------------------------------------------------------------------------
# TargetAnalyst tests
# ---------------------------------------------------------------------------


class TestTargetAnalyst:
    def test_populates_sequence(self, campaign, registry):
        agent = TargetAnalyst(registry)
        campaign = agent.run(campaign)
        assert campaign.target.target_length > 0
        assert len(campaign.target.target_sequence) > 0
        assert campaign.state == CampaignState.ANALYZING

    def test_recommends_installed_tools(self, campaign, registry):
        agent = TargetAnalyst(registry)
        campaign = agent.run(campaign)
        for tool in campaign.target.recommended_tools:
            assert tool in registry.list_installed()

    def test_suggests_length_range(self, campaign, registry):
        agent = TargetAnalyst(registry)
        campaign = agent.run(campaign)
        lo, hi = campaign.target.suggested_length_range
        assert lo > 0
        assert hi > lo

    def test_no_target_raises(self, registry):
        c = Campaign(id="x", name="x", target=None)
        agent = TargetAnalyst(registry)
        with pytest.raises(ValueError, match="no target"):
            agent.run(c)

    def test_wrong_state_raises(self, campaign, registry):
        campaign.state = CampaignState.RANKED
        agent = TargetAnalyst(registry)
        with pytest.raises(ValueError, match="requires state"):
            agent.run(campaign)


# ---------------------------------------------------------------------------
# StrategyPlanner tests
# ---------------------------------------------------------------------------


class TestStrategyPlanner:
    def test_creates_tool_runs(self, campaign, registry, config):
        # First run analyst
        TargetAnalyst(registry).run(campaign)

        agent = StrategyPlanner(registry, config)
        campaign = agent.run(campaign, total_designs=100)

        assert len(campaign.tool_runs) > 0
        assert campaign.state == CampaignState.PLANNING

    def test_allocates_proportionally(self, campaign, registry, config):
        TargetAnalyst(registry).run(campaign)

        agent = StrategyPlanner(registry, config)
        campaign = agent.run(campaign, total_designs=200)

        total = sum(tr.num_designs for tr in campaign.tool_runs)
        assert total > 0
        # Each tool gets at least 10
        for tr in campaign.tool_runs:
            assert tr.num_designs >= 10

    def test_tool_specific_settings(self, campaign, registry, config):
        # Add boltzgen to recommended
        TargetAnalyst(registry).run(campaign)

        agent = StrategyPlanner(registry, config)
        campaign = agent.run(campaign)

        boltzgen_runs = [
            tr for tr in campaign.tool_runs if tr.tool_name == "boltzgen"
        ]
        if boltzgen_runs:
            assert boltzgen_runs[0].extra_settings.get("protocol") == "protein-anything"


# ---------------------------------------------------------------------------
# DesignRunner tests
# ---------------------------------------------------------------------------


class TestDesignRunner:
    def test_runs_tools(self, campaign, registry, manager, config):
        TargetAnalyst(registry).run(campaign)
        StrategyPlanner(registry, config).run(campaign)

        agent = DesignRunner(registry, manager)
        campaign = agent.run(campaign)

        assert campaign.state == CampaignState.DESIGNING
        completed = [
            tr for tr in campaign.tool_runs if tr.status == "completed"
        ]
        assert len(completed) > 0

    def test_skips_completed_runs(self, campaign, registry, manager, config):
        TargetAnalyst(registry).run(campaign)
        StrategyPlanner(registry, config).run(campaign)

        # Mark all as completed already
        for tr in campaign.tool_runs:
            tr.status = "completed"

        agent = DesignRunner(registry, manager)
        campaign = agent.run(campaign)
        assert campaign.state == CampaignState.DESIGNING

    def test_continues_on_failure(self, campaign, manager, config):
        """If one tool fails, others still run."""
        reg = ToolRegistry()
        reg.register(MockToolLauncher("bindcraft", installed=True))

        # Add a failing tool
        fail_launcher = MockToolLauncher("failing_tool", installed=True)
        fail_launcher.launch = MagicMock(side_effect=RuntimeError("boom"))
        reg.register(fail_launcher)

        TargetAnalyst(reg).run(campaign)
        campaign.target.recommended_tools = ["bindcraft", "failing_tool"]
        StrategyPlanner(reg, config).run(campaign)

        agent = DesignRunner(reg, manager)
        campaign = agent.run(campaign)

        statuses = {tr.tool_name: tr.status for tr in campaign.tool_runs}
        assert statuses["bindcraft"] == "completed"
        assert statuses["failing_tool"] == "failed"


# ---------------------------------------------------------------------------
# WetLabAdvisor tests
# ---------------------------------------------------------------------------


class TestWetLabAdvisor:
    @pytest.fixture
    def ranked_campaign(self, campaign, tmp_path):
        campaign.state = CampaignState.RANKED
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        campaign.evaluation_dir = eval_dir

        # Create mock summary CSV
        with open(eval_dir / "evaluation_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank", "design_id", "source_tool", "tier",
                    "ensemble_ipsae_min", "ensemble_iptm",
                    "binder_length", "binder_sequence",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "rank": 1, "design_id": "d001", "source_tool": "bindcraft",
                "tier": "consensus_hit", "ensemble_ipsae_min": "0.72",
                "ensemble_iptm": "0.81", "binder_length": 80,
                "binder_sequence": "MKWASDEFGH" * 8,
            })
            writer.writerow({
                "rank": 2, "design_id": "d002", "source_tool": "boltzgen",
                "tier": "strong", "ensemble_ipsae_min": "0.55",
                "ensemble_iptm": "0.70", "binder_length": 65,
                "binder_sequence": "ACDEFGHIKL" * 6 + "ACDEF",
            })

        return campaign

    def test_generates_plan(self, ranked_campaign):
        agent = WetLabAdvisor()
        campaign = agent.run(ranked_campaign)
        assert campaign.state == CampaignState.WET_LAB_PREP

        eval_dir = Path(ranked_campaign.evaluation_dir)
        plan = eval_dir.parent / "reports" / "wetlab_plan.md"
        assert plan.exists()
        content = plan.read_text()
        assert "Design Selection" in content
        assert "Gene Synthesis" in content

    def test_exports_fasta(self, ranked_campaign):
        agent = WetLabAdvisor()
        agent.run(ranked_campaign)

        eval_dir = Path(ranked_campaign.evaluation_dir)
        fasta = eval_dir.parent / "reports" / "top_designs.fasta"
        assert fasta.exists()
        content = fasta.read_text()
        assert ">d001" in content

    def test_screening_method_by_count(self, ranked_campaign):
        agent = WetLabAdvisor()
        agent.run(ranked_campaign, num_to_test=5)
        eval_dir = Path(ranked_campaign.evaluation_dir)
        plan = (eval_dir.parent / "reports" / "wetlab_plan.md").read_text()
        assert "SPR" in plan or "BLI" in plan


# ---------------------------------------------------------------------------
# MaturationAgent tests
# ---------------------------------------------------------------------------


class TestMaturationAgent:
    def test_auto_selects_strategy(self, registry):
        agent = MaturationAgent(registry)
        # With bindcraft and boltzgen installed, should pick a strategy
        strategy = agent._auto_select_strategy()
        assert strategy in ("partial_diffusion", "warm_start_hallucination", "mpnn_redesign")

    def test_creates_maturation_runs(self, campaign, registry, tmp_path):
        campaign.state = CampaignState.RANKED
        campaign.evaluation_dir = tmp_path / "eval"
        campaign.evaluation_dir.mkdir()

        # Create mock summary
        with open(campaign.evaluation_dir / "evaluation_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["design_id", "tier"])
            writer.writeheader()
            writer.writerow({"design_id": "d001", "tier": "consensus_hit"})

        agent = MaturationAgent(registry)
        campaign = agent.run(campaign)

        assert campaign.state == CampaignState.MATURING
        assert len(campaign.maturation_rounds) == 1
        assert campaign.maturation_rounds[0]["round"] == 1

        # New tool runs should be added
        maturation_runs = [
            tr for tr in campaign.tool_runs
            if tr.extra_settings.get("strategy")
        ]
        assert len(maturation_runs) > 0


# ---------------------------------------------------------------------------
# CampaignOrchestrator tests
# ---------------------------------------------------------------------------


class TestCampaignOrchestrator:
    def test_run_analyze_only(self, campaign, registry, manager, config):
        orch = CampaignOrchestrator(registry, manager, config)
        campaign = orch.run_through(
            campaign, stop_at=CampaignState.ANALYZING
        )
        assert campaign.state == CampaignState.ANALYZING

    def test_run_through_planning(self, campaign, registry, manager, config):
        orch = CampaignOrchestrator(registry, manager, config)
        campaign = orch.run_through(
            campaign, stop_at=CampaignState.PLANNING
        )
        assert campaign.state == CampaignState.PLANNING
        assert len(campaign.tool_runs) > 0

    def test_run_single_agent(self, campaign, registry, manager, config):
        orch = CampaignOrchestrator(registry, manager, config)
        campaign = orch.run_agent(campaign, "analyze")
        assert campaign.state == CampaignState.ANALYZING

    def test_unknown_agent_raises(self, campaign, registry, manager, config):
        orch = CampaignOrchestrator(registry, manager, config)
        with pytest.raises(KeyError, match="Unknown agent"):
            orch.run_agent(campaign, "nonexistent")


# ---------------------------------------------------------------------------
# State machine lifecycle test
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_full_maturation_loop(self, campaign, registry):
        """Test INIT -> ... -> RANKED -> MATURING -> DESIGNING loop."""
        # Fast-forward to RANKED
        campaign.state = CampaignState.INIT
        campaign.transition_to(CampaignState.ANALYZING)
        campaign.transition_to(CampaignState.PLANNING)
        campaign.transition_to(CampaignState.DESIGNING)
        campaign.transition_to(CampaignState.EVALUATING)
        campaign.transition_to(CampaignState.RANKED)

        # Maturation
        campaign.transition_to(CampaignState.MATURING)
        assert campaign.state == CampaignState.MATURING

        # Loop back to designing
        campaign.transition_to(CampaignState.DESIGNING)
        assert campaign.state == CampaignState.DESIGNING
