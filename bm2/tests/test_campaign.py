"""Tests for BM2 campaign models, state machine, persistence, and manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from bm2.core.models import (
    Campaign,
    CampaignState,
    DesignSource,
    TargetProfile,
    ToolRunConfig,
    TRANSITIONS,
)
from bm2.core.campaign import CampaignManager
from bm2.core.config import BM2Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_PDB = """\
ATOM      1  CA  ALA A   1       1.0   1.0   1.0  1.00 90.00           C
ATOM      2  CA  GLY A   2       4.0   1.0   1.0  1.00 85.00           C
TER
END
"""


@pytest.fixture
def target_pdb(tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text(MINIMAL_PDB)
    return pdb


@pytest.fixture
def campaign():
    return Campaign(
        id="test_001",
        name="PDL1_binders",
        target=TargetProfile(
            pdb_path=Path("/tmp/target.pdb"),
            chains=["A"],
            target_sequence="AG",
            target_length=2,
            hotspot_residues=["A1"],
        ),
    )


@pytest.fixture
def manager(tmp_path):
    return CampaignManager(base_dir=tmp_path / ".bm2")


# ---------------------------------------------------------------------------
# Campaign model tests
# ---------------------------------------------------------------------------


class TestCampaignState:
    def test_valid_forward_transitions(self, campaign):
        campaign.transition_to(CampaignState.ANALYZING)
        assert campaign.state == CampaignState.ANALYZING
        campaign.transition_to(CampaignState.PLANNING)
        assert campaign.state == CampaignState.PLANNING
        campaign.transition_to(CampaignState.DESIGNING)
        campaign.transition_to(CampaignState.EVALUATING)
        campaign.transition_to(CampaignState.RANKED)
        assert campaign.state == CampaignState.RANKED

    def test_invalid_transition_raises(self, campaign):
        with pytest.raises(ValueError, match="Invalid transition"):
            campaign.transition_to(CampaignState.RANKED)

    def test_maturation_loop(self, campaign):
        """MATURING -> DESIGNING loop for iterative improvement."""
        campaign.transition_to(CampaignState.ANALYZING)
        campaign.transition_to(CampaignState.PLANNING)
        campaign.transition_to(CampaignState.DESIGNING)
        campaign.transition_to(CampaignState.EVALUATING)
        campaign.transition_to(CampaignState.RANKED)
        campaign.transition_to(CampaignState.MATURING)
        campaign.transition_to(CampaignState.DESIGNING)
        assert campaign.state == CampaignState.DESIGNING

    def test_failed_restart(self, campaign):
        """FAILED -> INIT restart."""
        campaign.transition_to(CampaignState.ANALYZING)
        campaign.transition_to(CampaignState.FAILED)
        campaign.transition_to(CampaignState.INIT)
        assert campaign.state == CampaignState.INIT

    def test_all_states_in_transitions(self):
        for state in CampaignState:
            assert state in TRANSITIONS


class TestCampaignSerialization:
    def test_save_and_load(self, tmp_path, campaign):
        path = tmp_path / "campaign.json"
        campaign.save(path)
        loaded = Campaign.load(path)

        assert loaded.id == campaign.id
        assert loaded.name == campaign.name
        assert loaded.state == campaign.state
        assert loaded.target is not None
        assert loaded.target.chains == ["A"]
        assert loaded.target.hotspot_residues == ["A1"]
        assert loaded.eval_engines == ["boltz2", "af2"]

    def test_save_load_with_tool_runs(self, tmp_path, campaign):
        campaign.tool_runs.append(
            ToolRunConfig(
                tool_name="bindcraft",
                num_designs=50,
                binder_length_range=(60, 80),
                hotspot_residues=["A1", "A2"],
                extra_settings={"key": "value"},
                status="completed",
            )
        )
        path = tmp_path / "campaign.json"
        campaign.save(path)
        loaded = Campaign.load(path)

        assert len(loaded.tool_runs) == 1
        tr = loaded.tool_runs[0]
        assert tr.tool_name == "bindcraft"
        assert tr.num_designs == 50
        assert tr.binder_length_range == (60, 80)
        assert tr.status == "completed"

    def test_save_load_none_target(self, tmp_path):
        c = Campaign(id="no_target", name="test")
        path = tmp_path / "campaign.json"
        c.save(path)
        loaded = Campaign.load(path)
        assert loaded.target is None

    def test_to_dict_roundtrip(self, campaign):
        d = campaign.to_dict()
        restored = Campaign.from_dict(d)
        assert restored.id == campaign.id
        assert restored.state == campaign.state


class TestToolRunConfig:
    def test_serialization(self):
        config = ToolRunConfig(
            tool_name="boltzgen",
            num_designs=200,
            extra_settings={"protocol": "protein-anything"},
        )
        d = config.to_dict()
        restored = ToolRunConfig.from_dict(d)
        assert restored.tool_name == "boltzgen"
        assert restored.num_designs == 200
        assert restored.extra_settings["protocol"] == "protein-anything"


class TestTargetProfile:
    def test_serialization(self):
        tp = TargetProfile(
            pdb_path=Path("/tmp/target.pdb"),
            chains=["A"],
            target_sequence="ACDEF",
            target_length=5,
            hotspot_residues=["A1", "A3"],
            difficulty_score=0.4,
        )
        d = tp.to_dict()
        restored = TargetProfile.from_dict(d)
        assert restored.chains == ["A"]
        assert restored.target_length == 5
        assert restored.hotspot_residues == ["A1", "A3"]
        assert restored.difficulty_score == 0.4


# ---------------------------------------------------------------------------
# CampaignManager tests
# ---------------------------------------------------------------------------


class TestCampaignManager:
    def test_create_campaign(self, manager, target_pdb):
        c = manager.create("test", target_pdb, ["A"])
        assert c.name == "test"
        assert c.state == CampaignState.INIT
        assert c.target is not None
        # Target PDB should be copied into campaign dir
        assert c.target.pdb_path.exists()
        assert str(c.target.pdb_path) != str(target_pdb)

    def test_create_makes_directories(self, manager, target_pdb):
        c = manager.create("test", target_pdb, ["A"])
        campaign_dir = manager.campaign_dir(c.id)
        assert (campaign_dir / "target").is_dir()
        assert (campaign_dir / "runs").is_dir()
        assert (campaign_dir / "evaluation").is_dir()
        assert (campaign_dir / "reports").is_dir()

    def test_load_campaign(self, manager, target_pdb):
        c = manager.create("test", target_pdb, ["A"])
        loaded = manager.load(c.id)
        assert loaded.id == c.id
        assert loaded.name == "test"

    def test_load_nonexistent_raises(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent_campaign")

    def test_list_campaigns(self, manager, target_pdb):
        manager.create("first", target_pdb, ["A"])
        manager.create("second", target_pdb, ["A", "B"])
        campaigns = manager.list_campaigns()
        assert len(campaigns) == 2
        names = {c["name"] for c in campaigns}
        assert names == {"first", "second"}

    def test_list_empty(self, manager):
        assert manager.list_campaigns() == []

    def test_save_campaign(self, manager, target_pdb):
        c = manager.create("test", target_pdb, ["A"])
        c.transition_to(CampaignState.ANALYZING)
        manager.save(c)
        loaded = manager.load(c.id)
        assert loaded.state == CampaignState.ANALYZING

    def test_campaign_with_hotspots(self, manager, target_pdb):
        c = manager.create("test", target_pdb, ["A"], hotspots=["A1", "A2"])
        assert c.target.hotspot_residues == ["A1", "A2"]


# ---------------------------------------------------------------------------
# DesignSource enum tests
# ---------------------------------------------------------------------------


class TestDesignSource:
    def test_all_values(self):
        assert DesignSource.BINDCRAFT == "bindcraft"
        assert DesignSource.BOLTZGEN == "boltzgen"
        assert DesignSource.MANUAL == "manual"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestBM2Config:
    def test_defaults(self):
        config = BM2Config()
        assert config.pae_cutoff == 15.0
        assert config.ipsae_threshold == 0.61
        assert config.eval_engines == ["boltz2", "af2"]

    def test_save_load_roundtrip(self, tmp_path):
        config = BM2Config(
            base_dir=tmp_path,
            pae_cutoff=10.0,
            gpu_ids=[0, 1],
        )
        path = tmp_path / "config.toml"
        config.save(path)
        loaded = BM2Config.load(path)
        assert loaded.pae_cutoff == 10.0
        assert loaded.gpu_ids == [0, 1]


# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_auto_discover_registers_installed(self):
        from bm2.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.auto_discover()
        # On this system, BindCraft should be found
        installed = registry.list_installed()
        assert isinstance(installed, list)

    def test_get_missing_raises(self):
        from bm2.tools.registry import ToolRegistry

        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_tool")

    def test_is_registered(self):
        from bm2.tools.registry import ToolRegistry
        from bm2.tools.bindcraft import BindCraftLauncher

        registry = ToolRegistry()
        launcher = BindCraftLauncher(install_dir=Path("/tmp/fake"))
        # Won't actually register since check_installed fails
        assert not registry.is_registered("bindcraft")


# ---------------------------------------------------------------------------
# Tool launcher tests (BindCraft as example)
# ---------------------------------------------------------------------------


class TestBindCraftLauncher:
    def test_check_installed_false(self):
        from bm2.tools.bindcraft import BindCraftLauncher

        launcher = BindCraftLauncher(install_dir=Path("/nonexistent"))
        assert launcher.check_installed() is False

    def test_name(self):
        from bm2.tools.bindcraft import BindCraftLauncher

        launcher = BindCraftLauncher()
        assert launcher.name == "bindcraft"

    def test_parser_name(self):
        from bm2.tools.bindcraft import BindCraftLauncher

        launcher = BindCraftLauncher()
        assert launcher.parser_name() == "bindcraft"

    def test_output_dir(self):
        from bm2.tools.bindcraft import BindCraftLauncher

        launcher = BindCraftLauncher()
        assert launcher.output_dir(Path("/tmp/run")) == Path("/tmp/run/output")
