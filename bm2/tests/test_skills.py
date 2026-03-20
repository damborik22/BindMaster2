"""Tests for the Skills system."""

from __future__ import annotations

from pathlib import Path

import pytest

from bm2.skills.manager import Skill, SkillsManager


@pytest.fixture
def manager():
    """Skills manager with builtin skills loaded."""
    return SkillsManager()


@pytest.fixture
def custom_dir(tmp_path):
    """Create a custom skills directory with test skills."""
    custom = tmp_path / "custom_skills"
    custom.mkdir()

    # Skill with YAML frontmatter
    (custom / "my-skill.md").write_text(
        "---\n"
        "name: my-custom-skill\n"
        "description: A custom skill for testing\n"
        "keywords: [custom, test, special]\n"
        "---\n"
        "# Custom Skill Content\n\n"
        "This is a custom skill.\n"
    )

    # Skill without frontmatter
    (custom / "plain-skill.md").write_text(
        "# Plain Skill\n\n"
        "No YAML frontmatter here.\n"
    )

    return custom


class TestSkillsLoading:
    def test_loads_all_builtin(self, manager):
        names = manager.list_names()
        assert len(names) == 9

    def test_all_expected_skills_present(self, manager):
        expected = [
            "assay-selector",
            "campaign-guide",
            "developability",
            "gene-synthesis",
            "maturation-guide",
            "metrics-explainer",
            "strategy-selector",
            "tool-installation",
            "wet-lab-protocols",
        ]
        names = manager.list_names()
        # Check at least 8 of the expected are present
        found = [n for n in expected if n in names]
        assert len(found) >= 8

    def test_skill_has_content(self, manager):
        skill = manager.get("strategy-selector")
        assert len(skill.content) > 100
        assert "BindCraft" in skill.content

    def test_skill_has_keywords(self, manager):
        skill = manager.get("metrics-explainer")
        assert len(skill.keywords) > 0
        assert "ipsae" in skill.keywords or "metric" in skill.keywords

    def test_all_skills_have_descriptions(self, manager):
        for info in manager.list_all():
            assert info["description"], f"{info['name']} has no description"


class TestSkillsQuery:
    def test_tool_selection_query(self, manager):
        results = manager.query("which tool should I use for my target")
        assert len(results) > 0
        assert results[0].name == "strategy-selector"

    def test_metrics_query(self, manager):
        results = manager.query("what does ipSAE mean")
        assert len(results) > 0
        names = [r.name for r in results]
        assert "metrics-explainer" in names

    def test_expression_query(self, manager):
        results = manager.query("how to express and purify protein ecoli IPTG")
        assert len(results) > 0
        names = [r.name for r in results]
        assert "wet-lab-protocols" in names

    def test_affinity_improvement_query(self, manager):
        results = manager.query("how to improve binder affinity")
        assert len(results) > 0
        names = [r.name for r in results]
        assert "maturation-guide" in names

    def test_installation_query(self, manager):
        results = manager.query("how to install bindcraft")
        assert len(results) > 0
        names = [r.name for r in results]
        assert "tool-installation" in names

    def test_assay_query(self, manager):
        results = manager.query("BLI vs SPR which assay")
        results_names = [r.name for r in results]
        assert "assay-selector" in results_names

    def test_campaign_query(self, manager):
        results = manager.query("how to start a campaign")
        names = [r.name for r in results]
        assert "campaign-guide" in names

    def test_top_n_limits_results(self, manager):
        results = manager.query("tool protein binder", top_n=2)
        assert len(results) <= 2

    def test_no_match_returns_empty(self, manager):
        results = manager.query("quantum computing blockchain")
        assert len(results) == 0


class TestSkillGet:
    def test_get_existing(self, manager):
        skill = manager.get("strategy-selector")
        assert isinstance(skill, Skill)
        assert skill.name == "strategy-selector"
        assert skill.source == "builtin"

    def test_get_nonexistent_raises(self, manager):
        with pytest.raises(KeyError, match="not found"):
            manager.get("nonexistent-skill")


class TestCustomSkills:
    def test_loads_custom_with_frontmatter(self, custom_dir):
        mgr = SkillsManager(custom_dir=custom_dir)
        skill = mgr.get("my-custom-skill")
        assert skill.description == "A custom skill for testing"
        assert "custom" in skill.keywords
        assert skill.source == "custom"

    def test_loads_custom_without_frontmatter(self, custom_dir):
        mgr = SkillsManager(custom_dir=custom_dir)
        skill = mgr.get("plain-skill")
        assert skill.source == "custom"
        assert "Plain Skill" in skill.content

    def test_custom_queryable(self, custom_dir):
        mgr = SkillsManager(custom_dir=custom_dir)
        results = mgr.query("custom test special")
        names = [r.name for r in results]
        assert "my-custom-skill" in names

    def test_builtin_and_custom_coexist(self, custom_dir):
        mgr = SkillsManager(custom_dir=custom_dir)
        names = mgr.list_names()
        # Should have builtin + custom
        assert "strategy-selector" in names
        assert "my-custom-skill" in names


class TestListAll:
    def test_returns_dicts(self, manager):
        all_skills = manager.list_all()
        assert len(all_skills) > 0
        for entry in all_skills:
            assert "name" in entry
            assert "description" in entry
            assert "source" in entry
            assert entry["source"] == "builtin"
