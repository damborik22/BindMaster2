"""Tests for the bm2 CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bm2.cli.main import cli

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


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_pdb(tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text(MINIMAL_PDB)
    return pdb


class TestCLIHelp:
    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "BindMaster 2.0" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_create_help(self, runner):
        result = runner.invoke(cli, ["create", "--help"])
        assert result.exit_code == 0
        assert "--hotspots" in result.output

    def test_run_help(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--through" in result.output

    def test_agent_help(self, runner):
        result = runner.invoke(cli, ["agent", "--help"])
        assert result.exit_code == 0

    def test_tools_help(self, runner):
        result = runner.invoke(cli, ["tools", "--help"])
        assert result.exit_code == 0

    def test_skills_help(self, runner):
        result = runner.invoke(cli, ["skills", "--help"])
        assert result.exit_code == 0

    def test_export_help(self, runner):
        result = runner.invoke(cli, ["export", "designs", "--help"])
        assert result.exit_code == 0

    def test_import_help(self, runner):
        result = runner.invoke(cli, ["import", "results", "--help"])
        assert result.exit_code == 0


class TestToolsCommands:
    def test_tools_list(self, runner):
        result = runner.invoke(cli, ["tools", "list"])
        assert result.exit_code == 0

    def test_tools_check(self, runner):
        result = runner.invoke(cli, ["tools", "check"])
        assert result.exit_code == 0
        # Should show all 6 tool names
        assert "bindcraft" in result.output
        assert "boltzgen" in result.output


class TestSkillsCommands:
    def test_skills_list(self, runner):
        result = runner.invoke(cli, ["skills", "list"])
        assert result.exit_code == 0
        assert "strategy-selector" in result.output

    def test_skills_query(self, runner):
        result = runner.invoke(cli, ["skills", "query", "which", "tool"])
        assert result.exit_code == 0
        assert "strategy-selector" in result.output

    def test_skills_show(self, runner):
        result = runner.invoke(cli, ["skills", "show", "metrics-explainer"])
        assert result.exit_code == 0
        assert "ipSAE" in result.output

    def test_skills_show_nonexistent(self, runner):
        result = runner.invoke(cli, ["skills", "show", "nonexistent"])
        assert result.exit_code == 1


class TestStatusCommand:
    def test_status_no_campaigns(self, runner, tmp_path):
        with patch("bm2.core.campaign.CampaignManager.__init__", return_value=None):
            with patch(
                "bm2.core.campaign.CampaignManager.list_campaigns",
                return_value=[],
            ):
                result = runner.invoke(cli, ["status"])
                assert result.exit_code == 0
