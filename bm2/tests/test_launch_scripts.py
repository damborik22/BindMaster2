"""Tests for launch script generation helpers."""

import os
from pathlib import Path

import pytest


class TestCondaLaunchScript:
    def test_generates_script_file(self, tmp_path):
        from bm2.tools.base import ToolLauncher

        class DummyLauncher(ToolLauncher):
            name = "dummy"
            env_spec = "conda:test"
            def check_installed(self): return True
            def prepare_config(self, campaign, run_config, run_dir): return {}
            def launch(self, prepared, run_dir): pass
            def is_complete(self, run_dir): return True
            def output_dir(self, run_dir): return run_dir
            def parser_name(self): return "generic"

        launcher = DummyLauncher()
        script = launcher._write_conda_launch_script(
            run_dir=tmp_path,
            env_name="BindCraft",
            commands="python bindcraft.py --help",
            cwd="/home/user/BindCraft",
        )

        assert script.exists()
        assert script.suffix == ".sh"
        content = script.read_text()
        assert "conda activate BindCraft" in content
        assert 'cd "/home/user/BindCraft"' in content
        assert "python bindcraft.py --help" in content
        assert "conda.sh" in content
        assert "set +u" in content
        assert os.access(script, os.X_OK)

    def test_script_has_log_redirect(self, tmp_path):
        from bm2.tools.base import ToolLauncher

        class DummyLauncher(ToolLauncher):
            name = "dummy"
            env_spec = "conda:test"
            def check_installed(self): return True
            def prepare_config(self, campaign, run_config, run_dir): return {}
            def launch(self, prepared, run_dir): pass
            def is_complete(self, run_dir): return True
            def output_dir(self, run_dir): return run_dir
            def parser_name(self): return "generic"

        launcher = DummyLauncher()
        script = launcher._write_conda_launch_script(
            run_dir=tmp_path,
            env_name="test_env",
            commands="echo hello",
            log_file="test.log",
        )
        content = script.read_text()
        assert "test.log" in content

    def test_no_cwd_omits_cd(self, tmp_path):
        from bm2.tools.base import ToolLauncher

        class DummyLauncher(ToolLauncher):
            name = "dummy"
            env_spec = "conda:test"
            def check_installed(self): return True
            def prepare_config(self, campaign, run_config, run_dir): return {}
            def launch(self, prepared, run_dir): pass
            def is_complete(self, run_dir): return True
            def output_dir(self, run_dir): return run_dir
            def parser_name(self): return "generic"

        launcher = DummyLauncher()
        script = launcher._write_conda_launch_script(
            run_dir=tmp_path,
            env_name="test_env",
            commands="echo hello",
        )
        content = script.read_text()
        assert "\ncd " not in content


class TestVenvLaunchScript:
    def test_generates_venv_script(self, tmp_path):
        from bm2.tools.base import ToolLauncher

        class DummyLauncher(ToolLauncher):
            name = "dummy"
            env_spec = "venv:test"
            def check_installed(self): return True
            def prepare_config(self, campaign, run_config, run_dir): return {}
            def launch(self, prepared, run_dir): pass
            def is_complete(self, run_dir): return True
            def output_dir(self, run_dir): return run_dir
            def parser_name(self): return "generic"

        launcher = DummyLauncher()
        script = launcher._write_venv_launch_script(
            run_dir=tmp_path,
            venv_path="/home/user/Mosaic/.venv",
            commands="python hallucinate.py",
            cwd="/home/user/run_dir",
        )

        assert script.exists()
        content = script.read_text()
        assert 'source "/home/user/Mosaic/.venv/bin/activate"' in content
        assert 'cd "/home/user/run_dir"' in content
        assert "python hallucinate.py" in content
        assert os.access(script, os.X_OK)
