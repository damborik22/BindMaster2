"""Tests for configuration loading and validation."""

import os
from pathlib import Path

import pytest

from bm2_evaluator.core.config import (
    load_config,
    save_config,
    validate_config,
    resolve_output_dir,
    pae_matrix_path,
)
from bm2_evaluator.core.models import EvalConfig


class TestLoadConfig:
    def test_defaults(self):
        config = load_config()
        assert config.pae_cutoff == 10.0
        assert config.engines == ["boltz2", "af2"]

    def test_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[eval]\npae_cutoff = 10.0\nengines = ["boltz2"]\n'
        )
        config = load_config(toml_path)
        assert config.pae_cutoff == 10.0
        assert config.engines == ["boltz2"]

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("BM2_PAE_CUTOFF", "12.0")
        monkeypatch.setenv("BM2_ENGINES", "boltz2,af2")
        monkeypatch.setenv("BM2_N_WORKERS", "4")
        config = load_config()
        assert config.pae_cutoff == 12.0
        assert config.engines == ["boltz2", "af2"]
        assert config.n_workers == 4

    def test_env_ipsae_threshold(self, monkeypatch):
        monkeypatch.setenv("BM2_IPSAE_THRESHOLD", "0.55")
        config = load_config()
        assert config.ipsae_consensus_threshold == 0.55


class TestValidateConfig:
    def test_valid(self):
        config = EvalConfig()
        assert validate_config(config) == []

    def test_invalid_pae_cutoff(self):
        config = EvalConfig(pae_cutoff=-1.0)
        errors = validate_config(config)
        assert len(errors) == 1
        assert "pae_cutoff" in errors[0]

    def test_invalid_threshold(self):
        config = EvalConfig(ipsae_consensus_threshold=1.5)
        errors = validate_config(config)
        assert any("ipsae_consensus_threshold" in e for e in errors)

    def test_empty_engines(self):
        config = EvalConfig(engines=[])
        errors = validate_config(config)
        assert any("engines" in e for e in errors)

    def test_invalid_workers(self):
        config = EvalConfig(n_workers=0)
        errors = validate_config(config)
        assert any("n_workers" in e for e in errors)


class TestSaveConfig:
    def test_round_trip(self, tmp_path):
        original = EvalConfig(pae_cutoff=12.0, n_workers=4)
        path = tmp_path / "config.toml"
        save_config(original, path)
        loaded = load_config(path)
        assert loaded.pae_cutoff == 12.0
        assert loaded.n_workers == 4


class TestPathHelpers:
    def test_resolve_output_dir(self, tmp_path):
        config = EvalConfig(output_dir=tmp_path / "eval_output")
        out = resolve_output_dir(config, "run_001")
        assert out == tmp_path / "eval_output" / "run_001"
        assert out.is_dir()

    def test_pae_matrix_path(self, tmp_path):
        path = pae_matrix_path(tmp_path, "design_001", "boltz2")
        assert path == tmp_path / "pae" / "boltz2" / "design_001.npy"
        assert path.parent.is_dir()
