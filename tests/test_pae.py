"""Tests for PAE matrix handling."""

import json

import numpy as np
import pytest

from bm2_evaluator.metrics.pae import (
    load_pae_matrix,
    save_pae_matrix,
    get_chain_slices,
    extract_interchain_pae,
    validate_pae_matrix,
)


class TestLoadPaeMatrix:
    def test_load_npy(self, pae_npy_file, strong_pae_matrix):
        mat = load_pae_matrix(pae_npy_file)
        np.testing.assert_array_equal(mat, strong_pae_matrix)

    def test_load_npz(self, pae_npz_file, strong_pae_matrix):
        mat = load_pae_matrix(pae_npz_file)
        np.testing.assert_array_equal(mat, strong_pae_matrix)

    def test_load_json_colabfold(self, pae_json_file, strong_pae_matrix):
        mat = load_pae_matrix(pae_json_file)
        np.testing.assert_array_almost_equal(mat, strong_pae_matrix)

    def test_load_json_alphafold_db_format(self, tmp_path, strong_pae_matrix):
        """AlphaFold DB format: [{"predicted_aligned_error": [[...]]}]"""
        path = tmp_path / "pae.json"
        data = [{"predicted_aligned_error": strong_pae_matrix.tolist(),
                 "max_predicted_aligned_error": 31.75}]
        path.write_text(json.dumps(data))
        mat = load_pae_matrix(path)
        np.testing.assert_array_almost_equal(mat, strong_pae_matrix)

    def test_auto_format_detection(self, pae_npy_file, strong_pae_matrix):
        mat = load_pae_matrix(pae_npy_file, source_format="auto")
        np.testing.assert_array_equal(mat, strong_pae_matrix)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pae_matrix(tmp_path / "nonexistent.npy")

    def test_unknown_format(self, tmp_path):
        path = tmp_path / "pae.xyz"
        path.write_text("not a pae file")
        with pytest.raises(ValueError, match="Cannot infer"):
            load_pae_matrix(path)


class TestSavePaeMatrix:
    def test_save_and_reload(self, tmp_path, strong_pae_matrix):
        path = tmp_path / "saved_pae.npy"
        save_pae_matrix(strong_pae_matrix, path)
        assert path.exists()
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, strong_pae_matrix)

    def test_creates_parent_dirs(self, tmp_path, strong_pae_matrix):
        path = tmp_path / "sub" / "dir" / "pae.npy"
        save_pae_matrix(strong_pae_matrix, path)
        assert path.exists()


class TestGetChainSlices:
    def test_two_chains(self):
        slices = get_chain_slices([290, 80])
        assert slices[0] == slice(0, 290)
        assert slices[1] == slice(290, 370)

    def test_three_chains(self):
        slices = get_chain_slices([100, 200, 50])
        assert slices[0] == slice(0, 100)
        assert slices[1] == slice(100, 300)
        assert slices[2] == slice(300, 350)

    def test_single_chain(self):
        slices = get_chain_slices([100])
        assert slices[0] == slice(0, 100)


class TestExtractInterchainPae:
    def test_correct_submatrix(self, strong_pae_matrix):
        target_slice = slice(0, 10)
        binder_slice = slice(10, 15)
        sub = extract_interchain_pae(
            strong_pae_matrix, target_slice, binder_slice
        )
        assert sub.shape == (10, 5)
        # In strong_pae_matrix, interchain PAE = 5.0
        assert np.all(sub == 5.0)

    def test_shape_validation(self):
        mat = np.zeros((20, 20))
        sub = extract_interchain_pae(mat, slice(0, 8), slice(8, 20))
        assert sub.shape == (8, 12)


class TestValidatePaeMatrix:
    def test_valid_matrix(self, strong_pae_matrix):
        warnings = validate_pae_matrix(strong_pae_matrix)
        assert warnings == []

    def test_wrong_dimensions(self):
        mat = np.zeros((5, 5, 5))
        warnings = validate_pae_matrix(mat)
        assert any("2D" in w for w in warnings)

    def test_non_square(self):
        mat = np.zeros((5, 10))
        warnings = validate_pae_matrix(mat)
        assert any("not square" in w for w in warnings)

    def test_expected_size_mismatch(self, strong_pae_matrix):
        warnings = validate_pae_matrix(strong_pae_matrix, expected_size=20)
        assert any("expected 20" in w for w in warnings)

    def test_nan_values(self):
        mat = np.zeros((5, 5))
        mat[2, 3] = np.nan
        warnings = validate_pae_matrix(mat)
        assert any("NaN" in w for w in warnings)

    def test_negative_values(self):
        mat = np.zeros((5, 5))
        mat[1, 2] = -1.0
        warnings = validate_pae_matrix(mat)
        assert any("negative" in w for w in warnings)

    def test_high_values_warning(self):
        mat = np.full((5, 5), 40.0)
        warnings = validate_pae_matrix(mat)
        assert any("exceeds expected range" in w for w in warnings)
