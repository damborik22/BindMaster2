"""Tests for pLDDT normalization."""

import numpy as np
import pytest

from bm2_evaluator.metrics.plddt import (
    normalize_plddt,
    plddt_scale_for_tool,
    detect_plddt_scale,
)


class TestNormalizePlddt:
    def test_scale_0_100(self):
        raw, norm = normalize_plddt(85.0, "0-100")
        assert raw == 85.0
        assert norm == 0.85

    def test_scale_0_100_max(self):
        raw, norm = normalize_plddt(100.0, "0-100")
        assert raw == 100.0
        assert norm == 1.0

    def test_scale_0_100_zero(self):
        raw, norm = normalize_plddt(0.0, "0-100")
        assert raw == 0.0
        assert norm == 0.0

    def test_scale_0_1(self):
        raw, norm = normalize_plddt(0.85, "0-1")
        assert raw == 0.85
        assert norm == 0.85

    def test_scale_0_1_max(self):
        raw, norm = normalize_plddt(1.0, "0-1")
        assert raw == 1.0
        assert norm == 1.0

    def test_out_of_range_0_100_high(self):
        with pytest.raises(ValueError, match="outside 0-100"):
            normalize_plddt(101.0, "0-100")

    def test_out_of_range_0_100_low(self):
        with pytest.raises(ValueError, match="outside 0-100"):
            normalize_plddt(-1.0, "0-100")

    def test_out_of_range_0_1(self):
        with pytest.raises(ValueError, match="outside 0-1"):
            normalize_plddt(1.1, "0-1")

    def test_unknown_scale(self):
        with pytest.raises(ValueError, match="Unknown pLDDT scale"):
            normalize_plddt(0.5, "unknown")


class TestPlddtScaleForTool:
    def test_bindcraft(self):
        assert plddt_scale_for_tool("bindcraft") == "0-100"

    def test_boltzgen(self):
        assert plddt_scale_for_tool("boltzgen") == "0-1"

    def test_boltz2(self):
        assert plddt_scale_for_tool("boltz2") == "0-1"

    def test_af2(self):
        assert plddt_scale_for_tool("af2") == "0-100"

    def test_pxdesign_ptx(self):
        assert plddt_scale_for_tool("pxdesign_ptx") == "0-1"

    def test_pxdesign_af2(self):
        assert plddt_scale_for_tool("pxdesign_af2") == "0-100"

    def test_unknown_defaults_to_0_100(self):
        assert plddt_scale_for_tool("some_new_tool") == "0-100"


class TestDetectPlddtScale:
    def test_detect_0_1(self):
        assert detect_plddt_scale([0.5, 0.8, 0.9]) == "0-1"

    def test_detect_0_100(self):
        assert detect_plddt_scale([50, 80, 90]) == "0-100"

    def test_edge_case_1_0(self):
        assert detect_plddt_scale([0.95, 1.0]) == "0-1"

    def test_edge_case_1_01(self):
        assert detect_plddt_scale([0.95, 1.01]) == "0-100"

    def test_empty(self):
        assert detect_plddt_scale([]) == "0-100"

    def test_numpy_array(self):
        assert detect_plddt_scale(np.array([0.5, 0.8])) == "0-1"
