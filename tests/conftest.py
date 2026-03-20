"""Shared test fixtures for BM2 Evaluator tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bm2_evaluator.core.models import EvalConfig, IngestedDesign, SourceTool

# ---------------------------------------------------------------------------
# Minimal PDB content for testing (two chains: A=target 10 res, B=binder 5 res)
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
ATOM      9  N   LEU A   3       6.000  -1.000   1.000  1.00 88.00           N
ATOM     10  CA  LEU A   3       7.000  -1.000   1.000  1.00 88.00           C
ATOM     11  C   LEU A   3       8.000  -1.000   1.000  1.00 88.00           C
ATOM     12  O   LEU A   3       8.500   0.000   1.000  1.00 88.00           O
ATOM     13  N   VAL A   4       8.500  -2.000   1.000  1.00 92.00           N
ATOM     14  CA  VAL A   4       9.500  -2.000   1.000  1.00 92.00           C
ATOM     15  C   VAL A   4      10.500  -2.000   1.000  1.00 92.00           C
ATOM     16  O   VAL A   4      11.000  -1.000   1.000  1.00 92.00           O
ATOM     17  N   ILE A   5      11.000  -3.000   1.000  1.00 87.00           N
ATOM     18  CA  ILE A   5      12.000  -3.000   1.000  1.00 87.00           C
ATOM     19  C   ILE A   5      13.000  -3.000   1.000  1.00 87.00           C
ATOM     20  O   ILE A   5      13.500  -2.000   1.000  1.00 87.00           O
ATOM     21  N   PHE A   6      13.500  -4.000   1.000  1.00 91.00           N
ATOM     22  CA  PHE A   6      14.500  -4.000   1.000  1.00 91.00           C
ATOM     23  C   PHE A   6      15.500  -4.000   1.000  1.00 91.00           C
ATOM     24  O   PHE A   6      16.000  -3.000   1.000  1.00 91.00           O
ATOM     25  N   ASP A   7      16.000  -5.000   1.000  1.00 83.00           N
ATOM     26  CA  ASP A   7      17.000  -5.000   1.000  1.00 83.00           C
ATOM     27  C   ASP A   7      18.000  -5.000   1.000  1.00 83.00           C
ATOM     28  O   ASP A   7      18.500  -4.000   1.000  1.00 83.00           O
ATOM     29  N   LYS A   8      18.500  -6.000   1.000  1.00 89.00           N
ATOM     30  CA  LYS A   8      19.500  -6.000   1.000  1.00 89.00           C
ATOM     31  C   LYS A   8      20.500  -6.000   1.000  1.00 89.00           C
ATOM     32  O   LYS A   8      21.000  -5.000   1.000  1.00 89.00           O
ATOM     33  N   GLU A   9      21.000  -7.000   1.000  1.00 86.00           N
ATOM     34  CA  GLU A   9      22.000  -7.000   1.000  1.00 86.00           C
ATOM     35  C   GLU A   9      23.000  -7.000   1.000  1.00 86.00           C
ATOM     36  O   GLU A   9      23.500  -6.000   1.000  1.00 86.00           O
ATOM     37  N   ARG A  10      23.500  -8.000   1.000  1.00 84.00           N
ATOM     38  CA  ARG A  10      24.500  -8.000   1.000  1.00 84.00           C
ATOM     39  C   ARG A  10      25.500  -8.000   1.000  1.00 84.00           C
ATOM     40  O   ARG A  10      26.000  -7.000   1.000  1.00 84.00           O
TER
ATOM     41  N   MET B   1      30.000  30.000  30.000  1.00 82.00           N
ATOM     42  CA  MET B   1      31.000  30.000  30.000  1.00 82.00           C
ATOM     43  C   MET B   1      32.000  30.000  30.000  1.00 82.00           C
ATOM     44  O   MET B   1      32.500  31.000  30.000  1.00 82.00           O
ATOM     45  N   LYS B   2      32.500  29.000  30.000  1.00 79.00           N
ATOM     46  CA  LYS B   2      33.500  29.000  30.000  1.00 79.00           C
ATOM     47  C   LYS B   2      34.500  29.000  30.000  1.00 79.00           C
ATOM     48  O   LYS B   2      35.000  30.000  30.000  1.00 79.00           O
ATOM     49  N   TRP B   3      35.000  28.000  30.000  1.00 88.00           N
ATOM     50  CA  TRP B   3      36.000  28.000  30.000  1.00 88.00           C
ATOM     51  C   TRP B   3      37.000  28.000  30.000  1.00 88.00           C
ATOM     52  O   TRP B   3      37.500  29.000  30.000  1.00 88.00           O
ATOM     53  N   ALA B   4      37.500  27.000  30.000  1.00 91.00           N
ATOM     54  CA  ALA B   4      38.500  27.000  30.000  1.00 91.00           C
ATOM     55  C   ALA B   4      39.500  27.000  30.000  1.00 91.00           C
ATOM     56  O   ALA B   4      40.000  28.000  30.000  1.00 91.00           O
ATOM     57  N   SER B   5      40.000  26.000  30.000  1.00 85.00           N
ATOM     58  CA  SER B   5      41.000  26.000  30.000  1.00 85.00           C
ATOM     59  C   SER B   5      42.000  26.000  30.000  1.00 85.00           C
ATOM     60  O   SER B   5      42.500  27.000  30.000  1.00 85.00           O
TER
END
"""


@pytest.fixture
def eval_config():
    """Default EvalConfig for testing."""
    return EvalConfig()


@pytest.fixture
def strong_pae_matrix():
    """PAE matrix representing a strong interaction.

    Size: 15 total (target=10, binder=5).
    Intrachain PAE = 2.0, interchain PAE = 5.0, diagonal = 0.
    All interchain PAE < 15A cutoff -> all qualify -> high ipSAE.
    """
    n = 15
    mat = np.full((n, n), 2.0)
    # Interchain blocks
    mat[:10, 10:] = 5.0  # target->binder
    mat[10:, :10] = 5.0  # binder->target
    np.fill_diagonal(mat, 0.0)
    return mat


@pytest.fixture
def weak_pae_matrix():
    """PAE matrix with high interchain PAE.

    Size: 15 total (target=10, binder=5).
    Intrachain PAE = 3.0, interchain PAE = 25.0.
    Most interchain PAE > 15A cutoff -> few/no qualify -> low ipSAE.
    """
    n = 15
    mat = np.full((n, n), 3.0)
    mat[:10, 10:] = 25.0
    mat[10:, :10] = 25.0
    np.fill_diagonal(mat, 0.0)
    return mat


@pytest.fixture
def minimal_pdb(tmp_path):
    """Create a minimal PDB file with 2 chains."""
    pdb_path = tmp_path / "test_complex.pdb"
    pdb_path.write_text(MINIMAL_PDB)
    return pdb_path


@pytest.fixture
def sample_ingested_design(minimal_pdb):
    """A minimal IngestedDesign for testing."""
    return IngestedDesign(
        design_id="test_001",
        source_tool=SourceTool.GENERIC,
        binder_sequence="MKWAS",
        binder_chain="B",
        target_sequence="AGLVIFDKER",
        target_chain="A",
        binder_length=5,
        target_length=10,
        complex_structure_path=minimal_pdb,
    )


@pytest.fixture
def bindcraft_output_dir(tmp_path):
    """Create a mock BindCraft output directory."""
    final_dir = tmp_path / "final_designs"
    final_dir.mkdir()
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()

    # Write PDB files
    (final_dir / "design_001.pdb").write_text(MINIMAL_PDB)
    (final_dir / "design_002.pdb").write_text(MINIMAL_PDB)

    # Write scores CSV
    with open(scores_dir / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "design_name", "plddt", "ptm", "i_ptm", "pae", "i_pae",
                "binder_plddt", "binder_rmsd", "dG", "dSASA",
                "shape_complementarity", "n_hbonds", "sequence",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "design_name": "design_001",
            "plddt": "87.5", "ptm": "0.82", "i_ptm": "0.75",
            "pae": "4.2", "i_pae": "6.1",
            "binder_plddt": "85.0", "binder_rmsd": "1.2",
            "dG": "-15.3", "dSASA": "1200.5",
            "shape_complementarity": "0.68", "n_hbonds": "5",
            "sequence": "MKWAS",
        })
        writer.writerow({
            "design_name": "design_002",
            "plddt": "82.1", "ptm": "0.78", "i_ptm": "0.65",
            "pae": "5.8", "i_pae": "8.3",
            "binder_plddt": "80.2", "binder_rmsd": "1.8",
            "dG": "-12.1", "dSASA": "950.2",
            "shape_complementarity": "0.62", "n_hbonds": "3",
            "sequence": "MKWAS",
        })

    return tmp_path


@pytest.fixture
def rfdiffusion_output_dir(tmp_path):
    """Create a mock RFdiffusion + LigandMPNN output directory."""
    seqs_dir = tmp_path / "seqs"
    seqs_dir.mkdir()

    # Write backbone PDB (chain A=binder, B=target for RFdiffusion)
    (tmp_path / "design_0.pdb").write_text(MINIMAL_PDB)

    # Write a .trb file (empty pickle dict for auto-detection)
    import pickle
    with open(tmp_path / "design_0.trb", "wb") as f:
        pickle.dump({"config": {}, "plddt": 0.85}, f)

    # Write FASTA with LigandMPNN scores
    (seqs_dir / "design_0.fa").write_text(
        ">design_0, score=1.234, global_score=1.456, "
        "fixed_chains=['B'], designed_chains=['A'], seed=37\n"
        "MKWAS\n"
        ">T=0.1, sample=1, score=1.123, global_score=1.345, seq_recovery=0.0\n"
        "MKWAS\n"
    )

    return tmp_path


@pytest.fixture
def generic_output_dir(tmp_path):
    """Create a generic output directory with PDB files."""
    (tmp_path / "complex_001.pdb").write_text(MINIMAL_PDB)
    (tmp_path / "complex_002.pdb").write_text(MINIMAL_PDB)
    return tmp_path


@pytest.fixture
def pae_npy_file(tmp_path, strong_pae_matrix):
    """Create a .npy PAE matrix file."""
    path = tmp_path / "test_pae.npy"
    np.save(path, strong_pae_matrix)
    return path


@pytest.fixture
def pae_npz_file(tmp_path, strong_pae_matrix):
    """Create a .npz PAE matrix file (Boltz2 format)."""
    path = tmp_path / "test_pae.npz"
    np.savez(path, pae=strong_pae_matrix)
    return path


@pytest.fixture
def pae_json_file(tmp_path, strong_pae_matrix):
    """Create a JSON PAE file (ColabFold format)."""
    path = tmp_path / "test_scores.json"
    data = {
        "pae": strong_pae_matrix.tolist(),
        "plddt": [0.9] * 15,
        "ptm": 0.85,
        "iptm": 0.78,
        "max_pae": 31.75,
    }
    path.write_text(json.dumps(data))
    return path
