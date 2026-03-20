"""PAE matrix handling: loading, chain ordering, submatrix extraction.

Chain ordering convention:
    Both Boltz2 and AF2 follow FASTA input order for PAE matrix
    row/column ordering. BM2 standardizes on target-first FASTA
    for all refolding (Step 2). Chain slices are computed from
    chain_lengths, not inferred from engine name.

PAE matrix format:
    - AF2/ColabFold: JSON with "pae" key, or pickle
    - Boltz2: .npz file, loaded via np.load(file)[key]
    - BM2 standard storage: .npy files

All functions operate on numpy arrays with shape (N_total, N_total)
where N_total = sum of all chain lengths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_pae_matrix(path: Path, source_format: str = "auto") -> np.ndarray:
    """Load a PAE matrix from various file formats.

    Args:
        path: Path to the PAE file.
        source_format: One of "auto", "npy", "npz", "json", "pkl".
            "auto" infers from file extension.

    Returns:
        2D numpy array of shape (N, N) with PAE values in Angstroms.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PAE file not found: {path}")

    if source_format == "auto":
        source_format = _infer_format(path)

    if source_format == "npy":
        return np.load(path)

    if source_format == "npz":
        data = np.load(path)
        # Try common keys used by Boltz2
        for key in ("pae", "predicted_aligned_error", "contact_probs"):
            if key in data:
                return data[key]
        # Fall back to first array
        keys = list(data.keys())
        if keys:
            logger.warning(
                f"No standard PAE key found in {path}, using first key: {keys[0]}"
            )
            return data[keys[0]]
        raise ValueError(f"No arrays found in npz file: {path}")

    if source_format == "json":
        with open(path) as f:
            data = json.load(f)

        # ColabFold scores JSON: {"pae": [[...]], ...}
        if isinstance(data, dict) and "pae" in data:
            return np.array(data["pae"], dtype=np.float64)

        # AlphaFold DB format: [{"predicted_aligned_error": [[...]]}]
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
            if isinstance(entry, dict) and "predicted_aligned_error" in entry:
                return np.array(
                    entry["predicted_aligned_error"], dtype=np.float64
                )

        # Direct 2D array
        if isinstance(data, list) and isinstance(data[0], list):
            return np.array(data, dtype=np.float64)

        raise ValueError(f"Cannot parse PAE from JSON structure in {path}")

    if source_format == "pkl":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            for key in (
                "predicted_aligned_error",
                "pae",
                "pae_output",
            ):
                if key in data:
                    return np.array(data[key], dtype=np.float64)

        raise ValueError(f"Cannot find PAE data in pickle file: {path}")

    raise ValueError(f"Unknown PAE format: {source_format}")


def save_pae_matrix(matrix: np.ndarray, path: Path) -> None:
    """Save PAE matrix as .npy (BM2 convention: always save as .npy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def get_chain_slices(chain_lengths: list[int]) -> list[slice]:
    """Convert chain lengths to array slices.

    Args:
        chain_lengths: [len_chain_1, len_chain_2, ...]
                       Order must match PAE matrix row/column order.

    Returns:
        List of slice objects for indexing into the PAE matrix.

    Example:
        chain_lengths = [290, 80]  # target 290 res, binder 80 res
        slices = get_chain_slices(chain_lengths)
        # slices[0] = slice(0, 290)    # target
        # slices[1] = slice(290, 370)  # binder
    """
    slices = []
    start = 0
    for length in chain_lengths:
        slices.append(slice(start, start + length))
        start += length
    return slices


def extract_interchain_pae(
    pae_matrix: np.ndarray,
    source_slice: slice,
    target_slice: slice,
) -> np.ndarray:
    """Extract the interchain PAE submatrix.

    Args:
        pae_matrix: Full PAE matrix.
        source_slice: Slice for source (aligned-on) residues.
        target_slice: Slice for target (scored) residues.

    Returns:
        2D array of shape (len_source, len_target).
    """
    return pae_matrix[source_slice, target_slice]


def validate_pae_matrix(
    pae_matrix: np.ndarray,
    expected_size: int | None = None,
) -> list[str]:
    """Validate PAE matrix shape and value range.

    Returns list of warnings (empty = valid).
    """
    warnings = []

    if pae_matrix.ndim != 2:
        warnings.append(f"PAE matrix is {pae_matrix.ndim}D, expected 2D")
        return warnings

    if pae_matrix.shape[0] != pae_matrix.shape[1]:
        warnings.append(
            f"PAE matrix is not square: {pae_matrix.shape}"
        )

    if expected_size is not None and pae_matrix.shape[0] != expected_size:
        warnings.append(
            f"PAE matrix size {pae_matrix.shape[0]} != expected {expected_size}"
        )

    if np.any(np.isnan(pae_matrix)):
        warnings.append("PAE matrix contains NaN values")

    if np.any(pae_matrix < 0):
        warnings.append("PAE matrix contains negative values")

    max_val = np.max(pae_matrix)
    if max_val > 35.0:
        warnings.append(
            f"PAE matrix max value {max_val:.1f} exceeds expected range (0-31.75)"
        )

    return warnings


def _infer_format(path: Path) -> str:
    """Infer PAE file format from extension."""
    ext = path.suffix.lower()
    format_map = {
        ".npy": "npy",
        ".npz": "npz",
        ".json": "json",
        ".pkl": "pkl",
        ".pickle": "pkl",
    }
    fmt = format_map.get(ext)
    if fmt is None:
        raise ValueError(
            f"Cannot infer PAE format from extension '{ext}'. "
            f"Supported: {list(format_map.keys())}"
        )
    return fmt
