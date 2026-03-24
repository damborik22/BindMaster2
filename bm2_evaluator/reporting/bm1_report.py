"""Bridge to BM1's binder-compare report pipeline.

Converts BM2 evaluator output to BM1-compatible format and calls
binder-compare report to generate HTML report, plots, top-20 CSV,
and PyMOL visualization script.

BM1's binder-compare is installed in the binder-eval conda env.
"""

from __future__ import annotations

import csv
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# BM1 binder-compare location
_BINDER_COMPARE_ENV = "binder-eval"


def generate_bm1_report(
    evaluation_dir: Path,
    output_dir: Path,
    target_pdb: Path | None = None,
    sequences_fasta: Path | None = None,
) -> bool:
    """Generate report using BM1's binder-compare pipeline.

    Args:
        evaluation_dir: BM2 evaluation output directory containing
            evaluation_summary.csv, evaluation_detail.csv, and PAE files.
        output_dir: Where to write the BM1 report (report.html, metrics.csv, etc.)
        target_pdb: Optional target PDB for native metrics attachment.
        sequences_fasta: Optional pre-built FASTA. If None, generated from summary CSV.

    Returns:
        True if report generation succeeded.
    """
    evaluation_dir = Path(evaluation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = evaluation_dir / "evaluation_detail.csv"
    summary_csv = evaluation_dir / "evaluation_summary.csv"

    if not detail_csv.exists():
        logger.error(f"evaluation_detail.csv not found in {evaluation_dir}")
        return False

    # Step 1: Split detail CSV into per-engine CSVs (BM1 format)
    boltz2_csv = output_dir / "boltz2_results.csv"
    af2_csv = output_dir / "af2_results.csv"
    _split_detail_csv(detail_csv, boltz2_csv, af2_csv)

    # Step 2: Generate sequences FASTA if not provided
    if sequences_fasta is None:
        sequences_fasta = output_dir / "sequences.fasta"
        _generate_fasta(summary_csv, sequences_fasta)

    # Step 3: Call binder-compare report
    cmd = _build_report_command(
        boltz2_csv=boltz2_csv,
        af2_csv=af2_csv,
        sequences_fasta=sequences_fasta,
        output_dir=output_dir,
    )

    logger.info(f"Running BM1 report: {' '.join(cmd[:6])}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"binder-compare report failed:\n{result.stderr[-500:]}")
            return False
        logger.info(f"Report generated: {output_dir}/report.html")
        return True
    except FileNotFoundError:
        logger.error(
            "binder-compare not found. Install: "
            "pip install -e ~/BindMaster/Evaluator in the binder-eval env"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("binder-compare report timed out")
        return False


def _split_detail_csv(
    detail_csv: Path,
    boltz2_csv: Path,
    af2_csv: Path,
) -> None:
    """Split BM2's evaluation_detail.csv into per-engine CSVs.

    BM2 detail CSV has one row per (design, engine) pair.
    BM1 expects separate CSVs per engine.
    """
    boltz2_rows = []
    af2_rows = []

    with open(detail_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            engine = row.get("engine", "")
            if "boltz" in engine.lower():
                boltz2_rows.append(row)
            elif "af2" in engine.lower():
                af2_rows.append(row)

    # Map BM2 column names to BM1 conventions
    col_map = {
        "design_id": "binder_id",
        "ipsae_min": "ipsae_min",
        "bt_ipsae": "bt_ipsae",
        "tb_ipsae": "tb_ipsae",
        "iptm": "iptm",
        "ptm": "ptm",
        "plddt_binder_mean_norm": "plddt_binder_mean",
        "plddt_binder_mean_raw": "plddt_binder_mean_raw",
        "plddt_target_mean_raw": "plddt_target_mean",
        "pae_interaction_mean": "pae_bt_mean",
        "pae_binder_mean": "pae_binder_mean",
        "refolded_structure_path": "pdb",
        "pae_matrix_path": "pae_file",
        "n_interface_contacts": "n_contacts",
    }

    def _remap_row(row: dict) -> dict:
        out = {}
        for bm2_col, bm1_col in col_map.items():
            if bm2_col in row:
                out[bm1_col] = row[bm2_col]
        # Pass through any columns not in the map
        for k, v in row.items():
            if k not in col_map and k not in ("engine", "chain_order"):
                out[k] = v
        return out

    def _write_engine_csv(rows: list[dict], path: Path, prefix: str) -> None:
        if not rows:
            # Write empty CSV with header
            path.write_text("binder_id\n")
            return
        remapped = [_remap_row(r) for r in rows]
        # Add engine prefix to PAE/PDB columns
        for r in remapped:
            if "pdb" in r:
                r[f"{prefix}_pdb"] = r.pop("pdb")
            if "pae_file" in r:
                r[f"{prefix}_pae_file"] = r.pop("pae_file")
        all_keys: list[str] = []
        for r in remapped:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(remapped)

    _write_engine_csv(boltz2_rows, boltz2_csv, "boltz")
    _write_engine_csv(af2_rows, af2_csv, "af2")

    logger.info(
        f"Split detail CSV: {len(boltz2_rows)} Boltz-2, {len(af2_rows)} AF2 rows"
    )


def _generate_fasta(summary_csv: Path, fasta_path: Path) -> None:
    """Generate sequences FASTA from BM2's evaluation_summary.csv.

    BM1 expects FASTA headers with source_tool tag:
    >binder_id source=tool_name
    SEQUENCE...
    """
    if not summary_csv.exists():
        fasta_path.write_text("")
        return

    lines: list[str] = []
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            binder_id = row.get("design_id", row.get("binder_id", ""))
            source = row.get("source_tool", "unknown")
            sequence = row.get("binder_sequence", row.get("sequence", ""))
            if binder_id and sequence:
                lines.append(f">{binder_id} source={source}")
                lines.append(sequence)

    fasta_path.write_text("\n".join(lines) + "\n" if lines else "")
    logger.info(f"Generated FASTA: {len(lines) // 2} sequences")


def _build_report_command(
    boltz2_csv: Path,
    af2_csv: Path,
    sequences_fasta: Path,
    output_dir: Path,
) -> list[str]:
    """Build the binder-compare report command using conda activation script."""
    # Write a small activation script (same pattern as tool launchers)
    script_path = output_dir / "run_bm1_report.sh"
    cmd = (
        f"binder-compare report "
        f"--boltz2-results {boltz2_csv} "
        f"--af2-results {af2_csv} "
        f"--sequences {sequences_fasta} "
        f"--output {output_dir}"
    )
    content = f"""\
#!/usr/bin/env bash
set -euo pipefail
set +u
_conda_found=false
for _conda_sh in \\
    "${{HOME}}/miniconda3/etc/profile.d/conda.sh" \\
    "${{HOME}}/miniforge3/etc/profile.d/conda.sh" \\
    "${{HOME}}/mambaforge/etc/profile.d/conda.sh" \\
    "${{HOME}}/BindMaster/conda/etc/profile.d/conda.sh" \\
    "${{HOME}}/anaconda3/etc/profile.d/conda.sh" \\
    "/opt/conda/etc/profile.d/conda.sh" \\
    "/opt/miniforge3/etc/profile.d/conda.sh"; do
    [[ -f "$_conda_sh" ]] && {{ source "$_conda_sh"; _conda_found=true; break; }}
done
[[ "$_conda_found" == true ]] || {{ echo "ERROR: conda not found" >&2; exit 1; }}
conda activate {_BINDER_COMPARE_ENV}
set -u
{cmd}
"""
    script_path.write_text(content)
    script_path.chmod(0o755)
    return ["bash", str(script_path)]
