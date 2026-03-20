"""CSV export with raw + normalized columns.

Two levels:
  - Summary: one row per design (evaluation_summary.csv)
  - Detail: one row per engine per design (evaluation_detail.csv)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = [
    "rank",
    "design_id",
    "source_tool",
    "binder_length",
    "tier",
    "composite_score",
    "ensemble_ipsae_min",
    "ensemble_iptm",
    "ensemble_plddt",
    "multi_model_agreement",
    "boltz2_ipsae_min",
    "af2_ipsae_min",
    "boltz2_iptm",
    "af2_iptm",
    "boltz2_plddt_raw",
    "af2_plddt_raw",
    "boltz2_plddt_norm",
    "af2_plddt_norm",
    "monomer_rmsd",
    "monomer_passes",
    "rosetta_dG",
    "rosetta_dSASA",
    "rosetta_shape_comp",
    "rosetta_n_hbonds",
    "binder_sequence",
]

DETAIL_COLUMNS = [
    "design_id",
    "engine",
    "bt_ipsae",
    "tb_ipsae",
    "ipsae_min",
    "ipsae_max",
    "iptm",
    "ptm",
    "plddt_binder_mean_raw",
    "plddt_binder_mean_norm",
    "plddt_binder_min_raw",
    "plddt_target_mean_raw",
    "plddt_scale_max",
    "pae_interaction_mean",
    "pae_binder_mean",
    "n_interface_contacts",
    "pae_matrix_path",
    "structure_path",
]


def export_summary_csv(
    scored_designs: list[dict],
    output_path: Path,
) -> None:
    """Export per-design summary CSV.

    Raw AND normalized columns for pLDDT. User can verify.
    None/null for missing values (engine failed, Rosetta not available).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()

        for d in scored_designs:
            row = _build_summary_row(d)
            writer.writerow(row)

    logger.info(f"Summary CSV: {output_path} ({len(scored_designs)} designs)")


def export_detail_csv(
    scored_designs: list[dict],
    output_path: Path,
) -> None:
    """Export per-engine detail CSV.

    One row per (design, engine) pair.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=DETAIL_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()

        for d in scored_designs:
            engine_results = d.get("engine_results", {})
            for engine_name, er in engine_results.items():
                row = {
                    "design_id": d.get("design_id", ""),
                    "engine": engine_name,
                    "bt_ipsae": _fmt(er.get("bt_ipsae")),
                    "tb_ipsae": _fmt(er.get("tb_ipsae")),
                    "ipsae_min": _fmt(er.get("ipsae_min")),
                    "ipsae_max": _fmt(er.get("ipsae_max")),
                    "iptm": _fmt(er.get("iptm")),
                    "ptm": _fmt(er.get("ptm")),
                    "plddt_binder_mean_raw": _fmt(er.get("plddt_binder_mean_raw")),
                    "plddt_binder_mean_norm": _fmt(er.get("plddt_binder_norm")),
                    "plddt_binder_min_raw": _fmt(er.get("plddt_binder_min_raw")),
                    "plddt_target_mean_raw": _fmt(er.get("plddt_target_mean_raw")),
                    "plddt_scale_max": _fmt(er.get("plddt_scale_max")),
                    "pae_interaction_mean": _fmt(er.get("pae_interaction_mean")),
                    "pae_binder_mean": _fmt(er.get("pae_binder_mean")),
                    "n_interface_contacts": er.get("n_interface_contacts", ""),
                    "pae_matrix_path": er.get("pae_matrix_path", ""),
                    "structure_path": er.get("structure_path", ""),
                }
                writer.writerow(row)

    logger.info(f"Detail CSV: {output_path}")


def _build_summary_row(d: dict) -> dict:
    """Build a summary CSV row from a scored design dict."""
    er = d.get("engine_results", {})
    rosetta = d.get("rosetta", {}) or {}

    row = {
        "rank": d.get("rank", ""),
        "design_id": d.get("design_id", ""),
        "source_tool": d.get("source_tool", ""),
        "binder_length": d.get("binder_length", ""),
        "tier": d.get("tier", ""),
        "composite_score": _fmt(d.get("composite_score")),
        "ensemble_ipsae_min": _fmt(d.get("ensemble_ipsae_min")),
        "ensemble_iptm": _fmt(d.get("ensemble_iptm")),
        "ensemble_plddt": _fmt(d.get("ensemble_plddt")),
        "multi_model_agreement": _fmt(d.get("multi_model_agreement")),
        "monomer_rmsd": _fmt(d.get("monomer_rmsd")),
        "monomer_passes": d.get("monomer_passes", ""),
        "rosetta_dG": _fmt(rosetta.get("dG")),
        "rosetta_dSASA": _fmt(rosetta.get("dSASA")),
        "rosetta_shape_comp": _fmt(rosetta.get("shape_complementarity")),
        "rosetta_n_hbonds": rosetta.get("n_hbonds", ""),
        "binder_sequence": d.get("binder_sequence", ""),
    }

    # Per-engine columns
    for engine in ["boltz2", "af2"]:
        engine_data = er.get(engine, {})
        row[f"{engine}_ipsae_min"] = _fmt(engine_data.get("ipsae_min"))
        row[f"{engine}_iptm"] = _fmt(engine_data.get("iptm"))
        row[f"{engine}_plddt_raw"] = _fmt(engine_data.get("plddt_binder_mean_raw"))
        row[f"{engine}_plddt_norm"] = _fmt(engine_data.get("plddt_binder_norm"))

    return row


def _fmt(val) -> str:
    """Format a value for CSV output. None -> empty string."""
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)
