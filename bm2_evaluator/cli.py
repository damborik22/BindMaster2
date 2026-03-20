"""BM2 Evaluator CLI.

Commands:
    bm2-eval score    Score designs through cross-model refolding
    bm2-eval report   Generate report from existing evaluation
    bm2-eval compare  Compare two evaluation runs
    bm2-eval export   Export top designs as FASTA/CSV/JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bm2_evaluator import __version__

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        prog="bm2-eval",
        description="BM2 Evaluator: standalone cross-model protein binder evaluation",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- score ---
    score_p = subparsers.add_parser("score", help="Score designs")
    score_p.add_argument(
        "--designs",
        required=True,
        nargs="+",
        help="Design directories (one or more)",
    )
    score_p.add_argument(
        "--parser",
        nargs="*",
        help="Parser per design dir (bindcraft, boltzgen, etc). Auto-detect if omitted.",
    )
    score_p.add_argument(
        "--target", required=True, help="Target PDB file"
    )
    score_p.add_argument(
        "--chain", required=True, help="Target chain ID"
    )
    score_p.add_argument(
        "--output",
        default="./bm2_evaluation",
        help="Output directory (default: ./bm2_evaluation)",
    )
    score_p.add_argument(
        "--engines",
        default="boltz2,af2",
        help="Comma-separated refolding engines (default: boltz2,af2)",
    )
    score_p.add_argument(
        "--rosetta",
        action="store_true",
        help="Enable PyRosetta scoring",
    )
    score_p.add_argument(
        "--pae-cutoff",
        type=float,
        default=15.0,
        help="PAE cutoff for ipSAE in Angstroms (default: 15.0, Dunbrack)",
    )
    score_p.add_argument(
        "--ipsae-threshold",
        type=float,
        default=0.61,
        help="ipSAE_min threshold for consensus tier (default: 0.61)",
    )
    score_p.add_argument(
        "--no-monomer",
        action="store_true",
        help="Skip monomer validation",
    )
    score_p.add_argument(
        "--top", type=int, default=None, help="Only report top N designs"
    )

    # --- report ---
    report_p = subparsers.add_parser(
        "report", help="Generate report from existing evaluation"
    )
    report_p.add_argument(
        "--eval-dir", required=True, help="Evaluation output directory"
    )
    report_p.add_argument(
        "--top", type=int, default=20, help="Number of top designs to show"
    )

    # --- compare ---
    compare_p = subparsers.add_parser(
        "compare", help="Compare two evaluation runs"
    )
    compare_p.add_argument("--run1", required=True, help="First eval dir")
    compare_p.add_argument("--run2", required=True, help="Second eval dir")

    # --- export ---
    export_p = subparsers.add_parser(
        "export", help="Export top designs as FASTA, CSV, or JSON"
    )
    export_p.add_argument(
        "--eval-dir", required=True, help="Evaluation output directory"
    )
    export_p.add_argument(
        "--top", required=True, type=int, help="Number of top designs"
    )
    export_p.add_argument(
        "--format",
        dest="fmt",
        choices=["fasta", "csv", "json"],
        default="fasta",
        help="Export format (default: fasta)",
    )
    export_p.add_argument(
        "--output", default=None, help="Output file path"
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.command == "score":
        _cmd_score(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "export":
        _cmd_export(args)
    else:
        parser.print_help()
        sys.exit(0)


def _cmd_score(args):
    """Run the full scoring pipeline."""
    from bm2_evaluator.core.config import load_config
    from bm2_evaluator.core.models import EvalConfig
    from bm2_evaluator.ingestion import auto_detect, get_ingestor
    from bm2_evaluator.scoring.composite import composite_basic, composite_with_rosetta
    from bm2_evaluator.scoring.ranking import (
        compute_ensemble_metrics,
        compute_multi_model_agreement,
        rank_designs,
    )
    from bm2_evaluator.scoring.tiers import classify_tier
    from bm2_evaluator.reporting.csv_export import export_detail_csv, export_summary_csv
    from bm2_evaluator.reporting.text_report import generate_report
    from bm2_evaluator.reporting.comparison import compare_tools

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. INGEST
    parsers = args.parser or [None] * len(args.designs)
    if len(parsers) < len(args.designs):
        parsers.extend([None] * (len(args.designs) - len(parsers)))

    all_designs = []
    for design_dir, parser_name in zip(args.designs, parsers):
        design_path = Path(design_dir)
        if parser_name is None:
            parser_name = auto_detect(design_path)
            logger.info(f"Auto-detected parser: {parser_name} for {design_path}")

        ingestor = get_ingestor(parser_name)
        designs = ingestor.ingest(design_path, target_chain=args.chain)
        all_designs.extend(designs)

    logger.info(
        f"Ingested {len(all_designs)} designs from {len(args.designs)} sources"
    )

    if not all_designs:
        logger.error("No designs found. Check --designs paths and --parser options.")
        sys.exit(1)

    # 2. REFOLD (if engines available)
    engine_names = [e.strip() for e in args.engines.split(",")]
    logger.info(f"Refolding engines requested: {engine_names}")
    logger.info(
        "Note: refolding requires engine environments to be available. "
        "Skipping refolding if engines are not installed."
    )

    # Build scored design dicts (without refolding for now —
    # refolding requires Step 2 engine envs to be set up)
    scored = []
    for design in all_designs:
        d = {
            "design_id": design.design_id,
            "source_tool": design.source_tool.value,
            "binder_length": design.binder_length,
            "binder_sequence": design.binder_sequence,
            "engine_results": {},
            "tier": "unscored",
            "composite_score": 0.0,
            "ensemble_ipsae_min": 0.0,
            "ensemble_iptm": 0.0,
            "ensemble_plddt": 0.0,
            "multi_model_agreement": 0.0,
            "monomer_rmsd": None,
            "monomer_passes": None,
            "rosetta": None,
        }

        # If tool_metrics include ipsae/iptm from the tool itself,
        # use those for scoring even without refolding
        tm = design.tool_metrics
        if tm:
            d["engine_results"]["tool_native"] = {
                "ipsae_min": tm.get("ipsae_min", 0.0),
                "iptm": tm.get("i_ptm", tm.get("iptm", 0.0)),
                "plddt_binder_norm": tm.get("binder_plddt", 0.0) / 100.0
                if tm.get("binder_plddt", 0) > 1
                else tm.get("binder_plddt", 0.0),
            }

        # Compute ensemble metrics
        if d["engine_results"]:
            ensemble = compute_ensemble_metrics(d["engine_results"])
            d.update(ensemble)
            d["multi_model_agreement"] = compute_multi_model_agreement(
                d["engine_results"], args.ipsae_threshold
            )
            d["tier"] = classify_tier(d["engine_results"])
            d["composite_score"] = composite_basic(
                d["ensemble_ipsae_min"],
                d["ensemble_iptm"],
                d["multi_model_agreement"],
                d["ensemble_plddt"],
                0.0,  # No PAE interaction without refolding
            )

        scored.append(d)

    # 3. RANK
    scored = rank_designs(scored)

    # 4. EXPORT
    target_info = {
        "name": Path(args.target).stem,
        "chain": args.chain,
        "n_residues": "?",
    }
    eval_config = {
        "engines": engine_names,
        "pae_cutoff": args.pae_cutoff,
        "ipsae_consensus_threshold": args.ipsae_threshold,
        "use_rosetta": args.rosetta,
    }

    export_summary_csv(scored, output_dir / "evaluation_summary.csv")
    export_detail_csv(scored, output_dir / "evaluation_detail.csv")
    report = generate_report(
        scored, target_info, eval_config, output_dir / "report.txt"
    )
    print(report)

    comparison = compare_tools(scored)
    print("\n" + comparison)

    # Save config for reproducibility
    config_path = output_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)

    logger.info(f"Evaluation complete. Results in {output_dir}")


def _cmd_report(args):
    """Generate report from existing evaluation."""
    import csv

    eval_dir = Path(args.eval_dir)
    summary_csv = eval_dir / "evaluation_summary.csv"

    if not summary_csv.exists():
        logger.error(f"No evaluation_summary.csv found in {eval_dir}")
        sys.exit(1)

    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        scored = list(reader)

    # Convert numeric fields
    for d in scored:
        for key in ["composite_score", "ensemble_ipsae_min", "ensemble_iptm",
                     "ensemble_plddt", "multi_model_agreement", "monomer_rmsd"]:
            if d.get(key):
                try:
                    d[key] = float(d[key])
                except (ValueError, TypeError):
                    d[key] = 0.0

    config_path = eval_dir / "eval_config.json"
    eval_config = {}
    if config_path.exists():
        with open(config_path) as f:
            eval_config = json.load(f)

    from bm2_evaluator.reporting.text_report import generate_report

    report = generate_report(
        scored,
        {"name": "unknown", "chain": "?", "n_residues": "?"},
        eval_config,
        eval_dir / "report.txt",
        top_n=args.top,
    )
    print(report)


def _cmd_compare(args):
    """Compare two evaluation runs."""
    import csv

    from bm2_evaluator.reporting.comparison import compare_tools

    for run_path in [args.run1, args.run2]:
        csv_path = Path(run_path) / "evaluation_summary.csv"
        if not csv_path.exists():
            logger.error(f"No evaluation_summary.csv in {run_path}")
            sys.exit(1)

        with open(csv_path, newline="") as f:
            scored = list(csv.DictReader(f))

        for d in scored:
            for key in ["ensemble_ipsae_min"]:
                if d.get(key):
                    try:
                        d[key] = float(d[key])
                    except (ValueError, TypeError):
                        d[key] = 0.0

        print(f"\n=== {run_path} ===")
        print(compare_tools(scored))


def _cmd_export(args):
    """Export top designs."""
    import csv

    eval_dir = Path(args.eval_dir)
    summary_csv = eval_dir / "evaluation_summary.csv"

    if not summary_csv.exists():
        logger.error(f"No evaluation_summary.csv in {eval_dir}")
        sys.exit(1)

    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        designs = list(reader)

    top = designs[: args.top]

    out_path = args.output or str(eval_dir / f"top_{args.top}.{args.fmt}")

    if args.fmt == "fasta":
        with open(out_path, "w") as f:
            for d in top:
                did = d.get("design_id", "unknown")
                seq = d.get("binder_sequence", "")
                rank = d.get("rank", "?")
                tier = d.get("tier", "?")
                score = d.get("composite_score", "?")
                f.write(f">{did} rank={rank} tier={tier} score={score}\n")
                f.write(f"{seq}\n")
        print(f"Exported {len(top)} designs to {out_path}")

    elif args.fmt == "csv":
        with open(out_path, "w", newline="") as f:
            if top:
                writer = csv.DictWriter(f, fieldnames=top[0].keys())
                writer.writeheader()
                writer.writerows(top)
        print(f"Exported {len(top)} designs to {out_path}")

    elif args.fmt == "json":
        with open(out_path, "w") as f:
            json.dump(top, f, indent=2)
        print(f"Exported {len(top)} designs to {out_path}")


if __name__ == "__main__":
    main()
