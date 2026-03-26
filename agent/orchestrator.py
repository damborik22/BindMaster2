#!/usr/bin/env python3
"""BM2 Agentic Orchestrator — runs the full binder design pipeline.

Phases:
1. Target Analysis: PDB → target_analysis.json + strategy.md
2. Loss Tuning: Karpathy loop → tuned design.py
3. Production Mosaic: run tuned design.py at scale
4. Parallel Tools: BindCraft, BoltzGen, RFAA, PXDesign, Complexa (via bm2 run)
5. Final Evaluation: cross-model refolding and ranking (via bm2-eval)

Usage:
    # Full pipeline (Phases 1-3)
    python agent/orchestrator.py --target target.pdb --chain A --output ./campaign

    # With tuning
    python agent/orchestrator.py --target target.pdb --chain A --output ./campaign \
        --tune --tune-mode simple --tune-experiments 10

    # Production run (skip tuning, use provided design.py)
    python agent/orchestrator.py --target target.pdb --chain A --output ./campaign \
        --design-script my_tuned_design.py --n-designs 500

    # Phase 1 only (analysis)
    python agent/orchestrator.py --target target.pdb --chain A --output ./campaign \
        --phase analyze
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def phase1_analyze(
    target_pdb: Path,
    chain: str,
    output_dir: Path,
) -> dict:
    """Phase 1: Target Analysis.

    Runs target_analyzer.py to produce target_analysis.json
    and fills strategy.md with target-specific context.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Target Analysis")
    print("=" * 60)

    analyzer = Path(__file__).parent / "target_analyzer.py"
    strategy_template = Path(__file__).parent / "strategy.md"
    analysis_json = output_dir / "target_analysis.json"
    strategy_filled = output_dir / "strategy.md"

    # Run analyzer
    cmd = [
        sys.executable, str(analyzer),
        "--pdb", str(target_pdb),
        "--chain", chain,
        "--output", str(analysis_json),
    ]
    if strategy_template.exists():
        cmd += [
            "--strategy-template", str(strategy_template),
            "--strategy-output", str(strategy_filled),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return {}

    print(result.stdout)

    with open(analysis_json) as f:
        analysis = json.load(f)

    print(f"  Target: {analysis['target_name']} ({analysis['n_residues']} residues)")
    print(f"  Difficulty: {analysis['difficulty']}")
    print(f"  Strategy: {analysis['strategy']}")

    return analysis


def phase2_tune(
    target_pdb: Path,
    chain: str,
    output_dir: Path,
    analysis: dict,
    mode: str = "simple",
    max_experiments: int = 10,
    mosaic_venv: str = "",
    n_designs_per_experiment: int = 10,
) -> Path:
    """Phase 2: Loss Function Tuning (Karpathy Loop).

    Copies design_template.py, injects target sequence, then runs
    the loss tuner to optimize parameters.

    Returns path to the tuned design.py.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Loss Function Tuning")
    print("=" * 60)

    template = Path(__file__).parent / "design_template.py"
    design_py = output_dir / "design.py"

    # Copy template and inject target sequence
    content = template.read_text()

    target_seq = analysis.get("sequence", "")
    target_pdb_str = str(target_pdb)

    # Replace target parameters
    content = re.sub(
        r'^TARGET_SEQUENCE\s*=\s*.+$',
        f'TARGET_SEQUENCE = {target_seq!r}',
        content, flags=re.MULTILINE,
    )
    content = re.sub(
        r'^TARGET_PDB\s*=\s*.+$',
        f'TARGET_PDB = {target_pdb_str!r}',
        content, flags=re.MULTILINE,
    )
    content = re.sub(
        r'^TARGET_CHAIN_ID\s*=\s*.+$',
        f'TARGET_CHAIN_ID = {chain!r}',
        content, flags=re.MULTILINE,
    )

    # Set N_DESIGNS for fast iteration
    content = re.sub(
        r'^N_DESIGNS\s*=\s*.+$',
        f'N_DESIGNS = {n_designs_per_experiment}  # tuning batch size',
        content, flags=re.MULTILINE,
    )

    # Set binder length from analysis
    difficulty = analysis.get("difficulty", 0.3)
    if difficulty > 0.6:
        binder_len = 150
    elif difficulty > 0.4:
        binder_len = 100
    else:
        binder_len = 80

    content = re.sub(
        r'^MIN_LENGTH\s*=\s*.+$',
        f'MIN_LENGTH = {binder_len}  # set by target analyzer',
        content, flags=re.MULTILINE,
    )
    content = re.sub(
        r'^MAX_LENGTH\s*=\s*.+$',
        f'MAX_LENGTH = {binder_len}  # single length for tuning',
        content, flags=re.MULTILINE,
    )

    design_py.write_text(content)
    print(f"  Design script: {design_py}")
    print(f"  Target: {analysis.get('target_name', '?')} ({len(target_seq)} aa)")
    print(f"  Binder length: {binder_len}")
    print(f"  Designs per experiment: {n_designs_per_experiment}")

    # Run loss tuner
    tuner = Path(__file__).parent / "loss_tuner.py"
    if not mosaic_venv:
        mosaic_venv = str(Path.home() / "BindMaster" / "Mosaic" / ".venv")

    cmd = [
        sys.executable, str(tuner),
        "--design", str(design_py),
        "--output", str(output_dir / "tuning"),
        "--mosaic-venv", mosaic_venv,
        "--mode", mode,
        "--max-experiments", str(max_experiments),
    ]

    print(f"  Tuning mode: {mode}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Running tuner...")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  WARNING: Tuner exited with non-zero code")

    return design_py


def phase3_production(
    design_py: Path,
    output_dir: Path,
    mosaic_venv: str = "",
    n_designs: int = 500,
    top_k: int = 50,
) -> Path:
    """Phase 3: Production Mosaic Run.

    Runs the tuned design.py at scale to generate production designs.

    Returns path to the production output directory.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Production Mosaic Run")
    print("=" * 60)

    production_dir = output_dir / "production_mosaic"
    production_dir.mkdir(parents=True, exist_ok=True)

    # Copy and configure design.py for production scale
    prod_design = production_dir / "design.py"
    content = design_py.read_text()

    content = re.sub(
        r'^N_DESIGNS\s*=\s*.+$',
        f'N_DESIGNS = {n_designs}  # production scale',
        content, flags=re.MULTILINE,
    )
    content = re.sub(
        r'^TOP_K\s*=\s*.+$',
        f'TOP_K = {top_k}  # production top-K',
        content, flags=re.MULTILINE,
    )

    prod_design.write_text(content)

    if not mosaic_venv:
        mosaic_venv = str(Path.home() / "BindMaster" / "Mosaic" / ".venv")

    python = f"{mosaic_venv}/bin/python"
    cmd = f"{python} {prod_design}"

    print(f"  Designs: {n_designs}")
    print(f"  Top-K: {top_k}")
    print(f"  Output: {production_dir}")
    print(f"  Running...")

    start = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=str(production_dir),
    )
    elapsed = time.time() - start

    print(f"  Completed in {elapsed / 60:.0f} min (exit code {result.returncode})")

    return production_dir


def run_pipeline(
    target_pdb: Path,
    chain: str,
    output_dir: Path,
    tune: bool = False,
    tune_mode: str = "simple",
    tune_experiments: int = 10,
    tune_designs: int = 10,
    design_script: Path | None = None,
    production_designs: int = 500,
    production_top_k: int = 50,
    mosaic_venv: str = "",
    phase: str = "all",
) -> dict:
    """Run the full pipeline or a specific phase.

    Args:
        target_pdb: Path to target PDB file.
        chain: Target chain ID.
        output_dir: Campaign output directory.
        tune: Whether to run Phase 2 (loss tuning).
        tune_mode: Tuning mode (simple/manual/auto).
        tune_experiments: Max tuning experiments.
        tune_designs: Designs per tuning experiment.
        design_script: Pre-tuned design.py (skips Phase 2).
        production_designs: Number of designs for Phase 3.
        production_top_k: Top-K for Phase 3 refolding.
        mosaic_venv: Path to Mosaic venv.
        phase: Which phase to run (analyze/tune/produce/all).

    Returns:
        Summary dict with paths to all outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_pdb": str(target_pdb),
        "chain": chain,
        "output_dir": str(output_dir),
        "started": datetime.now().isoformat(),
    }

    # Phase 1: Target Analysis (always runs)
    analysis = phase1_analyze(target_pdb, chain, output_dir)
    summary["analysis"] = analysis

    if phase == "analyze":
        summary["completed"] = datetime.now().isoformat()
        _save_summary(output_dir, summary)
        return summary

    # Phase 2: Loss Tuning (optional)
    if design_script:
        design_py = Path(design_script)
        print(f"\nUsing pre-tuned design script: {design_py}")
    elif tune or phase == "tune":
        design_py = phase2_tune(
            target_pdb=target_pdb,
            chain=chain,
            output_dir=output_dir,
            analysis=analysis,
            mode=tune_mode,
            max_experiments=tune_experiments,
            mosaic_venv=mosaic_venv,
            n_designs_per_experiment=tune_designs,
        )
    else:
        # No tuning — use template with defaults
        template = Path(__file__).parent / "design_template.py"
        design_py = output_dir / "design.py"
        content = template.read_text()
        target_seq = analysis.get("sequence", "")
        content = re.sub(
            r'^TARGET_SEQUENCE\s*=\s*.+$',
            f'TARGET_SEQUENCE = {target_seq!r}',
            content, flags=re.MULTILINE,
        )
        content = re.sub(
            r'^TARGET_PDB\s*=\s*.+$',
            f'TARGET_PDB = {str(target_pdb)!r}',
            content, flags=re.MULTILINE,
        )
        design_py.write_text(content)

    summary["design_script"] = str(design_py)

    if phase == "tune":
        summary["completed"] = datetime.now().isoformat()
        _save_summary(output_dir, summary)
        return summary

    # Phase 3: Production Mosaic Run
    production_dir = phase3_production(
        design_py=design_py,
        output_dir=output_dir,
        mosaic_venv=mosaic_venv,
        n_designs=production_designs,
        top_k=production_top_k,
    )
    summary["production_dir"] = str(production_dir)

    summary["completed"] = datetime.now().isoformat()
    _save_summary(output_dir, summary)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Analysis: target_analysis.json")
    print(f"  Strategy: strategy.md")
    if tune:
        print(f"  Tuning log: tuning/experiment_log.json")
    print(f"  Production: {production_dir}")
    print(f"\nTo run Phase 4-5 (other tools + final evaluation):")
    print(f"  bm2 create <name> {target_pdb} {chain}")
    print(f"  bm2 run <campaign_id> --through ranked")

    return summary


def _save_summary(output_dir: Path, summary: dict) -> None:
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="BM2 Agentic Orchestrator — protein binder design pipeline"
    )
    parser.add_argument("--target", required=True, help="Target PDB file")
    parser.add_argument("--chain", default="A", help="Target chain (default: A)")
    parser.add_argument("--output", required=True, help="Output directory")

    # Tuning options
    parser.add_argument("--tune", action="store_true", help="Run Phase 2 loss tuning")
    parser.add_argument("--tune-mode", choices=["simple", "manual", "auto"],
                        default="simple", help="Tuning mode")
    parser.add_argument("--tune-experiments", type=int, default=10,
                        help="Max tuning experiments")
    parser.add_argument("--tune-designs", type=int, default=10,
                        help="Designs per tuning experiment")

    # Production options
    parser.add_argument("--design-script", default=None,
                        help="Pre-tuned design.py (skip tuning)")
    parser.add_argument("--n-designs", type=int, default=500,
                        help="Production Mosaic designs")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K for production refolding")

    # Environment
    parser.add_argument("--mosaic-venv",
                        default=str(Path.home() / "BindMaster" / "Mosaic" / ".venv"),
                        help="Path to Mosaic UV venv")

    # Phase selection
    parser.add_argument("--phase", choices=["analyze", "tune", "produce", "all"],
                        default="all", help="Run specific phase only")

    args = parser.parse_args()

    run_pipeline(
        target_pdb=Path(args.target),
        chain=args.chain,
        output_dir=Path(args.output),
        tune=args.tune,
        tune_mode=args.tune_mode,
        tune_experiments=args.tune_experiments,
        tune_designs=args.tune_designs,
        design_script=Path(args.design_script) if args.design_script else None,
        production_designs=args.n_designs,
        production_top_k=args.top_k,
        mosaic_venv=args.mosaic_venv,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
