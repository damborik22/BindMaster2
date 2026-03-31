#!/usr/bin/env python3
"""Karpathy-style loss function tuning loop for Mosaic binder design.

Pattern: edit design.py constants -> run Mosaic -> score with inner_evaluator ->
commit (if improved) or revert (if not) -> loop.

Modes:
    simple  -- predefined grid search (validates loop machinery)
    manual  -- human decides what to try (prints state, reads input)
    auto    -- LLM decides (future, stub for now)

Usage:
    python agent/loss_tuner.py --design agent/design_template.py \\
        --mode simple --max-experiments 10 \\
        --mosaic-venv /home/david/BindMaster/Mosaic/.venv
"""

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================
# GRID SEARCH EXPERIMENTS
# ============================
# Each entry: (hypothesis_text, {CONSTANT: new_value, ...})
# The first entry is the baseline -- runs with defaults unchanged.

GRID_SEARCH_EXPERIMENTS = [
    ("Baseline: Escalante Nipah defaults", {}),
    ("Lower WEIGHT_SEQ_RECOVERY from 10.0 to 5.0 — less MPNN constraint",
     {"WEIGHT_SEQ_RECOVERY": 5.0}),
    ("Raise WEIGHT_SEQ_RECOVERY from 10.0 to 15.0 — stronger sequence naturalness",
     {"WEIGHT_SEQ_RECOVERY": 15.0}),
    ("Larger binder: 120 aa (from 80) — more surface area for binding",
     {"MIN_LENGTH": 120, "MAX_LENGTH": 120}),
    ("Much larger binder: 200 aa — Escalante Nipah was 220",
     {"MIN_LENGTH": 200, "MAX_LENGTH": 200}),
    ("Stronger binder internal confidence: WEIGHT_WITHIN_BINDER_PAE 0.4 → 0.8",
     {"WEIGHT_WITHIN_BINDER_PAE": 0.8}),
    ("Sharper optimizer: OPT_STAGE1_SCALE 1.0 → 2.0",
     {"OPT_STAGE1_SCALE": 2.0}),
    ("Stronger contact drive: WEIGHT_BINDER_TARGET_CONTACT 1.0 → 2.0",
     {"WEIGHT_BINDER_TARGET_CONTACT": 2.0}),
    ("Higher pLDDT weight: 0.1 → 0.3",
     {"WEIGHT_PLDDT": 0.3}),
    ("Looser MPNN temperature: 0.001 → 0.01 — more sequence diversity",
     {"MPNN_TEMPERATURE": 0.01}),
]

PROTEINA_GRID_SEARCH = [
    ("Baseline: ipTM + ipSAE beam scoring", {}),
    ("Wider beam: BEAM_WIDTH 4 → 8 — more candidates explored",
     {"BEAM_WIDTH": 8}),
    ("More branching: N_BRANCH 4 → 8 — wider tree search",
     {"N_BRANCH": 8}),
    ("Larger binder: 120 aa",
     {"MIN_LENGTH": 120, "MAX_LENGTH": 120}),
    ("Much larger binder: 200 aa",
     {"MIN_LENGTH": 200, "MAX_LENGTH": 200}),
    ("Add contact loss to beam scoring",
     {"WEIGHT_BINDER_TARGET_CONTACT": 1.0}),
    ("Add pLDDT to beam scoring",
     {"WEIGHT_PLDDT": 0.3}),
    ("Add binder PAE to beam scoring",
     {"WEIGHT_WITHIN_BINDER_PAE": 0.4}),
    ("More inverse folding: 10 per backbone",
     {"INVERSE_FOLD_SAMPLES": 10}),
    ("Fewer checkpoints (faster): [0, 200, 400]",
     {"STEP_CHECKPOINTS": [0, 200, 400]}),
]


# ============================
# DESIGN.PY EDITING
# ============================

def read_constants(design_path: Path) -> dict:
    """Read current constant values from the design script."""
    constants = {}
    content = design_path.read_text()
    # Match lines like: CONSTANT_NAME = value  # comment
    pattern = re.compile(
        r'^([A-Z][A-Z0-9_]+)\s*=\s*(.+?)(?:\s*#.*)?$',
        re.MULTILINE,
    )
    for match in pattern.finditer(content):
        name = match.group(1)
        raw_value = match.group(2).strip()
        try:
            value = eval(raw_value)  # handles int, float, bool, None, str, list
        except Exception:
            value = raw_value
        constants[name] = value
    return constants


def write_constant(design_path: Path, name: str, value) -> None:
    """Replace a single constant's value in the design script."""
    content = design_path.read_text()
    # Match: NAME = <old_value>  # optional comment
    # Preserve the comment if present
    pattern = re.compile(
        r'^(' + re.escape(name) + r'\s*=\s*)(.+?)(\s*#.*)?$',
        re.MULTILINE,
    )

    def replacer(m):
        prefix = m.group(1)
        comment = m.group(3) or ""
        return f"{prefix}{repr(value)}{comment}"

    new_content, n = pattern.subn(replacer, content)
    if n == 0:
        raise ValueError(f"Constant {name} not found in {design_path}")
    design_path.write_text(new_content)


def apply_changes(design_path: Path, changes: dict) -> None:
    """Apply a set of constant changes to the design script."""
    for name, value in changes.items():
        write_constant(design_path, name, value)


def reset_to_defaults(design_path: Path, defaults: dict, changes: dict) -> None:
    """Reset changed constants back to their default values."""
    for name in changes:
        if name in defaults:
            write_constant(design_path, name, defaults[name])


# ============================
# MOSAIC RUNNER
# ============================

def run_mosaic(
    design_path: Path,
    output_dir: Path,
    mosaic_venv: str,
    timeout: int = 3600,
) -> bool:
    """Run the design script in the Mosaic venv.

    Returns True if the run completed successfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    python = f"{mosaic_venv}/bin/python"
    cmd = f"{python} {design_path}"

    log_path = output_dir / "run.log"

    print(f"  Running Mosaic: {cmd}")
    print(f"  Output dir: {output_dir}")
    print(f"  Log: {log_path}")

    start = time.time()
    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(output_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.0f}s (exit code {result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ============================
# INNER EVALUATOR
# ============================

def run_inner_evaluator(
    designs_csv: Path,
    python: str = "python",
) -> dict | None:
    """Run inner_evaluator.py on a designs.csv and return the result."""
    evaluator_path = Path(__file__).parent / "inner_evaluator.py"

    if not designs_csv.exists():
        print(f"  ERROR: designs.csv not found at {designs_csv}")
        return None

    cmd = [python, str(evaluator_path), "--input", str(designs_csv)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  Evaluator error: {result.stderr}")
            return None
        return json.loads(result.stdout)
    except Exception as e:
        print(f"  Evaluator exception: {e}")
        return None


# ============================
# GIT OPERATIONS
# ============================

def git_commit(design_path: Path, message: str) -> bool:
    """Stage and commit the design script."""
    try:
        subprocess.run(["git", "add", str(design_path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Git commit failed: {e.stderr.decode() if e.stderr else e}")
        return False


def git_revert(design_path: Path) -> bool:
    """Revert the design script to the last committed version."""
    try:
        subprocess.run(
            ["git", "checkout", "--", str(design_path)],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Git revert failed: {e.stderr.decode() if e.stderr else e}")
        return False


# ============================
# EXPERIMENT LOG
# ============================

def load_experiment_log(log_path: Path) -> list[dict]:
    """Load experiment log from JSON file."""
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return []


def save_experiment_log(log_path: Path, experiments: list[dict]) -> None:
    """Save experiment log to JSON file."""
    with open(log_path, "w") as f:
        json.dump(experiments, f, indent=2)


# ============================
# DECISION MODES
# ============================

def decide_simple(
    experiment_id: int,
    defaults: dict,
    experiments: list[dict],
    engine: str = "boltz2",
) -> tuple[str, dict] | None:
    """Simple grid search -- return next experiment from predefined list."""
    grid = PROTEINA_GRID_SEARCH if engine == "proteina" else GRID_SEARCH_EXPERIMENTS
    if experiment_id >= len(grid):
        return None
    hypothesis, changes = grid[experiment_id]
    return hypothesis, changes


def decide_manual(
    experiment_id: int,
    defaults: dict,
    experiments: list[dict],
    engine: str = "boltz2",
) -> tuple[str, dict] | None:
    """Manual mode -- print state, ask human."""
    print("\n" + "=" * 60)
    print("MANUAL MODE — Current state:")
    print(f"  Experiment: {experiment_id}")
    if experiments:
        best = max(experiments, key=lambda e: e.get("inner_score", 0))
        print(f"  Best score: {best['inner_score']:.4f} (experiment {best['experiment_id']})")
        print(f"  Last: {experiments[-1].get('hypothesis', '?')} → {experiments[-1].get('inner_score', 0):.4f}")

    print("\nEnter changes as: CONSTANT=value,CONSTANT2=value2")
    print("Enter hypothesis text first, then changes on next line.")
    print("Type 'stop' to end.\n")

    hypothesis = input("Hypothesis: ").strip()
    if hypothesis.lower() == "stop":
        return None

    changes_str = input("Changes: ").strip()
    if not changes_str:
        return hypothesis, {}

    changes = {}
    for part in changes_str.split(","):
        key, val = part.strip().split("=", 1)
        try:
            changes[key.strip()] = eval(val.strip())
        except Exception:
            changes[key.strip()] = val.strip()

    return hypothesis, changes


def decide_auto(
    experiment_id: int,
    defaults: dict,
    experiments: list[dict],
    engine: str = "boltz2",
) -> tuple[str, dict] | None:
    """Auto mode -- LLM decides. Stub for now."""
    print("  AUTO mode not yet implemented. Use 'simple' or 'manual'.")
    return None


DECISION_MODES = {
    "simple": decide_simple,
    "manual": decide_manual,
    "auto": decide_auto,
}


# ============================
# MAIN LOOP
# ============================

def run_tuning_loop(
    design_path: Path,
    output_base: Path,
    mosaic_venv: str,
    mode: str = "simple",
    max_experiments: int = 10,
    log_path: Path | None = None,
    timeout: int = 3600,
    engine: str = "boltz2",
) -> dict:
    """Run the Karpathy loss-tuning loop.

    Returns the best experiment result.
    """
    design_path = Path(design_path).resolve()
    output_base = Path(output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = output_base / "experiment_log.json"

    decide_fn = DECISION_MODES.get(mode)
    if decide_fn is None:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {list(DECISION_MODES.keys())}")

    # Read baseline constants
    defaults = read_constants(design_path)
    experiments = load_experiment_log(log_path)

    best_score = -float("inf")
    best_experiment = None
    consecutive_failures = 0

    # Resume from existing log
    start_id = len(experiments)
    if experiments:
        best_exp = max(experiments, key=lambda e: e.get("inner_score", 0))
        best_score = best_exp.get("inner_score", -float("inf"))
        best_experiment = best_exp
        print(f"Resuming from experiment {start_id}. Best score so far: {best_score:.4f}")

    python = f"{mosaic_venv}/bin/python"

    for i in range(start_id, max_experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i}")
        print(f"{'='*60}")

        # 1. Decide what to change
        decision = decide_fn(i, defaults, experiments, engine=engine)
        if decision is None:
            print("No more experiments to run. Stopping.")
            break

        hypothesis, changes = decision
        print(f"Hypothesis: {hypothesis}")
        if changes:
            print(f"Changes: {changes}")
        else:
            print("Changes: none (baseline)")

        # 2. Apply changes to design.py
        if changes:
            apply_changes(design_path, changes)
            print(f"Applied {len(changes)} changes to {design_path.name}")

        # 3. Run Mosaic
        exp_output = output_base / f"experiment_{i:03d}"
        success = run_mosaic(
            design_path=design_path,
            output_dir=exp_output,
            mosaic_venv=mosaic_venv,
            timeout=timeout,
        )

        if not success:
            print("  Mosaic run FAILED")
            # Revert changes
            if changes:
                reset_to_defaults(design_path, defaults, changes)

            experiment = {
                "experiment_id": i,
                "hypothesis": hypothesis,
                "parameters_changed": changes,
                "inner_score": 0.0,
                "breakdown": {},
                "improved": False,
                "failed": True,
                "timestamp": datetime.now().isoformat(),
            }
            experiments.append(experiment)
            save_experiment_log(log_path, experiments)
            consecutive_failures += 1
            continue

        # 4. Find designs.csv (could be in output dir or cwd of the run)
        designs_csv = exp_output / "designs.csv"
        if not designs_csv.exists():
            # Mosaic writes to cwd -- check if design.py wrote there
            print(f"  designs.csv not at {designs_csv}")
            print("  Mosaic run completed but no designs.csv found")
            if changes:
                reset_to_defaults(design_path, defaults, changes)
            experiment = {
                "experiment_id": i,
                "hypothesis": hypothesis,
                "parameters_changed": changes,
                "inner_score": 0.0,
                "breakdown": {},
                "improved": False,
                "failed": True,
                "error": "designs.csv not found",
                "timestamp": datetime.now().isoformat(),
            }
            experiments.append(experiment)
            save_experiment_log(log_path, experiments)
            consecutive_failures += 1
            continue

        # 5. Score
        score_result = run_inner_evaluator(designs_csv, python=python)
        if score_result is None:
            print("  Scoring FAILED")
            if changes:
                reset_to_defaults(design_path, defaults, changes)
            experiment = {
                "experiment_id": i,
                "hypothesis": hypothesis,
                "parameters_changed": changes,
                "inner_score": 0.0,
                "breakdown": {},
                "improved": False,
                "failed": True,
                "error": "inner_evaluator failed",
                "timestamp": datetime.now().isoformat(),
            }
            experiments.append(experiment)
            save_experiment_log(log_path, experiments)
            consecutive_failures += 1
            continue

        inner_score = score_result["inner_score"]
        print(f"\n  Score: {inner_score:.4f}")
        print(f"    hit_rate={score_result['hit_rate']:.2f}  "
              f"mean_ipsae={score_result['mean_ipsae_min']:.4f}  "
              f"mean_plddt={score_result['mean_plddt_binder']:.4f}  "
              f"mean_iptm={score_result['mean_iptm']:.4f}  "
              f"diversity={score_result['sequence_diversity']:.2f}")

        # 6. Commit or revert
        improved = inner_score > best_score

        if improved:
            print(f"  IMPROVED: {best_score:.4f} → {inner_score:.4f}")
            commit_msg = (
                f"Experiment {i}: {hypothesis} → score {inner_score:.4f} "
                f"(improved from {best_score:.4f})"
            )
            git_commit(design_path, commit_msg)
            best_score = inner_score
            consecutive_failures = 0
            # Update defaults to current state (for future reverts)
            defaults = read_constants(design_path)
        else:
            print(f"  NOT improved: {inner_score:.4f} <= {best_score:.4f}")
            if changes:
                reset_to_defaults(design_path, defaults, changes)
            consecutive_failures += 1

        # 7. Log
        experiment = {
            "experiment_id": i,
            "hypothesis": hypothesis,
            "parameters_changed": changes,
            "inner_score": inner_score,
            "breakdown": score_result,
            "improved": improved,
            "failed": False,
            "timestamp": datetime.now().isoformat(),
        }
        experiments.append(experiment)
        save_experiment_log(log_path, experiments)

        if improved:
            best_experiment = experiment

        # 8. Check stopping conditions
        if consecutive_failures >= 5:
            print(f"\n  {consecutive_failures} consecutive failures. Stopping.")
            break

    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    if best_experiment:
        print(f"Best score: {best_score:.4f}")
        print(f"Best experiment: {best_experiment['experiment_id']}")
        print(f"Hypothesis: {best_experiment['hypothesis']}")
    else:
        print("No successful experiments.")

    return best_experiment or {}


def main():
    parser = argparse.ArgumentParser(
        description="BM2 Loss Function Tuner (Karpathy Loop)"
    )
    parser.add_argument(
        "--design", required=True,
        help="Path to design.py (the file to edit)",
    )
    parser.add_argument(
        "--output", default="agent/tuning_output",
        help="Base directory for experiment outputs",
    )
    parser.add_argument(
        "--mosaic-venv", default="/home/david/BindMaster/Mosaic/.venv",
        help="Path to Mosaic UV venv",
    )
    parser.add_argument(
        "--mode", choices=["simple", "manual", "auto"], default="simple",
        help="Decision mode (default: simple grid search)",
    )
    parser.add_argument(
        "--max-experiments", type=int, default=10,
        help="Maximum number of experiments to run",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Timeout per Mosaic run in seconds",
    )
    parser.add_argument(
        "--engine", choices=["boltz2", "proteina"], default="boltz2",
        help="Design engine (default: boltz2)",
    )
    args = parser.parse_args()

    result = run_tuning_loop(
        design_path=Path(args.design),
        output_base=Path(args.output),
        mosaic_venv=args.mosaic_venv,
        mode=args.mode,
        max_experiments=args.max_experiments,
        timeout=args.timeout,
        engine=args.engine,
    )

    return result


if __name__ == "__main__":
    main()
