"""Agent 4: Invoke the standalone bm2-evaluator on collected designs."""

from __future__ import annotations

import csv
import logging
import subprocess
from pathlib import Path

from bm2.agents.base import Agent
from bm2.core.models import Campaign, CampaignState
from bm2.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class EvaluatorAgent(Agent):
    """Runs bm2-evaluator on all collected designs."""

    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    @property
    def name(self) -> str:
        return "evaluator_agent"

    @property
    def required_state(self) -> CampaignState:
        return CampaignState.DESIGNING

    @property
    def target_state(self) -> CampaignState:
        return CampaignState.EVALUATING

    def _compatible_states(self) -> list[CampaignState]:
        return [CampaignState.DESIGNING, CampaignState.RANKED]

    def _execute(self, campaign: Campaign, **kwargs) -> None:
        if campaign.target is None:
            raise ValueError("No target in campaign")

        # Collect output dirs from completed tool runs
        design_args = []
        for run in campaign.tool_runs:
            if run.status != "completed" or not run.output_dir:
                continue
            if not self.registry.is_registered(run.tool_name):
                continue
            tool = self.registry.get(run.tool_name)
            design_args.append(f"--designs {run.output_dir}")
            design_args.append(f"--parser {tool.parser_name()}")

        if not design_args:
            raise RuntimeError("No completed tool runs to evaluate")

        eval_dir = kwargs.get("eval_dir")
        if eval_dir is None:
            # Derive from save_path or use default
            eval_dir = Path.home() / ".bm2" / "campaigns" / campaign.id / "evaluation"

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        pae_cutoff = campaign.eval_config.get("pae_cutoff", 15.0)
        ipsae_threshold = campaign.eval_config.get("ipsae_threshold", 0.61)

        cmd_parts = [
            "bm2-eval",
            "score",
            f"--target {campaign.target.pdb_path}",
            f"--chain {campaign.target.chains[0]}",
            f"--output {eval_dir}",
            f"--engines {','.join(campaign.eval_engines)}",
            f"--pae-cutoff {pae_cutoff}",
            f"--ipsae-threshold {ipsae_threshold}",
        ]
        cmd_parts.extend(design_args)

        if campaign.eval_rosetta:
            cmd_parts.append("--rosetta")

        cmd = " ".join(cmd_parts)
        logger.info(f"Running evaluator: bm2-eval score ...")

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Evaluator failed (exit {result.returncode}): "
                f"{result.stderr[-500:]}"
            )

        campaign.evaluation_dir = eval_dir

        report_path = eval_dir / "report.txt"
        if report_path.exists():
            logger.info("\n" + report_path.read_text())

        # Generate HTML reports: BM1 full report (optional) + standalone fallback
        self._generate_reports(eval_dir, campaign)

    def _generate_reports(self, eval_dir: Path, campaign: Campaign) -> None:
        """Generate HTML reports: try BM1 first, always produce standalone."""
        # Try BM1 report (rich plots + PyMOL scripts)
        try:
            from bm2_evaluator.reporting.bm1_report import generate_bm1_report
            report_dir = eval_dir / "report"
            if generate_bm1_report(evaluation_dir=eval_dir, output_dir=report_dir):
                logger.info("BM1 HTML report generated")
            else:
                raise RuntimeError("BM1 report returned False")
        except Exception as e:
            logger.info(f"BM1 report unavailable ({e}), skipping")

        # Standalone HTML report (always attempted as guaranteed fallback)
        try:
            from bm2_evaluator.reporting.html_report import generate_html_report

            scored_designs = self._load_summary_csv(eval_dir)
            if not scored_designs:
                logger.warning("No designs in summary CSV, skipping HTML report")
                return

            target_info = {
                "name": campaign.target.pdb_path.stem if campaign.target else "unknown",
                "chain": campaign.target.chains[0] if campaign.target and campaign.target.chains else "?",
                "n_residues": getattr(campaign.target, "target_length", "?"),
            }
            eval_config = {
                "engines": campaign.eval_engines,
                "ipsae_consensus_threshold": campaign.eval_config.get(
                    "ipsae_threshold", 0.61
                ),
                "pae_cutoff": campaign.eval_config.get("pae_cutoff", 15.0),
                "use_rosetta": campaign.eval_rosetta,
            }

            html_path = eval_dir / "report.html"
            generate_html_report(
                scored_designs=scored_designs,
                target_info=target_info,
                eval_config=eval_config,
                output_path=html_path,
            )
            logger.info(f"Standalone HTML report: {html_path}")
        except Exception as e2:
            logger.warning(f"Standalone HTML report failed: {e2}")

    @staticmethod
    def _load_summary_csv(eval_dir: Path) -> list[dict]:
        """Load evaluation_summary.csv into a list of dicts."""
        summary_csv = eval_dir / "evaluation_summary.csv"
        if not summary_csv.exists():
            return []

        designs: list[dict] = []
        with open(summary_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                d: dict = {}
                for key, val in row.items():
                    if val == "":
                        d[key] = None
                    else:
                        try:
                            d[key] = float(val)
                        except (ValueError, TypeError):
                            d[key] = val
                # Ensure rank is int if present
                if d.get("rank") is not None:
                    try:
                        d["rank"] = int(float(d["rank"]))
                    except (ValueError, TypeError):
                        pass
                designs.append(d)
        return designs
