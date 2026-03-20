"""Agent 4: Invoke the standalone bm2-evaluator on collected designs."""

from __future__ import annotations

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
