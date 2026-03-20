"""Agent 3: Execute design tools, monitor progress, collect outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bm2.agents.base import Agent
from bm2.core.campaign import CampaignManager
from bm2.core.models import Campaign, CampaignState
from bm2.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class DesignRunner(Agent):
    """Launches tools in their conda envs. Handles failures gracefully."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        campaign_manager: CampaignManager,
    ):
        self.registry = tool_registry
        self.campaign_mgr = campaign_manager

    @property
    def name(self) -> str:
        return "design_runner"

    @property
    def required_state(self) -> CampaignState:
        return CampaignState.PLANNING

    @property
    def target_state(self) -> CampaignState:
        return CampaignState.DESIGNING

    def _execute(
        self,
        campaign: Campaign,
        save_path: Optional[Path] = None,
        **kwargs,
    ) -> None:
        runs_dir = self.campaign_mgr.campaign_dir(campaign.id) / "runs"
        completed_count = 0

        for run_config in campaign.tool_runs:
            if run_config.status == "completed":
                completed_count += 1
                continue  # Resume support

            if not self.registry.is_registered(run_config.tool_name):
                run_config.status = "failed"
                run_config.error = f"Tool not installed: {run_config.tool_name}"
                logger.warning(run_config.error)
                continue

            tool = self.registry.get(run_config.tool_name)
            run_dir = runs_dir / run_config.tool_name
            run_config.output_dir = tool.output_dir(run_dir)

            logger.info(
                f"Launching {run_config.tool_name}: "
                f"{run_config.num_designs} designs"
            )

            try:
                run_config.status = "running"
                if save_path:
                    campaign.save(save_path)

                prepared = tool.prepare_config(campaign, run_config, run_dir)
                process = tool.launch(prepared, run_dir)
                returncode = process.wait()

                if returncode != 0:
                    run_config.status = "failed"
                    run_config.error = f"Exit code {returncode}"
                    logger.error(
                        f"{run_config.tool_name} failed (exit {returncode})"
                    )
                    continue

                if tool.is_complete(run_dir):
                    run_config.status = "completed"
                    completed_count += 1
                    logger.info(f"{run_config.tool_name} complete")
                else:
                    run_config.status = "failed"
                    run_config.error = "Exited 0 but output not found"
                    logger.error(f"{run_config.tool_name}: no output found")

            except Exception as e:
                run_config.status = "failed"
                run_config.error = str(e)
                logger.error(f"{run_config.tool_name} error: {e}")
                continue
            finally:
                if save_path:
                    campaign.save(save_path)

        if completed_count == 0:
            raise RuntimeError(
                "All tool runs failed. Check logs in campaign/runs/."
            )

        logger.info(
            f"{completed_count}/{len(campaign.tool_runs)} tools completed"
        )
