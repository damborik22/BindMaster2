"""Orchestrates all agents for a complete campaign run."""

from __future__ import annotations

import logging
from pathlib import Path

from bm2.agents.target_analyst import TargetAnalyst
from bm2.agents.strategy_planner import StrategyPlanner
from bm2.agents.design_runner import DesignRunner
from bm2.agents.evaluator_agent import EvaluatorAgent
from bm2.agents.wetlab_advisor import WetLabAdvisor
from bm2.agents.maturation_agent import MaturationAgent
from bm2.core.campaign import CampaignManager
from bm2.core.config import BM2Config
from bm2.core.models import Campaign, CampaignState
from bm2.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CampaignOrchestrator:
    """Runs agents in sequence through the campaign lifecycle.

    Can run the full pipeline or stop at any stage.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        campaign_manager: CampaignManager,
        config: BM2Config,
    ):
        self.registry = tool_registry
        self.campaign_mgr = campaign_manager
        self.config = config

        self.agents = {
            "analyze": TargetAnalyst(tool_registry),
            "plan": StrategyPlanner(tool_registry, config),
            "design": DesignRunner(tool_registry, campaign_manager),
            "evaluate": EvaluatorAgent(tool_registry),
            "wetlab": WetLabAdvisor(),
            "mature": MaturationAgent(tool_registry),
        }

    def run_through(
        self,
        campaign: Campaign,
        stop_at: CampaignState = CampaignState.RANKED,
        save_path: Path | None = None,
        **kwargs,
    ) -> Campaign:
        """Run agents in sequence until reaching stop_at state.

        Default: analyze -> plan -> design -> evaluate -> ranked
        """
        pipeline = [
            ("analyze", CampaignState.ANALYZING),
            ("plan", CampaignState.PLANNING),
            ("design", CampaignState.DESIGNING),
            ("evaluate", CampaignState.EVALUATING),
        ]

        if stop_at in (CampaignState.WET_LAB_PREP, CampaignState.TESTING):
            pipeline.append(("wetlab", CampaignState.WET_LAB_PREP))

        for agent_name, _ in pipeline:
            if campaign.state == stop_at:
                break

            agent = self.agents.get(agent_name)
            if agent is None:
                continue

            try:
                campaign = agent.run(
                    campaign, save_path=save_path, **kwargs
                )
            except Exception as e:
                logger.error(f"Pipeline stopped at {agent_name}: {e}")
                break

        return campaign

    def run_agent(
        self,
        campaign: Campaign,
        agent_name: str,
        save_path: Path | None = None,
        **kwargs,
    ) -> Campaign:
        """Run a single named agent."""
        if agent_name not in self.agents:
            raise KeyError(
                f"Unknown agent: {agent_name}. "
                f"Available: {list(self.agents.keys())}"
            )

        agent = self.agents[agent_name]
        return agent.run(campaign, save_path=save_path, **kwargs)
