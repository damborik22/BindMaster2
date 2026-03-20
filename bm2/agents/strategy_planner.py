"""Agent 2: Decide which tools to run, how many designs, what settings."""

from __future__ import annotations

import logging

from bm2.agents.base import Agent
from bm2.core.config import BM2Config
from bm2.core.models import Campaign, CampaignState, ToolRunConfig
from bm2.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class StrategyPlanner(Agent):
    """Plans tool allocation and settings for a campaign."""

    def __init__(self, tool_registry: ToolRegistry, config: BM2Config):
        self.registry = tool_registry
        self.config = config

    @property
    def name(self) -> str:
        return "strategy_planner"

    @property
    def required_state(self) -> CampaignState:
        return CampaignState.ANALYZING

    @property
    def target_state(self) -> CampaignState:
        return CampaignState.PLANNING

    def _execute(
        self, campaign: Campaign, total_designs: int = 500, **kwargs
    ) -> None:
        target = campaign.target
        if target is None:
            raise ValueError("No target profile. Run TargetAnalyst first.")

        tools = target.recommended_tools
        if not tools:
            raise RuntimeError(
                "No tools available. Install at least one design tool."
            )

        allocations = self._allocate_designs(tools, target, total_designs)

        for tool_name, num_designs in allocations.items():
            run_config = ToolRunConfig(
                tool_name=tool_name,
                num_designs=num_designs,
                binder_length_range=target.suggested_length_range,
                hotspot_residues=target.hotspot_residues,
                extra_settings=self._tool_specific_settings(
                    tool_name, target
                ),
            )
            campaign.tool_runs.append(run_config)
            logger.info(f"Planned: {tool_name} x {num_designs} designs")

        campaign.eval_engines = self.config.eval_engines
        campaign.eval_rosetta = self.config.eval_rosetta

    def _allocate_designs(
        self, tools: list[str], target, total: int
    ) -> dict[str, int]:
        """Allocate design count per tool.

        BindCraft: fewer (expensive, high quality)
        BoltzGen: more (fast, needs volume)
        RFAA: medium (backbone diversity)
        Others: proportional
        """
        weights = {
            "bindcraft": 0.25,
            "boltzgen": 0.30,
            "rfdiffusion": 0.20,
            "complexa": 0.15,
            "pxdesign": 0.15,
            "mosaic": 0.10,
        }

        active_weight = sum(weights.get(t, 0.1) for t in tools)
        allocations = {}
        for tool in tools:
            w = weights.get(tool, 0.1)
            allocations[tool] = max(10, int(total * w / active_weight))

        return allocations

    def _tool_specific_settings(self, tool_name: str, target) -> dict:
        """Per-tool extra settings based on target properties."""
        settings: dict = {}

        if tool_name == "boltzgen":
            settings["protocol"] = "protein-anything"

        if tool_name == "complexa":
            settings["search_strategy"] = (
                "beam_search" if target.difficulty_score > 0.6 else "best_of_n"
            )

        return settings
