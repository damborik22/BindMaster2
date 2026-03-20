"""Agent 6: Computational improvement of validated binders."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from bm2.agents.base import Agent
from bm2.core.models import Campaign, CampaignState, ToolRunConfig
from bm2.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MaturationAgent(Agent):
    """Plans computational maturation of promising designs.

    Strategies:
        mpnn_redesign: Keep backbone, redesign sequence with ProteinMPNN
        partial_diffusion: RFdiffusion partial noise on backbone
        warm_start_hallucination: Re-run BindCraft from hit sequence
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    @property
    def name(self) -> str:
        return "maturation_agent"

    @property
    def required_state(self) -> CampaignState:
        return CampaignState.RANKED

    @property
    def target_state(self) -> CampaignState:
        return CampaignState.MATURING

    def _compatible_states(self) -> list[CampaignState]:
        return [CampaignState.RANKED, CampaignState.TESTING]

    def _execute(
        self,
        campaign: Campaign,
        parent_design_ids: list[str] | None = None,
        strategy: str = "auto",
        **kwargs,
    ) -> None:
        # Auto-select parent designs from top hits
        if not parent_design_ids:
            parent_design_ids = self._auto_select_parents(campaign)

        if not parent_design_ids:
            logger.warning(
                "No parent designs for maturation. "
                "Need top-ranked or experimental hits."
            )
            return

        if strategy == "auto":
            strategy = self._auto_select_strategy()

        round_num = len(campaign.maturation_rounds) + 1

        maturation_round = {
            "round": round_num,
            "strategy": strategy,
            "parent_ids": parent_design_ids,
            "status": "planned",
        }

        logger.info(
            f"Maturation round {round_num}: {strategy} "
            f"on {len(parent_design_ids)} parents"
        )

        new_runs = self._create_maturation_runs(
            parent_design_ids, strategy
        )
        campaign.tool_runs.extend(new_runs)
        campaign.maturation_rounds.append(maturation_round)

    def _auto_select_parents(self, campaign) -> list[str]:
        """Select top consensus/strong hits as maturation parents."""
        eval_dir = campaign.evaluation_dir
        if not eval_dir:
            return []

        summary = Path(eval_dir) / "evaluation_summary.csv"
        if not summary.exists():
            return []

        with open(summary, newline="") as f:
            designs = list(csv.DictReader(f))

        parents = [
            d["design_id"]
            for d in designs
            if d.get("tier") in ("consensus_hit", "strong")
        ][:5]
        return parents

    def _auto_select_strategy(self) -> str:
        """Pick strategy based on installed tools."""
        installed = self.registry.list_installed()
        if "rfdiffusion" in installed:
            return "partial_diffusion"
        elif "bindcraft" in installed:
            return "warm_start_hallucination"
        else:
            return "mpnn_redesign"

    def _create_maturation_runs(
        self, parent_ids: list[str], strategy: str
    ) -> list[ToolRunConfig]:
        """Create tool run configs for maturation."""
        runs = []

        if strategy == "partial_diffusion":
            for pid in parent_ids:
                runs.append(
                    ToolRunConfig(
                        tool_name="rfdiffusion",
                        num_designs=50,
                        extra_settings={
                            "parent_id": pid,
                            "strategy": "partial_diffusion",
                            "noise_scale": 0.5,
                        },
                    )
                )
        elif strategy == "warm_start_hallucination":
            for pid in parent_ids:
                runs.append(
                    ToolRunConfig(
                        tool_name="bindcraft",
                        num_designs=20,
                        extra_settings={
                            "parent_id": pid,
                            "strategy": "warm_start",
                        },
                    )
                )
        elif strategy == "mpnn_redesign":
            for pid in parent_ids:
                runs.append(
                    ToolRunConfig(
                        tool_name="mpnn",
                        num_designs=20,
                        extra_settings={
                            "parent_id": pid,
                            "strategy": "mpnn_redesign",
                            "temperature": 0.1,
                        },
                    )
                )

        return runs
