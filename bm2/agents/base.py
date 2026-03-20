"""Agent base class. Agents are deterministic logic, not LLM calls."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from bm2.core.models import Campaign, CampaignState

logger = logging.getLogger(__name__)


class Agent(ABC):
    """An agent performs one stage of the campaign lifecycle.

    It reads campaign state, does work, updates campaign, transitions state.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""

    @property
    @abstractmethod
    def required_state(self) -> CampaignState:
        """Campaign must be in this state for agent to run."""

    @property
    @abstractmethod
    def target_state(self) -> CampaignState:
        """State to transition to on success."""

    @abstractmethod
    def _execute(self, campaign: Campaign, **kwargs) -> None:
        """Do the work. Modify campaign in place."""

    def run(
        self, campaign: Campaign, save_path: Path | None = None, **kwargs
    ) -> Campaign:
        """Public entry point. Validates state, executes, transitions.

        Args:
            campaign: Campaign to operate on.
            save_path: Path to save campaign state after execution.
            **kwargs: Passed to _execute.

        Returns:
            Modified campaign.
        """
        compatible = self._compatible_states()
        if campaign.state not in compatible:
            raise ValueError(
                f"{self.name} requires state "
                f"{[s.value for s in compatible]}, "
                f"got {campaign.state.value}"
            )

        logger.info(f"Agent {self.name} starting (campaign: {campaign.id})")

        try:
            self._execute(campaign, **kwargs)
            campaign.transition_to(self.target_state)
            logger.info(
                f"Agent {self.name} complete -> {self.target_state.value}"
            )
        except Exception as e:
            logger.error(f"Agent {self.name} failed: {e}")
            try:
                campaign.transition_to(CampaignState.FAILED)
            except ValueError:
                campaign.state = CampaignState.FAILED
            campaign.notes += f"\n{self.name} failed: {e}"
            raise
        finally:
            if save_path:
                campaign.save(save_path)

        return campaign

    def _compatible_states(self) -> list[CampaignState]:
        """Override to allow running from additional states."""
        return [self.required_state]
