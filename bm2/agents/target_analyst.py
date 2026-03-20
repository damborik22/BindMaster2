"""Agent 1: Analyze target structure, assess difficulty, recommend tools."""

from __future__ import annotations

import logging

from bm2.agents.base import Agent
from bm2.core.models import Campaign, CampaignState
from bm2.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class TargetAnalyst(Agent):
    """Analyzes target and recommends tools and parameters."""

    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    @property
    def name(self) -> str:
        return "target_analyst"

    @property
    def required_state(self) -> CampaignState:
        return CampaignState.INIT

    @property
    def target_state(self) -> CampaignState:
        return CampaignState.ANALYZING

    def _execute(self, campaign: Campaign, **kwargs) -> None:
        target = campaign.target
        if target is None:
            raise ValueError("Campaign has no target. Set target before analysis.")

        # 1. Parse PDB
        from bm2.core.target import parse_target_pdb

        seqs = parse_target_pdb(target.pdb_path, target.chains)
        target.target_sequence = "".join(seqs.values())
        target.target_length = len(target.target_sequence)
        logger.info(
            f"Target: {target.target_length} residues, chains {target.chains}"
        )

        # 2. Compute SASA
        try:
            from bm2.core.target import compute_sasa

            target.sasa_per_residue = {
                str(k): v
                for k, v in compute_sasa(
                    target.pdb_path, target.chains[0]
                ).items()
            }
        except Exception as e:
            logger.warning(f"SASA computation failed: {e}")

        # 3. Auto-detect hotspots if not provided
        if not target.hotspot_residues:
            try:
                from bm2.core.target import auto_detect_hotspots

                target.hotspot_residues = auto_detect_hotspots(
                    target.pdb_path,
                    target.chains[0],
                    {int(k): v for k, v in target.sasa_per_residue.items()}
                    if target.sasa_per_residue
                    else None,
                )
                logger.info(
                    f"Auto-detected {len(target.hotspot_residues)} hotspots"
                )
            except Exception as e:
                logger.warning(f"Hotspot detection failed: {e}")

        # 4. Assess difficulty
        from bm2.core.target import assess_difficulty

        target.difficulty_score = assess_difficulty(target)
        logger.info(f"Difficulty score: {target.difficulty_score:.2f}")

        # 5. Recommend tools
        target.recommended_tools = self._recommend_tools(target)
        logger.info(f"Recommended tools: {target.recommended_tools}")

        # 6. Suggest binder properties
        target.suggested_length_range = self._suggest_length(target)
        target.suggested_modality = self._suggest_modality(target)

    def _recommend_tools(self, target) -> list[str]:
        """Recommend tools based on target properties + availability.

        Core principle: always recommend at least 2 tools for diversity.
        Different tools find different solutions.
        """
        installed = self.registry.list_installed()
        recommended = []

        # Tier 1: Proven, reliable
        for tool in ["bindcraft", "boltzgen"]:
            if tool in installed:
                recommended.append(tool)

        # Tier 2: Backbone diversity
        if "rfdiffusion" in installed:
            recommended.append("rfdiffusion")

        # Tier 3: Hard targets benefit from test-time compute scaling
        if target.difficulty_score > 0.6 and "complexa" in installed:
            recommended.append("complexa")

        # Tier 4: Additional methods
        for tool in ["pxdesign", "mosaic"]:
            if tool in installed and tool not in recommended:
                recommended.append(tool)

        if not recommended and installed:
            recommended = installed[:1]

        return recommended

    def _suggest_length(self, target) -> tuple[int, int]:
        """Suggest binder length range based on target size."""
        if target.target_length < 200:
            return (40, 80)
        elif target.target_length < 500:
            return (60, 100)
        else:
            return (80, 120)

    def _suggest_modality(self, target) -> str:
        """Suggest binder modality."""
        return "protein"
