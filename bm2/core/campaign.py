"""Campaign manager — handles persistence, directory structure, state tracking.

Directory structure:
    ~/.bm2/
    +-- config.toml           Global BM2 config
    +-- campaigns/
        +-- {campaign_id}/
            +-- campaign.json  Campaign state
            +-- target/        Target PDB + analysis
            +-- runs/          One subdir per tool run
            +-- evaluation/    bm2-evaluator output
            +-- reports/       Generated reports
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from bm2.core.models import Campaign, CampaignState, TargetProfile

logger = logging.getLogger(__name__)


class CampaignManager:
    """Manages campaigns on disk."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path.home() / ".bm2"
        self.campaigns_dir = self.base_dir / "campaigns"

    def create(
        self,
        name: str,
        target_pdb: Path,
        chains: list[str],
        hotspots: list[str] | None = None,
    ) -> Campaign:
        """Create a new campaign.

        Args:
            name: Campaign name (e.g., "PDL1_binders").
            target_pdb: Path to target PDB file.
            chains: Target chain IDs (e.g., ["A"]).
            hotspots: Optional hotspot residue IDs (e.g., ["A10", "A25"]).

        Returns:
            Created Campaign object.
        """
        campaign_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        campaign_dir = self.campaigns_dir / campaign_id

        # Create directory structure
        target_dir = campaign_dir / "target"
        target_dir.mkdir(parents=True, exist_ok=True)
        (campaign_dir / "runs").mkdir(exist_ok=True)
        (campaign_dir / "evaluation").mkdir(exist_ok=True)
        (campaign_dir / "reports").mkdir(exist_ok=True)

        # Copy target PDB
        target_pdb = Path(target_pdb)
        target_copy = target_dir / target_pdb.name
        shutil.copy2(target_pdb, target_copy)

        campaign = Campaign(
            id=campaign_id,
            name=name,
            target=TargetProfile(
                pdb_path=target_copy,
                chains=chains,
                hotspot_residues=hotspots or [],
            ),
        )

        campaign.save(campaign_dir / "campaign.json")
        logger.info(f"Campaign created: {campaign_id} at {campaign_dir}")
        return campaign

    def load(self, campaign_id: str) -> Campaign:
        """Load an existing campaign.

        Raises:
            FileNotFoundError: If campaign doesn't exist.
        """
        path = self.campaigns_dir / campaign_id / "campaign.json"
        if not path.exists():
            raise FileNotFoundError(f"Campaign not found: {campaign_id}")
        return Campaign.load(path)

    def save(self, campaign: Campaign) -> None:
        """Save campaign state to disk."""
        path = self.campaigns_dir / campaign.id / "campaign.json"
        campaign.save(path)

    def list_campaigns(self) -> list[dict[str, str]]:
        """List all campaigns with their status."""
        campaigns = []
        if not self.campaigns_dir.exists():
            return campaigns

        for d in sorted(self.campaigns_dir.iterdir()):
            if not d.is_dir():
                continue
            json_path = d / "campaign.json"
            if json_path.exists():
                try:
                    c = Campaign.load(json_path)
                    campaigns.append(
                        {
                            "id": c.id,
                            "name": c.name,
                            "state": c.state.value,
                            "created": c.created_at,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to load campaign {d.name}: {e}")

        return campaigns

    def campaign_dir(self, campaign_id: str) -> Path:
        """Get the directory for a campaign."""
        return self.campaigns_dir / campaign_id

    def run_dir(self, campaign_id: str, tool_name: str) -> Path:
        """Get the run directory for a specific tool within a campaign."""
        d = self.campaigns_dir / campaign_id / "runs" / tool_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def evaluation_dir(self, campaign_id: str) -> Path:
        """Get the evaluation directory for a campaign."""
        d = self.campaigns_dir / campaign_id / "evaluation"
        d.mkdir(parents=True, exist_ok=True)
        return d
