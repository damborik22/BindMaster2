"""Core data models for BM2 campaigns.

Campaign, TargetProfile, ToolRunConfig, and the campaign state machine.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class CampaignState(str, enum.Enum):
    """Campaign lifecycle states."""

    INIT = "init"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    DESIGNING = "designing"
    EVALUATING = "evaluating"
    RANKED = "ranked"
    WET_LAB_PREP = "wet_lab_prep"
    TESTING = "testing"
    MATURING = "maturing"
    COMPLETED = "completed"
    FAILED = "failed"


# Valid state transitions
TRANSITIONS: dict[CampaignState, list[CampaignState]] = {
    CampaignState.INIT: [CampaignState.ANALYZING],
    CampaignState.ANALYZING: [CampaignState.PLANNING, CampaignState.FAILED],
    CampaignState.PLANNING: [CampaignState.DESIGNING, CampaignState.FAILED],
    CampaignState.DESIGNING: [CampaignState.EVALUATING, CampaignState.FAILED],
    CampaignState.EVALUATING: [CampaignState.RANKED, CampaignState.FAILED],
    CampaignState.RANKED: [CampaignState.WET_LAB_PREP, CampaignState.MATURING],
    CampaignState.WET_LAB_PREP: [CampaignState.TESTING],
    CampaignState.TESTING: [CampaignState.MATURING, CampaignState.COMPLETED],
    CampaignState.MATURING: [CampaignState.DESIGNING],  # Loop back for iteration
    CampaignState.COMPLETED: [],
    CampaignState.FAILED: [CampaignState.INIT],  # Can restart
}


class DesignSource(str, enum.Enum):
    """Design tool that produced binder designs."""

    BINDCRAFT = "bindcraft"
    BOLTZGEN = "boltzgen"
    MOSAIC = "mosaic"
    PXDESIGN = "pxdesign"
    RFDIFFUSION = "rfdiffusion"
    COMPLEXA = "complexa"
    MANUAL = "manual"


@dataclass
class TargetProfile:
    """Everything we know about the target protein."""

    pdb_path: Path
    chains: list[str]
    target_sequence: str = ""
    target_length: int = 0
    hotspot_residues: list[str] = field(default_factory=list)
    difficulty_score: float = 0.0
    sasa_per_residue: dict[str, float] = field(default_factory=dict)
    recommended_tools: list[str] = field(default_factory=list)
    suggested_length_range: tuple[int, int] = (60, 100)
    suggested_modality: str = "protein"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "pdb_path": str(self.pdb_path),
            "chains": self.chains,
            "target_sequence": self.target_sequence,
            "target_length": self.target_length,
            "hotspot_residues": self.hotspot_residues,
            "difficulty_score": self.difficulty_score,
            "recommended_tools": self.recommended_tools,
            "suggested_length_range": list(self.suggested_length_range),
            "suggested_modality": self.suggested_modality,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetProfile:
        length_range = data.get("suggested_length_range", [60, 100])
        return cls(
            pdb_path=Path(data["pdb_path"]),
            chains=data["chains"],
            target_sequence=data.get("target_sequence", ""),
            target_length=data.get("target_length", 0),
            hotspot_residues=data.get("hotspot_residues", []),
            difficulty_score=data.get("difficulty_score", 0.0),
            recommended_tools=data.get("recommended_tools", []),
            suggested_length_range=tuple(length_range),
            suggested_modality=data.get("suggested_modality", "protein"),
            notes=data.get("notes", ""),
        )


@dataclass
class ToolRunConfig:
    """Configuration for one tool run within a campaign."""

    tool_name: str
    num_designs: int = 100
    binder_length_range: tuple[int, int] = (60, 100)
    hotspot_residues: list[str] = field(default_factory=list)
    extra_settings: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    output_dir: Optional[Path] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "num_designs": self.num_designs,
            "binder_length_range": list(self.binder_length_range),
            "hotspot_residues": self.hotspot_residues,
            "extra_settings": self.extra_settings,
            "status": self.status,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolRunConfig:
        length_range = data.get("binder_length_range", [60, 100])
        output_dir = data.get("output_dir")
        return cls(
            tool_name=data["tool_name"],
            num_designs=data.get("num_designs", 100),
            binder_length_range=tuple(length_range),
            hotspot_residues=data.get("hotspot_residues", []),
            extra_settings=data.get("extra_settings", {}),
            status=data.get("status", "pending"),
            output_dir=Path(output_dir) if output_dir else None,
            error=data.get("error"),
        )


@dataclass
class Campaign:
    """Central object tracking a binder design campaign."""

    id: str
    name: str
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    state: CampaignState = CampaignState.INIT
    target: Optional[TargetProfile] = None
    tool_runs: list[ToolRunConfig] = field(default_factory=list)
    eval_engines: list[str] = field(
        default_factory=lambda: ["boltz2", "af2"]
    )
    eval_rosetta: bool = False
    eval_config: dict[str, Any] = field(default_factory=dict)
    evaluation_dir: Optional[Path] = None
    maturation_rounds: list[dict] = field(default_factory=list)
    experimental_results: list[dict] = field(default_factory=list)
    notes: str = ""

    def transition_to(self, new_state: CampaignState) -> None:
        """Validate and execute state transition.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self.state.value} -> {new_state.value}. "
                f"Valid transitions: {[s.value for s in allowed]}"
            )
        self.state = new_state

    def save(self, path: Path) -> None:
        """Persist campaign to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> Campaign:
        """Load campaign from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "state": self.state.value,
            "target": self.target.to_dict() if self.target else None,
            "tool_runs": [tr.to_dict() for tr in self.tool_runs],
            "eval_engines": self.eval_engines,
            "eval_rosetta": self.eval_rosetta,
            "eval_config": self.eval_config,
            "evaluation_dir": str(self.evaluation_dir)
            if self.evaluation_dir
            else None,
            "maturation_rounds": self.maturation_rounds,
            "experimental_results": self.experimental_results,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Campaign:
        target_data = data.get("target")
        eval_dir = data.get("evaluation_dir")
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data.get("created_at", ""),
            state=CampaignState(data.get("state", "init")),
            target=TargetProfile.from_dict(target_data) if target_data else None,
            tool_runs=[
                ToolRunConfig.from_dict(tr) for tr in data.get("tool_runs", [])
            ],
            eval_engines=data.get("eval_engines", ["boltz2", "af2"]),
            eval_rosetta=data.get("eval_rosetta", False),
            eval_config=data.get("eval_config", {}),
            evaluation_dir=Path(eval_dir) if eval_dir else None,
            maturation_rounds=data.get("maturation_rounds", []),
            experimental_results=data.get("experimental_results", []),
            notes=data.get("notes", ""),
        )
