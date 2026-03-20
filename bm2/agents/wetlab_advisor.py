"""Agent 5: Generate wet-lab testing plan."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from bm2.agents.base import Agent
from bm2.core.models import Campaign, CampaignState

logger = logging.getLogger(__name__)


class WetLabAdvisor(Agent):
    """Generates comprehensive wet-lab testing plan from ranked designs."""

    @property
    def name(self) -> str:
        return "wetlab_advisor"

    @property
    def required_state(self) -> CampaignState:
        return CampaignState.RANKED

    @property
    def target_state(self) -> CampaignState:
        return CampaignState.WET_LAB_PREP

    def _execute(
        self,
        campaign: Campaign,
        num_to_test: int = 20,
        budget_usd: int = 10000,
        **kwargs,
    ) -> None:
        eval_dir = campaign.evaluation_dir
        if not eval_dir or not (Path(eval_dir) / "evaluation_summary.csv").exists():
            raise FileNotFoundError(
                "No evaluation results. Run evaluator first."
            )

        with open(Path(eval_dir) / "evaluation_summary.csv", newline="") as f:
            designs = list(csv.DictReader(f))

        top = designs[:num_to_test]
        consensus = [d for d in top if d.get("tier") == "consensus_hit"]
        strong = [d for d in top if d.get("tier") == "strong"]

        sections = [
            self._header(campaign, len(top)),
            self._selection(top, consensus, strong),
            self._synthesis(top, budget_usd),
            self._expression(),
            self._screening(len(top), budget_usd),
            self._characterization(),
            self._controls(),
            self._design_table(top),
        ]

        # Write plan
        reports_dir = Path(eval_dir).parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        plan_path = reports_dir / "wetlab_plan.md"
        plan_path.write_text("\n\n".join(sections))

        # Export FASTA
        fasta_path = reports_dir / "top_designs.fasta"
        with open(fasta_path, "w") as f:
            for d in top:
                did = d.get("design_id", "unknown")
                src = d.get("source_tool", "")
                rank = d.get("rank", "")
                f.write(f">{did}|{src}|rank={rank}\n")
                f.write(f"{d.get('binder_sequence', '')}\n")

        logger.info(f"Wet-lab plan: {plan_path}")
        logger.info(f"Top designs FASTA: {fasta_path}")

    def _header(self, campaign, n):
        return (
            f"# Wet-Lab Testing Plan: {campaign.name}\n\n"
            f"Campaign: {campaign.id}\n"
            f"Designs selected: {n}\n"
        )

    def _selection(self, top, consensus, strong):
        other = len(top) - len(consensus) - len(strong)
        return (
            "# Design Selection\n\n"
            f"- {len(consensus)} consensus hits (both engines agree)\n"
            f"- {len(strong)} strong candidates\n"
            f"- {other} additional by composite score\n\n"
            "Consensus hits should be prioritized."
        )

    def _synthesis(self, top, budget):
        n = len(top)
        cost = n * 150
        return (
            "# Gene Synthesis\n\n"
            f"Order {n} genes. Est. cost: ~${cost}\n"
            "Construct: His6-TEV-binder in pET vector\n"
            "Codon optimization: E. coli\n"
            "Vendors: Twist Bioscience (~2 wk) or IDT gBlocks (~1 wk)\n"
        )

    def _expression(self):
        return (
            "# Expression\n\n"
            "E. coli BL21(DE3), pET vector\n"
            "Induce OD600 ~0.6 with 0.5 mM IPTG\n"
            "18C overnight (soluble) or 37C 4h\n"
            "Purify: Ni-NTA (IMAC) then SEC\n"
        )

    def _screening(self, n, budget):
        if n > 50:
            method = "Split-luciferase (NanoBiT) — high-throughput"
        elif n > 10:
            method = "BLI (Octet) — medium throughput, label-free"
        else:
            method = "SPR (Biacore) — full kinetics"
        return (
            "# Binding Screening\n\n"
            f"Recommended: {method}\n"
            f"Screen all {n} at 1 uM, dose-response on hits\n"
        )

    def _characterization(self):
        return (
            "# Hit Characterization\n\n"
            "1. SPR/BLI kinetics (kon, koff, KD)\n"
            "2. DSF thermal stability (Tm > 50C)\n"
            "3. SEC-MALS monodispersity\n"
            "4. X-ray crystallography for top 1-3\n"
        )

    def _controls(self):
        return (
            "# Controls\n\n"
            "- Positive: known binder (if available)\n"
            "- Negative: scrambled sequence\n"
            "- Target-only: DSF/SEC to verify folding\n"
        )

    def _design_table(self, top):
        header = (
            "# Selected Designs\n\n"
            "| Rank | ID | Source | ipSAE_min | ipTM | Tier | Length |\n"
            "|------|-----|--------|-----------|------|------|--------|\n"
        )
        rows = []
        for d in top:
            rows.append(
                f"| {d.get('rank', '')} | {d.get('design_id', '')} | "
                f"{d.get('source_tool', '')} | "
                f"{d.get('ensemble_ipsae_min', '')} | "
                f"{d.get('ensemble_iptm', '')} | "
                f"{d.get('tier', '')} | {d.get('binder_length', '')} |"
            )
        return header + "\n".join(rows)
