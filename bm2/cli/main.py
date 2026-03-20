"""bm2 CLI — the user-facing interface to BindMaster 2.0.

Commands:
    bm2 init                              Create workspace
    bm2 create <name> <pdb> <chain>       New campaign
    bm2 run <id> [--through STATE]        Run pipeline
    bm2 status [id]                       Show campaign status
    bm2 report <id>                       Print evaluation report
    bm2 agent <name> <id>                 Run a specific agent
    bm2 tools list|check                  Tool management
    bm2 skills list|query|show            Skills system
    bm2 export designs <id> --top N       Export designs
    bm2 import results <id> <csv>         Import experimental data
"""

from __future__ import annotations

import csv as csv_mod
import json
import logging
import sys
from pathlib import Path

import click

from bm2 import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def cli(verbose):
    """BindMaster 2.0 -- Multi-tool protein binder design platform."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


# ── Workspace ────────────────────────────────────────────────────────────────


@cli.command()
def init():
    """Initialize BM2 workspace at ~/.bm2/"""
    from bm2.core.config import BM2Config
    from bm2.tools.registry import ToolRegistry

    config = BM2Config()
    config.base_dir.mkdir(parents=True, exist_ok=True)
    (config.base_dir / "campaigns").mkdir(exist_ok=True)
    config.save()

    registry = ToolRegistry()
    registry.auto_discover()

    click.echo(f"BM2 workspace initialized at {config.base_dir}")
    installed = registry.list_installed()
    click.echo(f"Tools found: {', '.join(installed) or 'none'}")


# ── Campaign Management ─────────────────────────────────────────────────────


@cli.command()
@click.argument("name")
@click.argument("pdb", type=click.Path(exists=True))
@click.argument("chain")
@click.option("--hotspots", default=None, help="Comma-separated hotspot residues")
def create(name, pdb, chain, hotspots):
    """Create a new design campaign."""
    from bm2.core.campaign import CampaignManager

    mgr = CampaignManager()
    hotspot_list = [h.strip() for h in hotspots.split(",")] if hotspots else []
    campaign = mgr.create(name, Path(pdb), [chain], hotspot_list)

    click.echo(f"Campaign created: {campaign.id}")
    click.echo(f"Target: {pdb} chain {chain}")
    if hotspot_list:
        click.echo(f"Hotspots: {', '.join(hotspot_list)}")
    click.echo(f"Directory: {mgr.campaign_dir(campaign.id)}")


@cli.command("run")
@click.argument("campaign_id")
@click.option(
    "--through",
    "stop_at",
    default="ranked",
    type=click.Choice(
        ["analyzing", "planning", "designing", "evaluating", "ranked", "wet_lab_prep"]
    ),
    help="Run pipeline through this state",
)
@click.option("--total-designs", default=500, help="Total designs across tools")
def run_campaign(campaign_id, stop_at, total_designs):
    """Run the design pipeline for a campaign."""
    from bm2.agents.campaign_orchestrator import CampaignOrchestrator
    from bm2.core.campaign import CampaignManager
    from bm2.core.config import BM2Config
    from bm2.core.models import CampaignState
    from bm2.tools.registry import ToolRegistry

    config = BM2Config.load()
    mgr = CampaignManager(config.base_dir)
    campaign = mgr.load(campaign_id)

    registry = ToolRegistry()
    registry.auto_discover()

    state_map = {
        "analyzing": CampaignState.ANALYZING,
        "planning": CampaignState.PLANNING,
        "designing": CampaignState.DESIGNING,
        "evaluating": CampaignState.EVALUATING,
        "ranked": CampaignState.RANKED,
        "wet_lab_prep": CampaignState.WET_LAB_PREP,
    }

    save_path = mgr.campaign_dir(campaign_id) / "campaign.json"
    orchestrator = CampaignOrchestrator(registry, mgr, config)
    campaign = orchestrator.run_through(
        campaign,
        stop_at=state_map[stop_at],
        save_path=save_path,
        total_designs=total_designs,
    )

    click.echo(f"Pipeline complete. State: {campaign.state.value}")

    if campaign.evaluation_dir:
        report_path = Path(campaign.evaluation_dir) / "report.txt"
        if report_path.exists():
            click.echo("\n" + report_path.read_text())


@cli.command()
@click.argument("campaign_id", required=False)
def status(campaign_id):
    """Show campaign status. Without ID, list all campaigns."""
    from bm2.core.campaign import CampaignManager

    mgr = CampaignManager()

    if campaign_id:
        campaign = mgr.load(campaign_id)
        click.echo(f"Campaign: {campaign.id}")
        click.echo(f"Name:     {campaign.name}")
        click.echo(f"State:    {campaign.state.value}")
        click.echo(f"Created:  {campaign.created_at}")
        if campaign.target:
            click.echo(
                f"Target:   {campaign.target.pdb_path.name} "
                f"({campaign.target.target_length} residues)"
            )
        click.echo(f"Tool runs: {len(campaign.tool_runs)}")
        for tr in campaign.tool_runs:
            click.echo(f"  {tr.tool_name}: {tr.status} ({tr.num_designs} designs)")
    else:
        campaigns = mgr.list_campaigns()
        if not campaigns:
            click.echo("No campaigns. Run 'bm2 create' to start.")
            return
        click.echo(f"{'ID':<40} {'State':<15} {'Created':<20}")
        click.echo("-" * 75)
        for c in campaigns:
            click.echo(f"{c['id']:<40} {c['state']:<15} {c['created']:<20}")


@cli.command()
@click.argument("campaign_id")
def report(campaign_id):
    """Print the evaluation report for a campaign."""
    from bm2.core.campaign import CampaignManager

    mgr = CampaignManager()
    campaign = mgr.load(campaign_id)

    if not campaign.evaluation_dir:
        click.echo("No evaluation results yet. Run 'bm2 run' first.")
        sys.exit(1)

    report_path = Path(campaign.evaluation_dir) / "report.txt"
    if report_path.exists():
        click.echo(report_path.read_text())
    else:
        click.echo(f"Report not found at {report_path}")


# ── Agent Control ────────────────────────────────────────────────────────────


@cli.group()
def agent():
    """Run individual agents on a campaign."""


@agent.command("analyze")
@click.argument("campaign_id")
def agent_analyze(campaign_id):
    """Run target analysis."""
    _run_single_agent(campaign_id, "analyze")


@agent.command("plan")
@click.argument("campaign_id")
@click.option("--total-designs", default=500)
def agent_plan(campaign_id, total_designs):
    """Run strategy planning."""
    _run_single_agent(campaign_id, "plan", total_designs=total_designs)


@agent.command("design")
@click.argument("campaign_id")
def agent_design(campaign_id):
    """Run design tools."""
    _run_single_agent(campaign_id, "design")


@agent.command("evaluate")
@click.argument("campaign_id")
def agent_evaluate(campaign_id):
    """Run cross-model evaluation."""
    _run_single_agent(campaign_id, "evaluate")


@agent.command("wetlab")
@click.argument("campaign_id")
@click.option("--num", default=20, help="Designs for testing")
@click.option("--budget", default=10000, help="Budget in USD")
def agent_wetlab(campaign_id, num, budget):
    """Generate wet-lab testing plan."""
    _run_single_agent(campaign_id, "wetlab", num_to_test=num, budget_usd=budget)


@agent.command("mature")
@click.argument("campaign_id")
@click.option(
    "--strategy",
    default="auto",
    type=click.Choice(["auto", "mpnn_redesign", "partial_diffusion", "warm_start_hallucination"]),
)
def agent_mature(campaign_id, strategy):
    """Plan maturation of promising designs."""
    _run_single_agent(campaign_id, "mature", strategy=strategy)


def _run_single_agent(campaign_id, agent_name, **kwargs):
    from bm2.agents.campaign_orchestrator import CampaignOrchestrator
    from bm2.core.campaign import CampaignManager
    from bm2.core.config import BM2Config
    from bm2.tools.registry import ToolRegistry

    config = BM2Config.load()
    mgr = CampaignManager(config.base_dir)
    campaign = mgr.load(campaign_id)
    registry = ToolRegistry()
    registry.auto_discover()

    save_path = mgr.campaign_dir(campaign_id) / "campaign.json"
    orchestrator = CampaignOrchestrator(registry, mgr, config)
    orchestrator.run_agent(campaign, agent_name, save_path=save_path, **kwargs)

    click.echo(f"Agent '{agent_name}' complete. State: {campaign.state.value}")


# ── Tools ────────────────────────────────────────────────────────────────────


@cli.group()
def tools():
    """Manage design tools."""


@tools.command("list")
def tools_list():
    """List all registered tools."""
    from bm2.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.auto_discover()

    for name in registry.list_installed():
        launcher = registry.get(name)
        click.echo(f"  + {name:<15} {launcher.env_spec}")

    if not registry.list_installed():
        click.echo("No tools found. Run 'bm2 skills show tool-installation'.")


@tools.command("check")
def tools_check():
    """Verify all tool installations."""
    from bm2.tools.registry import ToolRegistry

    registry = ToolRegistry()
    all_names = ["bindcraft", "boltzgen", "mosaic", "pxdesign", "rfdiffusion", "complexa"]
    registry.auto_discover()
    installed = registry.list_installed()

    for name in all_names:
        if name in installed:
            launcher = registry.get(name)
            click.echo(f"  + {name:<15} {launcher.env_spec}")
        else:
            click.echo(f"  x {name:<15} NOT INSTALLED")


# ── Skills ───────────────────────────────────────────────────────────────────


@cli.group()
def skills():
    """Query domain-expertise knowledge."""


@skills.command("list")
def skills_list():
    """List all available skills."""
    from bm2.skills.manager import SkillsManager

    mgr = SkillsManager()
    for info in mgr.list_all():
        tag = " (custom)" if info["source"] == "custom" else ""
        click.echo(f"  {info['name']:<25} {info['description']}{tag}")


@skills.command("query")
@click.argument("question", nargs=-1)
def skills_query(question):
    """Ask a question -- find relevant skills."""
    from bm2.skills.manager import SkillsManager

    mgr = SkillsManager()
    q = " ".join(question)
    results = mgr.query(q, top_n=3)

    if not results:
        click.echo("No relevant skills found. Try different keywords.")
        return

    for skill in results:
        click.echo(f"\n--- {skill.name}: {skill.description} ---\n")
        click.echo(skill.content)


@skills.command("show")
@click.argument("name")
def skills_show(name):
    """Show full content of a specific skill."""
    from bm2.skills.manager import SkillsManager

    mgr = SkillsManager()
    try:
        skill = mgr.get(name)
        click.echo(f"\n--- {skill.name}: {skill.description} ---\n")
        click.echo(skill.content)
    except KeyError as e:
        click.echo(str(e))
        sys.exit(1)


# ── Export / Import ──────────────────────────────────────────────────────────


@cli.group("export")
def export_group():
    """Export designs from a campaign."""


@export_group.command("designs")
@click.argument("campaign_id")
@click.option("--top", required=True, type=int, help="Number of top designs")
@click.option(
    "--format", "fmt", type=click.Choice(["fasta", "csv", "json"]), default="fasta"
)
@click.option("--output", "-o", default=None, help="Output file path")
def export_designs(campaign_id, top, fmt, output):
    """Export top designs as FASTA, CSV, or JSON."""
    from bm2.core.campaign import CampaignManager

    mgr = CampaignManager()
    campaign = mgr.load(campaign_id)

    if not campaign.evaluation_dir:
        click.echo("No evaluation results. Run 'bm2 run' first.")
        sys.exit(1)

    summary_csv = Path(campaign.evaluation_dir) / "evaluation_summary.csv"
    if not summary_csv.exists():
        click.echo(f"Summary CSV not found: {summary_csv}")
        sys.exit(1)

    with open(summary_csv, newline="") as f:
        designs = list(csv_mod.DictReader(f))[:top]

    out_path = (
        Path(output)
        if output
        else Path(campaign.evaluation_dir) / f"top_{top}_designs.{fmt}"
    )

    if fmt == "fasta":
        with open(out_path, "w") as f:
            for d in designs:
                f.write(
                    f">{d.get('design_id', '')}|{d.get('source_tool', '')}|"
                    f"rank={d.get('rank', '')}|tier={d.get('tier', '')}\n"
                )
                f.write(f"{d.get('binder_sequence', '')}\n")
    elif fmt == "csv":
        with open(out_path, "w", newline="") as f:
            if designs:
                writer = csv_mod.DictWriter(f, fieldnames=designs[0].keys())
                writer.writeheader()
                writer.writerows(designs)
    elif fmt == "json":
        with open(out_path, "w") as f:
            json.dump(designs, f, indent=2)

    click.echo(f"Exported {len(designs)} designs to {out_path}")


@cli.group("import")
def import_group():
    """Import data into a campaign."""


@import_group.command("results")
@click.argument("campaign_id")
@click.argument("csv_path", type=click.Path(exists=True))
def import_results(campaign_id, csv_path):
    """Import experimental results CSV (design_id, binds, kd_nm, notes)."""
    from bm2.core.campaign import CampaignManager

    mgr = CampaignManager()
    campaign = mgr.load(campaign_id)

    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        results = []
        for row in reader:
            binds_raw = row.get("binds", "").lower().strip()
            binds = binds_raw in ("yes", "true", "1")
            results.append(
                {
                    "design_id": row.get("design_id", ""),
                    "binds": binds,
                    "kd_nm": float(row["kd_nm"]) if row.get("kd_nm") else None,
                    "notes": row.get("notes", ""),
                }
            )

    campaign.experimental_results.extend(results)
    mgr.save(campaign)

    n_binders = sum(1 for r in results if r["binds"])
    click.echo(f"Imported {len(results)} results ({n_binders} binders)")


# ── Entry Point ──────────────────────────────────────────────────────────────


def main():
    cli()


if __name__ == "__main__":
    main()
