# BindMaster 2.0 — Master Plan for Claude Code

## HOW TO USE THIS DOCUMENT

Place as `CLAUDE.md` in the BM2 repo root. Claude Code reads it automatically.

**Before EVERY section:** Deploy research agents. Read actual source code. Validate assumptions. Present findings. Wait for approval. Only then implement.

**Scientific rigor rule:** Every metric, every normalization, every threshold must trace to a published formula or reference implementation. No ad-hoc data editing. Raw data in → documented processing → validated output.

---

# PART 1: ARCHITECTURE

## 1.1 Philosophy

The design tools work. BindCraft installs easily and performs well. BoltzGen has its own pipeline. Don't reinvent them. Instead, build two things that don't exist yet:

1. **A standalone Evaluator** that takes designs from ANY tool and scores them with scientific rigor — completely independent of any design tool's environment
2. **An Agent + Skills layer** that helps users choose tools, configure runs, interpret results, and plan experiments

```
┌─────────────────────────────────────────────────────┐
│                  BM2 Agent + Skills                  │
│  (recommends tools, interprets results, guides user) │
└──────────┬──────────────────────────────┬────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐    ┌──────────────────────────┐
│   Design Tools       │    │   BM2 Evaluator          │
│   (external, as-is)  │───▶│   (standalone, own env)   │
│                      │    │                          │
│  • BindCraft         │    │  • Boltz2 refolding      │
│  • BoltzGen          │    │  • AF2 refolding         │
│  • Mosaic            │    │  • ipSAE (Dunbrack)      │
│  • PXDesign          │    │  • PyRosetta (optional)  │
│  • RFAA + LigandMPNN │    │  • Composite scoring     │
│  • Proteina-Complexa │    │  • Ranking + reporting   │
│                      │    │                          │
│  Each in own env     │    │  ONE env: bm2_eval       │
└─────────────────────┘    └──────────────────────────┘
```

## 1.2 Three Separate Components

### Component A: BM2 Evaluator (standalone Python package)
- **Package name:** `bm2-evaluator`
- **Own conda env:** `bm2_eval` (NOT Mosaic, NOT BindCraft, NOT any tool env)
- **Dependencies:** numpy, biopython, gemmi (minimal). Optional: pyrosetta
- **Refolding deps:** boltz2, colabfold (each can be in sub-envs called via subprocess)
- **Can be used without BM2 at all** — anyone can pip install it and score their designs
- **Input:** directory of PDB files + FASTA sequences + metadata CSV (tool, binder chain, target chain)
- **Output:** scored CSV with all metrics (raw + normalized) + PAE matrices + report

### Component B: BM2 Agents (orchestration layer)
- **Package name:** `bm2`
- **Depends on:** bm2-evaluator
- **Manages:** launching tools in their own envs, feeding output to evaluator, tracking campaigns
- **6 agents** covering the full lifecycle

### Component C: BM2 Skills (knowledge layer)
- **Part of:** `bm2` package
- **8 domain-expertise modules** for conversational AI guidance
- **Extensible:** users add custom skills as markdown files

## 1.3 The 6 Design Tools (External, Untouched)

Each tool stays in its own conda environment. BM2 calls them as subprocesses.

| Tool | Env | What It Does | Install |
|------|-----|-------------|---------|
| BindCraft | `bindcraft` | AF2 hallucination + MPNNsol + AF2 mono filter + Rosetta | `bash install_bindcraft.sh` |
| BoltzGen | `boltzgen` | All-atom diffusion + BoltzIF + Boltz2 validation | `pip install boltzgen` |
| Mosaic | `mosaic` | Multi-objective on AF2/Boltz2 + ProteinMPNN | `pip install escalante-mosaic` |
| PXDesign | `pxdesign` | Protenix hallucination + diffusion + dual filter | ByteDance install / server |
| RFAA + LigandMPNN | `rfdiffusion` | Backbone diffusion + inverse folding | RosettaCommons install |
| Proteina-Complexa | `complexa` | Flow-matching + test-time compute scaling | NVIDIA install |

**BM2 never modifies these tools.** It reads their output directories and feeds designs to the Evaluator.

---

# PART 2: BM2 EVALUATOR — STANDALONE

## 2.1 Design Principles

1. **Standalone.** Works without BM2 agents. Anyone can use it:
   ```bash
   pip install bm2-evaluator
   bm2-eval score --designs ./my_designs/ --target target.pdb --target-chain A
   ```

2. **Raw data sacred.** Never modify tool outputs. Read PDBs and sequences as-is. All transformations happen in documented functions with raw values preserved alongside.

3. **Every number has a source.** Every threshold cites a paper. Every formula cites equations. Every normalization is explicit and logged.

4. **PAE matrices always saved.** Every refolding produces a .npy file. No exceptions.

## 2.2 Package Structure

```
bm2_evaluator/
├── __init__.py
├── cli.py                    # bm2-eval CLI
├── core/
│   ├── models.py             # EvaluationResult, ScoredDesign, EvalConfig
│   └── config.py             # Paths, thresholds, engine selection
├── ingestion/
│   ├── base.py               # DesignIngestor interface
│   ├── bindcraft.py          # Parse BindCraft output (final_designs/, scores.csv)
│   ├── boltzgen.py           # Parse BoltzGen output (PDBs + ranking CSV)
│   ├── mosaic.py             # Parse Mosaic output
│   ├── pxdesign.py           # Parse PXDesign output
│   ├── rfdiffusion.py        # Parse RFdiffusion + LigandMPNN output
│   ├── complexa.py           # Parse Proteina-Complexa output
│   └── generic.py            # Parse any PDB + FASTA directory
├── refolding/
│   ├── base.py               # RefoldingEngine interface
│   ├── boltz2.py             # Boltz2 refolding runner
│   ├── af2.py                # AF2-Multimer refolding runner
│   └── monomer.py            # Binder-alone monomer validation
├── metrics/
│   ├── ipsae.py              # Dunbrack ipSAE (Eq 14-16, d0_res)
│   ├── plddt.py              # pLDDT extraction + normalization
│   ├── iptm.py               # ipTM/pTM extraction
│   ├── pae.py                # PAE matrix handling, chain ordering
│   ├── interface.py          # Contact count, H-bonds, buried SASA
│   └── rosetta.py            # PyRosetta scoring (optional)
├── scoring/
│   ├── composite.py          # Composite score formulas
│   ├── tiers.py              # Quality tier classification
│   ├── ranking.py            # Multi-engine ranking logic
│   └── diversity.py          # Structural clustering for diverse selection
├── reporting/
│   ├── csv_export.py         # Full CSV with raw + normalized columns
│   ├── text_report.py        # Human-readable summary
│   └── comparison.py         # Cross-tool comparison tables
└── tests/
    ├── test_ipsae.py          # Validate against Dunbrack reference
    ├── test_normalization.py  # Scale handling tests
    ├── test_chain_ordering.py # PAE matrix ordering tests
    ├── test_scoring.py        # Composite score tests
    └── fixtures/              # Known PAE matrices, test PDBs
```

## 2.3 Ingestion Layer

Each tool has its own ingestor that knows how to parse that tool's output format. All ingestors produce the same standardized intermediate:

```python
@dataclass
class IngestedDesign:
    design_id: str
    source_tool: str              # "bindcraft", "boltzgen", "mosaic", etc.
    binder_sequence: str
    binder_chain: str             # Chain ID in the PDB
    target_sequence: str
    target_chain: str
    binder_length: int
    target_length: int
    complex_pdb_path: Path        # Path to designed complex PDB
    binder_pdb_path: Optional[Path]  # Binder alone (if available)
    tool_metrics: dict[str, float]   # Tool-native metrics, stored as-is
```

**RESEARCH TASK:** For each tool, Claude Code must read the actual output directory structure and document:
- Where are the PDB files?
- What CSV/JSON metrics files exist?
- What columns do they contain?
- What scales are the metrics on?
- Which chain is the binder, which is the target?

## 2.4 Refolding Layer

```python
class RefoldingEngine(ABC):
    @abstractmethod
    def refold_complex(self, binder_seq, target_seq, output_dir) -> RefoldResult:
        """Refold binder+target as complex. Save PAE matrix."""

    @abstractmethod
    def refold_monomer(self, binder_seq, output_dir) -> MonomerResult:
        """Refold binder alone for monomer validation."""
```

**Boltz2 runner:**
- Conda env: `bm2_boltz2` (or shared with existing Boltz2 install)
- Command: `boltz predict input.fasta --output_dir DIR --save_pae`
- PAE matrix: .npy, binder-first ordering
- pLDDT: 0-1 scale
- ipTM: 0-1 scale
- ΔG affinity prediction: available

**AF2 runner:**
- Conda env: `bm2_af2` (or shared with existing ColabFold install)
- Command: `colabfold_batch input.fasta DIR --model-type alphafold2_multimer_v3`
- PAE matrix: from JSON/pkl, target-first ordering — MUST save as .npy
- pLDDT: 0-100 scale (RAW — normalize later)
- ipTM: 0-1 scale

**Monomer validation:**
- Run binder sequence alone through AF2 monomer or Boltz2
- Compare predicted binder structure (alone) vs binder structure (in complex)
- Report binder_rmsd: if > 3.0Å, binder fold depends on target → flag

## 2.5 Metrics Layer

### ipSAE (`metrics/ipsae.py`)
**Source:** Dunbrack 2025 Eq 14-16. Reference: github.com/DunbrackLab/IPSAE
**d0 variant:** d0_res. **PAE cutoff:** 10Å.

```python
def compute_d0_res(n_interface_residues: int) -> float:
    """d0 = max(0.5, 1.24 * (n - 15)^(1/3) - 1.8)"""
    # Source: Dunbrack 2025, Equation 15

def compute_ipsae_directional(pae_matrix, source_res, target_res, cutoff=10.0) -> float:
    """Dunbrack Eq 14-16 in one direction."""
    # Source: github.com/DunbrackLab/IPSAE/blob/main/ipsae.py

def compute_ipsae(pae_matrix, binder_res, target_res, cutoff=10.0) -> dict:
    """Returns: bt_ipsae, tb_ipsae, ipsae_min, ipsae_max"""
    # ipsae_min = min(bt, tb) — Overath "weakest link" for ranking
    # Source: Overath 2025, bioRxiv 2025.08.14.670059
```

**VALIDATION REQUIREMENT:** Before deployment, compute ipSAE on a known PAE matrix using both our code AND the Dunbrack `ipsae.py`. Results must match within floating-point tolerance.

### PAE Chain Ordering (`metrics/pae.py`)
```python
def get_chain_indices(engine: str, binder_len: int, target_len: int) -> dict:
    """
    Boltz2:  binder=[0, binder_len), target=[binder_len, binder_len+target_len)
    AF2:     target=[0, target_len), binder=[target_len, target_len+binder_len)
    """
    # ASSERT: binder_len + target_len == pae_matrix.shape[0]
```

### pLDDT (`metrics/plddt.py`)
```python
def extract_and_normalize(raw_value: float, engine: str) -> tuple[float, float]:
    """Returns (raw, normalized_0to1).
    AF2: raw is 0-100, normalized = raw/100
    Boltz2: raw is 0-1, normalized = raw
    """
    # Both values stored. User can verify.
```

### PyRosetta (`metrics/rosetta.py`)
```python
def score_interface(pdb_path: str, binder_chain: str) -> Optional[dict]:
    """
    Returns: dG, dSASA, shape_complementarity, n_hbonds, clash_score
    Returns None if PyRosetta not installed.
    """
    try:
        import pyrosetta
        ...
    except ImportError:
        return None  # Graceful degradation
```

## 2.6 Scoring and Ranking

### Composite Scores (`scoring/composite.py`)
```python
def composite_with_rosetta(ipsae_min, dG, dSASA) -> float:
    """Best single predictor per Overath 2025: ipsae_min × |dG / dSASA|"""
    # Source: Overath 2025, bioRxiv 2025.08.14.670059, Table 2

def composite_basic(ipsae_min, iptm, agreement, plddt, pae_mean) -> float:
    """Fallback when Rosetta not available.
    0.40 * ipsae_min + 0.25 * iptm + 0.15 * agreement +
    0.10 * plddt + 0.10 * (1 - min(pae_mean/30, 1))
    """
```

### Tier Classification (`scoring/tiers.py`)
```python
def classify_tier(boltz2_ipsae_min, af2_ipsae_min) -> str:
    """
    consensus_hit: BOTH engines ipsae_min > 0.61
    strong:        ONE > 0.61, OTHER > 0.40
    moderate:      best ipsae_min > 0.40 AND best iptm > 0.6
    weak:          at least one engine passes basic filters
    fail:          nothing passes

    Thresholds: Overath 2025, bioRxiv 2025.08.14.670059
    """
```

## 2.7 Reporting

### CSV Output — Two Levels

**Level 1: Per-design summary** (`evaluation_summary.csv`)
```
design_id, source_tool, binder_length, tier, rank, composite_score,
boltz2_ipsae_min, af2_ipsae_min, ensemble_ipsae_min,
boltz2_iptm, af2_iptm, ensemble_iptm,
boltz2_plddt_raw, af2_plddt_raw, boltz2_plddt_norm, af2_plddt_norm,
boltz2_pae_interaction, af2_pae_interaction,
monomer_rmsd, monomer_passes,
rosetta_dG, rosetta_dSASA, rosetta_shape_comp, rosetta_hbonds,
multi_model_agreement, binder_sequence
```

**Level 2: Per-engine detail** (`evaluation_detail.csv`)
```
design_id, engine, bt_ipsae, tb_ipsae, ipsae_min, ipsae_max,
iptm_raw, ptm_raw, plddt_binder_mean_raw, plddt_binder_min_raw,
plddt_target_mean_raw, pae_interaction_mean, pae_binder_mean,
n_interface_contacts, pae_matrix_path, pdb_path
```

All raw values preserved. Normalized columns clearly labeled `_norm`.

### Text Report
```
═══════════════════════════════════════════
BM2 Evaluator — Campaign Report
═══════════════════════════════════════════

Target: PD-L1 (chain A, 290 residues)
Designs evaluated: 387 from 4 tools
Refolding engines: Boltz2, AF2-Multimer

Tier Distribution:
  consensus_hit:    12  ←  TEST THESE FIRST
  strong:           34
  moderate:         89
  weak:            142
  fail:            110

Source Breakdown:
  BindCraft:         97 designs (5 consensus, 12 strong)
  BoltzGen:         120 designs (4 consensus, 10 strong)
  RFAA+LigandMPNN:   80 designs (2 consensus, 7 strong)
  Complexa:          90 designs (1 consensus, 5 strong)

Top 10 Designs:
Rank  ID          Source       Score   ipSAE_min  ipTM   Tier
  1   bc_042      BindCraft    0.813   0.74       0.82   consensus_hit
  2   bg_117      BoltzGen     0.798   0.71       0.79   consensus_hit
  ...
```

## 2.8 CLI

```bash
# Score designs from any tool
bm2-eval score --designs ./designs/ --target target.pdb --chain A

# Score with specific tool parser
bm2-eval score --designs ./bindcraft_output/ --parser bindcraft --target target.pdb --chain A

# Score with PyRosetta
bm2-eval score --designs ./designs/ --target target.pdb --chain A --rosetta

# Score with custom refolding engines
bm2-eval score --designs ./designs/ --target target.pdb --chain A --engines boltz2,af2

# Generate report only (from existing evaluation)
bm2-eval report --eval-dir ./evaluation_output/

# Compare two evaluation runs
bm2-eval compare --run1 ./eval_bindcraft/ --run2 ./eval_boltzgen/

# Export top designs as FASTA (for gene synthesis)
bm2-eval export --eval-dir ./evaluation_output/ --top 20 --format fasta
```

---

# PART 3: BM2 AGENTS

## 3.1 Package Structure

```
bm2/
├── __init__.py
├── core/
│   ├── models.py             # Campaign, DesignSource, ToolConfig, TargetProfile
│   ├── config.py             # BM2Config (tools registry, paths, GPU settings)
│   └── campaign.py           # Campaign state machine, persistence
├── agents/
│   ├── base.py               # Agent abstract class
│   ├── target_analyst.py     # Structural analysis, difficulty assessment
│   ├── strategy_planner.py   # Tool selection, resource allocation
│   ├── design_runner.py      # Execute tools, collect outputs
│   ├── evaluator_agent.py    # Invoke bm2-evaluator, interpret results
│   ├── wetlab_advisor.py     # Experimental planning
│   ├── maturation_agent.py   # Iterative improvement
│   └── campaign_manager.py   # Orchestrates all agents
├── tools/
│   ├── base.py               # ToolLauncher interface
│   ├── bindcraft.py          # Launch BindCraft in its env
│   ├── boltzgen.py           # Launch BoltzGen in its env
│   ├── mosaic.py             # Launch Mosaic in its env
│   ├── pxdesign.py           # Launch PXDesign
│   ├── rfdiffusion.py        # Launch RFAA + LigandMPNN
│   ├── complexa.py           # Launch Proteina-Complexa
│   └── registry.py           # Available tools registry
├── skills/
│   ├── manager.py            # SkillsManager + query matching
│   ├── builtin/              # 8 built-in skill documents (.md)
│   └── custom/               # User-added skills (.md)
├── cli/
│   └── main.py               # bm2 CLI commands
└── tests/
```

## 3.2 Tool Launcher Interface

BM2 doesn't modify tools. It just knows how to:
1. Prepare input (target PDB + config JSON)
2. Launch the tool in its conda env as a subprocess
3. Parse the output directory

```python
class ToolLauncher(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def conda_env(self) -> str: ...

    @abstractmethod
    def prepare(self, campaign, tool_config) -> dict:
        """Create tool-native config files in the run directory."""

    @abstractmethod
    def launch(self, prepared, run_dir) -> subprocess.Popen:
        """Start the tool as a subprocess in its conda env."""

    @abstractmethod
    def is_complete(self, run_dir) -> bool:
        """Check if the tool has finished."""

    @abstractmethod
    def output_dir(self, run_dir) -> Path:
        """Path to the tool's output directory (fed to evaluator)."""

    def check_installed(self) -> bool:
        """Verify conda env exists and tool is accessible."""
```

## 3.3 Agent Specifications

### Agent 1: Target Analyst

**Trigger:** User provides target PDB
**Output:** TargetProfile with structural analysis and recommendations
**Transition:** INIT → ANALYZING → PLANNING

```python
class TargetAnalyst(Agent):
    """
    Analyzes target structure. Recommends tools and parameters.
    Uses Skills knowledge to make informed recommendations.
    """
    def run(self, campaign):
        target = campaign.target

        # 1. Parse PDB: extract chains, sequences, residue numbering
        self._parse_structure(target)

        # 2. Compute SASA per residue (freesasa or BioPython fallback)
        self._compute_sasa(target)

        # 3. Identify binding sites (user hotspots, or auto from SASA)
        self._identify_sites(target)

        # 4. Assess target difficulty (0-1 score)
        #    Factors: length, disorder fraction, glycosylation, SASA distribution
        target.difficulty = self._assess_difficulty(target)

        # 5. Recommend tools using strategy-selector Skill knowledge
        #    ALWAYS include: BindCraft (proven), BoltzGen (diverse)
        #    Add RFAA for backbone diversity
        #    Add Complexa for hard targets (test-time scaling)
        #    Add Mosaic for custom objectives
        target.recommended_tools = self._recommend_tools(target)

        # 6. Suggest binder length range and modality
        target.suggested_length_range = self._suggest_length(target)
        target.suggested_modality = self._suggest_modality(target)

        # 7. Estimate hotspots if user didn't provide
        if not target.hotspot_residues:
            target.hotspot_residues = self._auto_hotspots(target)

        campaign.transition_to("planning")
```

### Agent 2: Strategy Planner

**Trigger:** Target analysis complete
**Output:** List of ToolConfig objects (which tools, how many designs, what settings)
**Transition:** PLANNING → DESIGNING

```python
class StrategyPlanner(Agent):
    """
    Decides which tools to run and with what parameters.
    Allocates GPU budget across tools.
    Considers what's actually installed.
    """
    def run(self, campaign):
        available = self.registry.list_installed()
        recommended = campaign.target.recommended_tools

        # Intersect recommended with available
        tools_to_run = [t for t in recommended if t in available]

        if not tools_to_run:
            raise RuntimeError(f"No recommended tools installed. Need one of: {recommended}")

        # Allocate designs per tool
        # BindCraft: fewer designs (each is expensive, but high quality)
        # BoltzGen: more designs (faster, diverse)
        # RFAA: medium (backbone diversity)
        # Complexa: depends on search strategy
        allocations = self._allocate(tools_to_run, campaign.target, campaign.gpu_budget)

        for tool_name, num_designs in allocations.items():
            config = ToolConfig(
                tool_name=tool_name,
                num_designs=num_designs,
                target_pdb=campaign.target.pdb_path,
                target_chains=campaign.target.chains,
                hotspot_residues=campaign.target.hotspot_residues,
                binder_length_range=campaign.target.suggested_length_range,
            )
            # Tool-specific settings
            if tool_name == "proteina_complexa":
                config.extra["search_strategy"] = (
                    "beam_search" if campaign.target.difficulty > 0.6
                    else "best_of_n"
                )
            campaign.tool_configs.append(config)

        # Set evaluation engines
        campaign.eval_engines = ["boltz2", "af2"]
        campaign.eval_rosetta = self._check_rosetta_available()

        campaign.transition_to("designing")
```

### Agent 3: Design Runner

**Trigger:** Strategy planned
**Output:** All designs collected in campaign directory
**Transition:** DESIGNING → EVALUATING

```python
class DesignRunner(Agent):
    """
    Launches tools in their conda envs. Monitors progress.
    Handles failures gracefully — if one tool fails, others continue.
    """
    def run(self, campaign):
        campaign_dir = self.config.campaign_dir(campaign.id)

        for tool_config in campaign.tool_configs:
            tool = self.registry.get(tool_config.tool_name)
            run_dir = campaign_dir / "runs" / tool_config.tool_name

            self.logger.info(f"Launching {tool_config.tool_name}: "
                           f"{tool_config.num_designs} designs")
            try:
                prepared = tool.prepare(campaign, tool_config)
                process = tool.launch(prepared, run_dir)
                process.wait()  # Or monitor with timeout

                if process.returncode != 0:
                    self.logger.error(f"{tool_config.tool_name} failed")
                    continue

                self.logger.info(f"{tool_config.tool_name} complete. "
                               f"Output at: {tool.output_dir(run_dir)}")
            except Exception as e:
                self.logger.error(f"{tool_config.tool_name} error: {e}")
                continue  # Keep going with other tools

        campaign.transition_to("evaluating")
```

### Agent 4: Evaluator Agent

**Trigger:** Designs collected
**Output:** Scored, ranked, tiered designs
**Transition:** EVALUATING → RANKED

```python
class EvaluatorAgent(Agent):
    """
    Invokes the standalone bm2-evaluator on all collected designs.
    Interprets results. Generates report.
    """
    def run(self, campaign):
        campaign_dir = self.config.campaign_dir(campaign.id)

        # Collect all output directories
        design_dirs = []
        for tool_config in campaign.tool_configs:
            tool = self.registry.get(tool_config.tool_name)
            run_dir = campaign_dir / "runs" / tool_config.tool_name
            if tool.is_complete(run_dir):
                design_dirs.append({
                    "dir": tool.output_dir(run_dir),
                    "parser": tool_config.tool_name,
                })

        # Call bm2-evaluator as subprocess (it's standalone!)
        eval_dir = campaign_dir / "evaluation"
        cmd = (
            f"bm2-eval score "
            f"--target {campaign.target.pdb_path} "
            f"--chain {campaign.target.chains[0]} "
            f"--output {eval_dir} "
            f"--engines {','.join(campaign.eval_engines)} "
        )
        if campaign.eval_rosetta:
            cmd += "--rosetta "

        for dd in design_dirs:
            cmd += f"--designs {dd['dir']} --parser {dd['parser']} "

        # Run evaluator
        subprocess.run(cmd, shell=True, check=True)

        # Load results
        campaign.load_evaluation_results(eval_dir)

        # Log summary
        report_path = eval_dir / "report.txt"
        if report_path.exists():
            self.logger.info(report_path.read_text())

        campaign.transition_to("ranked")
```

### Agent 5: Wet-Lab Advisor

**Trigger:** User requests wet-lab plan (state = RANKED)
**Output:** Comprehensive experimental plan document
**Transition:** RANKED → WET_LAB_PREP

Uses Skills: `wet-lab-protocols`, `assay-selector`, `gene-synthesis`

```python
class WetLabAdvisor(Agent):
    """
    Generates experimental testing plan based on:
    - Number and quality of designs
    - User's budget and equipment
    - Skills knowledge of assays and protocols
    """
    def run(self, campaign, num_to_test=20, budget_usd=10000):
        top_designs = campaign.get_ranked_designs(top_n=num_to_test)

        plan = []
        plan.append(self._gene_synthesis_section(top_designs))
        plan.append(self._expression_section(top_designs))
        plan.append(self._screening_section(top_designs, budget_usd))
        plan.append(self._characterization_section())
        plan.append(self._controls_section(campaign))
        plan.append(self._design_list_section(top_designs))

        # Save plan
        plan_path = self.config.campaign_dir(campaign.id) / "wetlab_plan.md"
        plan_path.write_text("\n\n".join(plan))

        campaign.transition_to("wet_lab_prep")
```

### Agent 6: Maturation Agent

**Trigger:** Experimental results imported
**Output:** Next-round strategy recommendations
**Transition:** TESTING → MATURING → (back to DESIGNING)

```python
class MaturationAgent(Agent):
    """
    Analyzes experimental results and recommends computational
    improvement strategies for validated binders.
    """
    def run(self, campaign, strategy="auto"):
        # Find designs that bound experimentally
        hits = [d for d in campaign.designs.values()
                if any(r.binds for r in d.experimental_results)]

        if not hits:
            self.logger.warning("No experimental hits. Consider:")
            self.logger.warning("  1. More designs (increase num_designs)")
            self.logger.warning("  2. Different binding site (change hotspots)")
            self.logger.warning("  3. Different tool (add another tool)")
            return campaign

        # Select strategy based on results
        if strategy == "auto":
            best_kd = min((r.kd_nm for d in hits for r in d.experimental_results
                          if r.kd_nm), default=None)
            if best_kd and best_kd > 500:
                strategy = "partial_diffusion"  # Explore nearby backbone space
            elif best_kd and best_kd > 50:
                strategy = "mpnn_redesign"  # Keep backbone, optimize sequence
            else:
                strategy = "mutation_scan"  # Fine-tune individual residues

        # Record maturation round
        round_config = MaturationRound(
            round_number=len(campaign.maturation_rounds) + 1,
            parent_ids=[d.id for d in hits[:5]],
            strategy=strategy,
        )
        campaign.maturation_rounds.append(round_config)
        campaign.transition_to("maturing")
```

---

# PART 4: SKILLS SYSTEM

## 4.1 Architecture

```python
@dataclass
class Skill:
    name: str
    description: str
    keywords: list[str]
    content: str  # Markdown

class SkillsManager:
    def query(self, user_question: str, top_n=3) -> list[Skill]:
        """Find most relevant skills by keyword overlap."""

    def get(self, name: str) -> Skill: ...

    def load_custom(self, directory: Path) -> None:
        """Load .md files with YAML frontmatter from custom/ directory."""
```

## 4.2 The 8 Skills — What Each Must Cover

### Skill 1: `strategy-selector`
When to use each tool. Target type → tool recommendation. GPU requirements. Design count guidance. When to use Complexa's beam_search vs best_of_n. When BindCraft is sufficient alone. When to add BoltzGen for diversity. When RFAA adds value (backbone diversity, β-sheet interfaces). Honest assessment of each tool's strengths and weaknesses.

### Skill 2: `metrics-explainer`
Every metric in the evaluator: pLDDT, ipTM, pTM, bt_ipsae, tb_ipsae, ipsae_min, ipsae_max, PAE, composite scores, multi_model_agreement. Scales and normalization rules. Overath thresholds. Tier definitions. What engine disagreement means. Dunbrack d0_res formula in plain English. Rosetta metrics (dG, dSASA, shape complementarity) when available.

### Skill 3: `wet-lab-protocols`
E. coli pET expression (strain, IPTG, temperature). Mammalian HEK293F. IMAC purification. SEC for monodispersity. Split-luciferase NanoBiT workflow. BLI (Octet) setup. SPR (Biacore) overview. DSF thermal stability. Troubleshooting: no expression, aggregation, weak binding, non-specific binding.

### Skill 4: `assay-selector`
Comparison table: assay type vs throughput vs KD range vs cost vs best-for. Decision trees by budget and design count. When to use split-luciferase (>50 designs) vs BLI (<50). When SPR is worth it (lead characterization). ITC for thermodynamics.

### Skill 5: `gene-synthesis`
Vendor comparison (Twist, IDT, GenScript): cost, turnaround, minimum order. Construct design: His6-TEV-binder for E. coli, signal peptide for mammalian. Codon optimization rules. Cost estimation. Bulk pricing. When to use Gibson cloning vs Golden Gate.

### Skill 6: `maturation-guide`
4 computational strategies with expected improvement: partial diffusion (RFdiffusion, 2-10× improvement), ProteinMPNN redesign (2-5×), interface mutation scanning (identify beneficial point mutations), PPIFlow in silico maturation (can reach pM from nM). 2 experimental: error-prone PCR + display, site-saturation mutagenesis. When to use which.

### Skill 7: `tool-installation`
Installation guides for each tool. General prerequisites (NVIDIA GPU, CUDA, conda). Per-tool commands and common errors. How to verify installation. Evaluator standalone installation.

### Skill 8: `campaign-guide`
How to set up a campaign. Interpreting results by tier distribution. What to do with no hits (change binding site, more designs, add tools, try different modality). How many to test experimentally. When to stop and when to iterate. Reading the evaluation report.

## 4.3 Custom Skills

Users can add domain-specific skills by creating `.md` files in `skills/custom/`:

```yaml
---
name: my-target-class
description: Specific guidance for my target type
keywords: [GPCR, membrane protein, transmembrane]
---
# Content here
For membrane protein targets, consider...
```

---

# PART 5: CLI

```bash
# === Campaign Management ===
bm2 init                                      # Create workspace
bm2 create <name> <pdb> <chain> [--hotspots]  # New campaign
bm2 run <id> [--through ranked|wet_lab]        # Full pipeline
bm2 status [id]                                # Show status
bm2 report <id>                                # Print evaluation report

# === Individual Agent Control ===
bm2 agent analyze <id>                         # Run Target Analyst
bm2 agent plan <id>                            # Run Strategy Planner
bm2 agent design <id>                          # Run Design Runner
bm2 agent evaluate <id>                        # Run Evaluator Agent
bm2 agent wetlab <id> [--num 20] [--budget 10k] # Run Wet-Lab Advisor
bm2 agent mature <id> [--strategy auto]        # Run Maturation Agent

# === Tools ===
bm2 tools list                                 # Show all registered tools
bm2 tools check                                # Verify installations
bm2 tools add <name> <env> <script>            # Register custom tool

# === Skills ===
bm2 skills list                                # List all skills
bm2 skills query "your question here"          # Ask the skills system
bm2 skills show <skill-name>                   # Show full skill content

# === Export ===
bm2 export <id> --format fasta|csv|json        # Export designs
bm2 export <id> --top 20 --format fasta        # Top 20 as FASTA

# === Import ===
bm2 import results <id> <csv>                  # Import experimental data
```

---

# PART 6: IMPLEMENTATION ORDER FOR CLAUDE CODE

## Step 0: Research (MANDATORY — NO CODE)

Deploy research agents:
1. Read BindCraft repo output format (final_designs/, scores.csv columns, filter JSON)
2. Read BoltzGen repo output format (DSL, ranking CSV, PDB naming)
3. Read Mosaic repo output format
4. Read PXDesign paper for output format and filter methodology
5. Read RFdiffusion output format + LigandMPNN output
6. Read Proteina-Complexa CLI and eval pipeline output
7. Read Dunbrack IPSAE repo — verify d0_res implementation line by line
8. Read Overath paper — extract exact thresholds and composite formulas
9. Read BindCraft Nature paper Methods — exact loss terms, filter thresholds
10. Verify: what does Boltz2 `--save_pae` actually produce? File format?
11. Verify: what does ColabFold produce for PAE? JSON structure?
12. Test: can you call bm2-evaluator from outside any tool env?

**Present ALL findings with specific file paths, column names, metric scales. Wait for approval.**

## Step 1: Evaluator Core
Ingestion layer (all 6 tool parsers) + metrics (ipSAE validated against Dunbrack) + PAE handling + pLDDT normalization + tests

## Step 2: Evaluator Refolding
Boltz2 runner + AF2 runner + monomer validation + PAE matrix saving + tests

## Step 3: Evaluator Scoring + Reporting
Composite scores + tier classification + ranking + diversity clustering + CSV export + text report + CLI

## Step 4: BM2 Campaign Core
Campaign model + state machine + TargetProfile + ToolConfig + persistence

## Step 5: Tool Launchers
ToolLauncher interface + 6 implementations (BindCraft, BoltzGen, Mosaic, PXDesign, RFAA, Complexa) + registry

## Step 6: Agents
6 agents + CampaignManager + integration with evaluator and tool launchers

## Step 7: Skills
SkillsManager + 8 built-in skills + custom skill loading + query interface

## Step 8: CLI + Integration
bm2 CLI commands + end-to-end test + documentation

---

# PART 7: CONSISTENCY RULES (NON-NEGOTIABLE)

1. **Raw data never modified.** Tool outputs read as-is. Transformations in documented functions.
2. **Every metric cites its source.** Code comments include paper reference and equation number.
3. **Every normalization is explicit.** Function named `normalize_X()`, stores both raw and normalized.
4. **PAE matrices always saved as .npy.** Path recorded in evaluation record.
5. **No threshold without citation.** The 0.61 comes from Overath. The d0_res comes from Dunbrack. Document it.
6. **CSV exports include raw AND processed columns.** Users can always trace back.
7. **Evaluator works standalone.** `pip install bm2-evaluator && bm2-eval score` must work without BM2 agents.
8. **Tools are never modified.** BM2 reads their output. If a tool changes its format, only the ingestor changes.
9. **Every metric computation has a unit test.** Known input → known output, validated against reference implementations.
10. **Config is self-documenting.** Every default value has a comment explaining why.
