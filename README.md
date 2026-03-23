# BindMaster 2.0

Multi-tool computational protein binder design platform developed at [Loschmidt Laboratories](https://loschmidt.chemi.muni.cz/), Masaryk University Brno.

BindMaster 2.0 replaces the deterministic decision layer of [BindMaster 1](https://github.com/damborik22/BindMaster) with agentic reasoning while keeping all infrastructure (tool launchers, evaluator, persistence) intact. It orchestrates six external design tools, evaluates designs through cross-model validation (Boltz2 + AF2), and guides the full campaign lifecycle from target analysis to wet-lab preparation.

## Architecture

```
bm2/                         Orchestration platform
  agents/                    6 agents: target analyst, strategy planner,
                             design runner, evaluator, wet-lab advisor,
                             maturation agent
  tools/                     Tool launchers (BindCraft, BoltzGen, Mosaic,
                             RFdiffusion, PXDesign, Proteina-Complexa)
  skills/                    Domain knowledge documents
  cli/                       CLI entry point (bm2)

bm2_evaluator/               Standalone evaluator package
  ingestion/                 7 tool-specific output parsers
  refolding/                 Boltz2 + AF2 worker subprocess engines
  metrics/                   ipSAE (Dunbrack d0_res), PAE, pLDDT
  scoring/                   Composite scores, 5-tier classification
  reporting/                 CSV, text reports, cross-tool comparison
```

## Design Tools

| Tool | Type | Environment | Reference |
|------|------|-------------|-----------|
| [BindCraft](https://github.com/martinpacesa/BindCraft) | AF2-based hallucination | conda: BindCraft | Pacesa et al., Nature 2024 |
| [BoltzGen](https://github.com/jwohlwend/boltzgen) | Boltz2 flow matching | conda: BoltzGen | Wohlwend et al., 2025 |
| [Mosaic](https://github.com/escalante-bio/mosaic) | Boltz2 hallucination | uv venv | Escalante Bio |
| [RFdiffusion](https://github.com/baker-laboratory/rf_diffusion_all_atom) | All-atom diffusion | conda: bindmaster_rfaa | Watson et al., Nature 2023 |
| [PXDesign](https://github.com/bytedance/PXDesign) | Protenix + AF2-IG | conda: bindmaster_pxdesign | ByteDance, 2025 |
| [Proteina-Complexa](https://github.com/NVIDIA-Digital-Bio/proteina-complexa) | Flow matching + test-time compute | uv venv | NVIDIA, 2025 |

## Evaluation Pipeline

Every design passes through cross-model refolding:

1. **Ingestion** -- Parse each tool's native output format, extract sequences and structures
2. **Refolding** -- Predict complex structure independently with Boltz2 and AF2
3. **Scoring** -- Compute ipSAE (Dunbrack d0_res variant), ipTM, pLDDT, PAE from refolded structures
4. **Tier classification** -- Rank designs into 5 tiers based on cross-model consensus:
   - **consensus_hit**: All engines ipSAE > 0.61
   - **strong**: At least one > 0.61, all > 0.40
   - **moderate**: Best ipSAE > 0.40 and best ipTM > 0.6
   - **weak**: Best pLDDT > 0.70 and best ipTM > 0.5
   - **fail**: Nothing passes

Native tool metrics are preserved alongside BM2's independent metrics for comparison.

## Quick Start

### Install

```bash
# BM2 platform
pip install -e .

# External tools (via BindMaster 1 installer)
bash ~/BindMaster/install/install.sh --tool all
```

### Run a Campaign

```bash
# Create campaign
bm2 create my_campaign target.pdb A --hotspots "A37,A39,A49"

# Run full pipeline: analyze -> plan -> design -> evaluate -> rank
bm2 run <campaign_id> --through ranked --total-designs 50

# Check status
bm2 status <campaign_id>

# Export top designs
bm2 export designs <campaign_id> --top 10 --format fasta
```

### Standalone Evaluator

```bash
# Score designs from any tool
bm2-eval score --designs ./my_designs/ --target target.pdb --chain A --engines boltz2,af2

# Generate report
bm2-eval report --eval-dir ./evaluation_output/

# Compare two runs
bm2-eval compare --run1 ./eval_v1/ --run2 ./eval_v2/
```

## Campaign Pipeline

```
INIT -> ANALYZING -> PLANNING -> DESIGNING -> EVALUATING -> RANKED
                                                             |
                                              WET_LAB_PREP -> TESTING -> MATURING
                                                                           |
                                                                    (loop to DESIGNING)
```

Each stage is handled by a dedicated agent:

| Agent | Role |
|-------|------|
| **Target Analyst** | Parse PDB, compute SASA, detect hotspots, assess difficulty |
| **Strategy Planner** | Allocate designs across tools based on target properties |
| **Design Runner** | Launch tools sequentially, monitor completion |
| **Evaluator** | Run cross-model refolding and scoring |
| **Wet-Lab Advisor** | Generate experimental protocols and gene FASTA |
| **Maturation Agent** | Plan improvement rounds (partial diffusion, MPNN redesign) |

## Key Metrics

- **ipSAE** (interaction predicted Structural Alignment Error): Primary ranking metric. Dunbrack d0_res variant, threshold >= 0.61 for hits.
- **ipTM** (interaction predicted TM-score): >= 0.6 is good.
- **pLDDT** (predicted Local Distance Difference Test): >= 0.85 is well-folded. Boltz2 uses 0-1 scale, AF2 uses 0-100 -- the evaluator normalizes automatically.
- **Cross-model agreement**: Fraction of engines with ipSAE above threshold. High agreement = reliable prediction.

## Requirements

- Python >= 3.10
- NVIDIA GPU >= 16 GB VRAM (RTX 3090+ recommended)
- CUDA 11.8+
- conda or mamba (for tool environments)
- uv (for Mosaic and Complexa environments)
- ~100 GB disk for all tools + model weights

## Project Structure

```
~/BindMaster2/              This repository
~/BindMaster/               Tool installations (BindMaster 1)
  BindCraft/                conda: BindCraft
  BoltzGen/                 conda: BoltzGen
  Mosaic/                   uv venv
  rf_diffusion_all_atom/    conda: bindmaster_rfaa
  PXDesign/                 conda: bindmaster_pxdesign
  Proteina-Complexa/        uv venv
~/.bm2/                     Campaign state and config
  config.toml               Global configuration
  campaigns/                Campaign directories
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

265 tests covering: ipSAE computation, PAE handling, pLDDT normalization, ingestion parsers, scoring pipeline, campaign state machine, agents, CLI, and skills system.

## License

Loschmidt Laboratories, Masaryk University Brno.

## References

- ipSAE: Dunbrack et al., d0_res variant (2025)
- Composite scoring: Overath et al. (2025)
- Tier thresholds: Calibrated from Overath Table 2, validated against lab data
