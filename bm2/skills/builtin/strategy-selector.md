---
name: strategy-selector
description: Which design tool to use for your target
keywords: [tool, strategy, recommend, which, choose, select, best, pick, target, difficult, easy]
---
# Choosing the Right Design Tool

## Quick Decision Guide

**Start with BindCraft** unless you have a specific reason not to. It's the most validated
open-source tool with 10-100% success rates across 12 diverse targets (Nature 2024).

**Add BoltzGen** for diversity. Different generative approach (diffusion vs hallucination)
finds different solutions. Needs more designs (5,000-60,000) but each is faster.

**Add RFAA + LigandMPNN** for backbone diversity. Diffusion generates backbones from scratch,
then LigandMPNN designs sequences.

**Add Proteina-Complexa** for hard targets. Test-time compute scaling (beam search, MCTS)
can push through where other methods fail.

## When to Use Each Tool

### BindCraft
- **Best for:** General minibinder design, first attempt at any target
- **Strengths:** Highest validated success rates, AF2 hallucination + MPNNsol + filters
- **Weaknesses:** Slow (~10,000 attempts for ~100 final designs), alpha-helix biased
- **Designs needed:** 100-300 trajectories -> ~10-50 final filtered designs
- **GPU time:** ~1-4 hours per 100 trajectories on A100

### BoltzGen
- **Best for:** Volume generation, diverse scaffold exploration
- **Strengths:** All-atom diffusion, handles proteins/peptides/nanobodies/small molecules
- **Designs needed:** 5,000-60,000 -> budget of 10-50 final designs
- **GPU time:** ~30-60 sec per design on A100

### RFAA + LigandMPNN
- **Best for:** Backbone diversity, beta-sheet interfaces, scaffold control
- **Strengths:** Can specify topology, handles DNA/RNA/small-molecule targets
- **Designs needed:** 1,000-10,000 backbones -> filter to top 50

### PXDesign
- **Best for:** Protenix-based diversity, dual-filter validation
- **Strengths:** Protenix hallucination + diffusion, strong dual-filter methodology

### Mosaic
- **Best for:** Custom objectives, multi-model optimization, research
- **Strengths:** Compose losses from AF2 + Boltz2 + Protenix + MPNN

### Proteina-Complexa
- **Best for:** Hard targets, test-time compute scaling
- **Strengths:** Flow-matching + beam search/MCTS, 63.5% hit rate across 133 targets

## Target Difficulty and Tool Selection

| Difficulty | Indicators | Recommended Tools |
|-----------|-----------|-------------------|
| Easy (<0.3) | Well-folded, clear pocket, <300 res | BindCraft alone |
| Medium (0.3-0.6) | Larger target, shallow pocket | BindCraft + BoltzGen |
| Hard (0.6-0.8) | Flat surface, disorder, multi-domain | BindCraft + BoltzGen + RFAA + Complexa |
| Very hard (>0.8) | Glycosylated, membrane-proximal | All tools, Complexa with MCTS |

## How Many Designs to Test

From Overath 2025 meta-analysis: overall experimental success rate ~11.6%.
- To get ~2 hits: test 20 designs (consensus_hit + strong tiers)
- To get ~5 hits: test 50 designs
- Test consensus_hits FIRST -- cross-model agreement is the best predictor
