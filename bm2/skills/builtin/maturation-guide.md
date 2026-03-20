---
name: maturation-guide
description: Computational and experimental strategies to improve binder affinity
keywords: [maturation, improve, affinity, optimize, mutation, evolution, KD, stronger, better]
---
# Binder Maturation Guide

## Computational Strategies

### 1. ProteinMPNN Redesign (fastest, 2-5x improvement)
- Keep backbone fixed, redesign surface/core sequence
- Fix interface residues (within 4 A of target)
- Generate 20-100 variants per backbone
- Good for: improving expression/stability without changing interface

### 2. RFdiffusion Partial Diffusion (2-10x improvement)
- Add partial noise to backbone (partial_T = 5-20)
- Generates structurally similar but varied backbones
- Run LigandMPNN for new sequences on each
- Good for: exploring nearby backbone space

### 3. BindCraft/Mosaic Warm Start (2-5x improvement)
- Initialize from hit sequence instead of random
- Optimizer refines rather than explores
- Good for: fine-tuning moderate binders (100 nM - 1 uM)

### 4. Interface Mutation Scanning
- Score all 19 substitutions at each interface residue
- Use Boltz2 ddG or FoldX
- Combine beneficial mutations (check epistasis)

## Experimental Strategies

| Strategy | Improvement | Time |
|----------|------------|------|
| MPNN redesign | 2-5x | 1 day compute |
| Partial diffusion | 2-10x | 1-3 days |
| Error-prone PCR + display | 10-100x | 4-8 weeks |
| Site-saturation mutagenesis | 5-50x | 2-4 weeks |

## When to Mature
- KD > 100 nM and want < 10 nM
- Poor stability (Tm < 50C)
- Aggregation issues
- Low expression yield
