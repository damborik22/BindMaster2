---
name: metrics-explainer
description: Understanding BM2 evaluation metrics
keywords: [metric, score, ipsae, iptm, plddt, pae, composite, tier, consensus, explain, meaning, threshold]
---
# Understanding BM2 Metrics

## Primary Ranking Metric: ipSAE_min

**What:** How confidently the structure predictor places binder interface residues
relative to the target. The "weakest link" -- min(binder->target, target->binder).

**Why it's best:** Overath 2025 meta-analysis (3,766 designs, 15 targets) showed
ipSAE_min is the single best predictor of experimental binding (1.4x better AP than ipAE).

**Scale:** 0-1. Higher = more confident. Threshold >0.61 (BM2 default from Overath).

**Formula (Dunbrack 2025, Eq 14-16):**
d0 = max(0.5, 1.24 * (N_qualifying - 15)^(1/3) - 1.8)
score = 1 / (1 + (PAE / d0)^2) per residue pair
ipSAE = max over source residues of mean scores

## Confidence Metrics

| Metric | Scale | Direction | Threshold | Notes |
|--------|-------|-----------|-----------|-------|
| ipSAE_min | 0-1 | Higher | >0.61 | Best single predictor |
| ipTM | 0-1 | Higher | >0.6 | Can be inflated by hallucination |
| pLDDT | 0-1 (norm) | Higher | >0.7 | Per-residue confidence |
| PAE | 0-32 A | Lower | <10 A | Raw error estimate |

## PyRosetta Metrics

| Metric | Units | Direction | Threshold | Source |
|--------|-------|-----------|-----------|--------|
| dG | REU | Lower | <=-10 | BindCraft |
| dSASA | A^2 | Higher | >=800 | BindCraft |
| Shape comp. | 0-1 | Higher | >=0.55 | BindCraft |
| Best composite | - | Higher | - | ipSAE_min * abs(dG/dSASA) (Overath 2025) |

## Quality Tiers

| Tier | Definition | Action |
|------|-----------|--------|
| consensus_hit | ALL engines ipSAE_min > 0.61 | Test first |
| strong | ONE > 0.61, ALL > 0.40 | Test second |
| moderate | Best ipSAE > 0.40, best ipTM > 0.6 | Consider testing |
| weak | Basic filters pass | Low priority |
| fail | Nothing passes | Skip |

## Multi-Model Agreement

Fraction of engines where ipSAE_min > threshold. Agreement = 1.0 is the strongest filter.

## Monomer Validation

Refold binder alone. RMSD <= 3.0 A vs complex = folds independently (good).
RMSD > 3.0 A = target-dependent folder (concerning).
