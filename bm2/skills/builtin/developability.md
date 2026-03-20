---
name: developability
description: Assessing whether a designed binder will work in practice
keywords: [developability, solubility, aggregation, stability, expression, manufacture, practical]
---
# Binder Developability Assessment

## Beyond Binding: Will It Work?

A computationally strong design can fail due to:
- Poor expression in E. coli
- Aggregation during purification
- Low thermal stability
- Immunogenicity (therapeutic applications)

## Computational Checks

### Solubility: SoluProt (Loschmidt Labs)
- Sequence-based E. coli solubility prediction
- Score > 0.5 = likely soluble
- Web: loschmidt.chemi.muni.cz/soluprot/

### Aggregation: AggreProt (Loschmidt Labs)
- Per-residue aggregation propensity
- Flag exposed aggregation-prone patches
- Web: loschmidt.chemi.muni.cz/aggreprot/

### Thermostability: FireProt 2.0 (Loschmidt Labs)
- Suggests stabilizing mutations (EXCLUDE interface residues)
- Combines FoldX + Rosetta + ancestral reconstruction
- Can improve Tm by 5-20C

### Sequence Checks
- Surface hydrophobicity: not too hydrophobic
- Free cysteines: cause aggregation (most tools exclude Cys)
- Unusual AA composition: may indicate design artifacts

## Pre-Order Checklist

1. Consensus hit or strong tier
2. Monomer validation passes (RMSD <= 3.0 A)
3. No sequence pathologies
4. SoluProt > 0.5 (if checked)
5. Diverse scaffolds selected (not 20 variants of same fold)
