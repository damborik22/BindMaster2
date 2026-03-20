---
name: wet-lab-protocols
description: Expression, purification, and characterization protocols
keywords: [expression, purification, ecoli, BLI, SPR, protocol, IPTG, NiNTA, SEC, assay, experiment, lab]
---
# Wet-Lab Protocols for Designed Binders

## E. coli Expression

### Construct: His6-TEV-binder in pET21b/pET28a
- Strain: BL21(DE3) or SHuffle T7 (for disulfide bonds)
- Induce OD600 ~0.6 with 0.5 mM IPTG
- Express 18C overnight (soluble) or 37C 4h
- Lyse by sonication, clarify 20,000g 30min 4C

### IMAC Purification
- Ni-NTA column, wash 20 CV (20 mM imidazole), elute (250 mM imidazole)
- Expected: 1-50 mg/L culture

### SEC (optional)
- Superdex 75 for 5-75 kDa binders
- Check monodispersity (single peak at expected MW)

## Binding Assays

### BLI (Octet) -- Medium Throughput
- 8-16 samples simultaneously, KD range 1 nM - 100 uM
- Screen: single conc (1 uM), response > 0.1 nm = hit
- Kinetics: serial dilution, global fit for kon/koff/KD

### SPR (Biacore) -- Gold Standard
- KD range pM - uM, publication quality
- Use for lead characterization after BLI screen

### Split-Luciferase (NanoBiT) -- High Throughput
- 96-384 well, no purification needed
- Good for > 50 designs, semi-quantitative

## Characterization
- DSF: Tm > 50C for therapeutic, > 40C for research
- SEC-MALS: monodispersity check
- CD: secondary structure confirmation

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No expression | Lower IPTG, auto-induction, codon optimize |
| Insoluble | 18C expression, SUMO/MBP tag |
| Aggregation | Surface mutations, check AggreProt |
| No binding | Test more designs, try different site |
| Weak binding | Maturation round |
