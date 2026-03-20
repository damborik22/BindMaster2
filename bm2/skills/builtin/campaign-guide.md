---
name: campaign-guide
description: How to run a BM2 campaign from start to finish
keywords: [campaign, run, start, begin, how, guide, tutorial, workflow, pipeline, step]
---
# Running a BM2 Campaign

## Quick Start
```bash
bm2 create my_target target.pdb A --hotspots A10,A25,A42
bm2 run my_target_20260320 --through ranked
bm2 report my_target_20260320
bm2 agent wetlab my_target_20260320 --num 20
```

## Step by Step

### 1. Prepare Target
- High-quality PDB (experimental or AlphaFold)
- Identify chain(s) to bind
- Optional: specify hotspot residues

### 2. Create Campaign
```bash
bm2 create <name> <pdb> <chain> [--hotspots A10,A25]
```

### 3. Run Pipeline
```bash
bm2 run <id>                     # Through RANKED (default)
bm2 run <id> --through wet_lab   # Also generate wet-lab plan
```

Or individually:
```bash
bm2 agent analyze <id>
bm2 agent plan <id>
bm2 agent design <id>
bm2 agent evaluate <id>
bm2 agent wetlab <id>
```

### 4. Interpret Results

| Outcome | Action |
|---------|--------|
| >= 5 consensus_hits | Good campaign, test top 20 |
| 1-4 consensus_hits | Test top 10-15 |
| 0 consensus, 10+ strong | Test top 20, consider maturation |
| Few strong | More designs, different site, add tools |

### 5. After Wet-Lab Results
```bash
bm2 import results <id> experimental_data.csv
bm2 agent mature <id>
```

### 6. What If No Hits?
1. More designs (increase num_designs)
2. Different binding site (change hotspots)
3. Add tools (BoltzGen, RFAA, Complexa)
4. Try peptide mode
5. Complexa with beam_search/MCTS
