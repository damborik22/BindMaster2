# Target Misfolding in Cross-Model Evaluation

## Problem

Some targets misfold when predicted from sequence alone during cross-model
refolding. This corrupts all interface metrics (ipSAE, ipTM, PAE) because
the binder is evaluated against an incorrect target structure.

Example: CALCA (calcitonin gene-related peptide) — Boltz-2 predicts a
28 Å RMSD structure vs the experimental PDB. AF2 shows similar issues.

## Impact

- ipSAE scores are unreliable for misfolding targets
- Cross-model "agreement" may reflect both models being wrong the same way
- Designs that appear to fail evaluation may actually be good binders

## Solution: Template-Constrained Refolding

When `target_pdb` is provided to the Boltz2 engine, the target backbone is
constrained via `force_template=True`. The binder is still predicted de novo
(no template), so the evaluation tests whether the binder can independently
fold and bind to the known target structure.

```python
# In BM2 evaluator config or CLI:
boltz2_engine = Boltz2Engine(target_pdb="/path/to/target.pdb")
```

## When to Use

- Always provide target_pdb when available
- Essential for: small peptides, disordered proteins, novel folds
- For well-characterized targets (PDL1, PD1, etc.), sequence-only mode is usually fine

## Detection

Compare target pLDDT and PAE between refolding runs with and without template.
If target pLDDT drops significantly or target-target PAE is high (>10 Å),
the target is likely misfolding.
