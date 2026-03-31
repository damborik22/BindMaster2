# Proteina-Complexa Mosaic Integration Design

**Date:** 2026-03-31
**Status:** Approved

---

## Goal

Add Proteina-Complexa flow matching as a second design engine in BM2's loss tuning system, running through Mosaic's JAX wrapper with beam search + inverse folding. Output compatible with existing evaluator pipeline.

## Constraints

- Same `designs.csv` output format as Boltz-2 hallucination
- Separate tuning track (independent grid search, no shared loss state with Boltz-2)
- No ProteinMPNN in Proteina pipeline (PC generates sequences end-to-end via decoder)
- Runs in existing Mosaic UV venv (jproteina-complexa already installed)
- Inner evaluator and BM2 ingestor work unchanged

---

## Architecture

Two independent design engines, shared scoring and evaluation:

```
Boltz-2 path:     design_template.py      → simplex_APGM optimization → designs.csv
Proteina path:    design_template_proteina.py → beam search + inverse folding → designs.csv
                                                          ↓
                                               Same inner_evaluator.py
                                               Same loss_tuner.py (--engine flag)
                                               Same bm2-eval pipeline
```

## Section A: design_template_proteina.py

New file at `agent/design_template_proteina.py`. Follows the same pattern as `design_template.py` — all tunables as named constants at the top, same output format.

### Constants

```python
# TARGET PARAMETERS
TARGET_PDB = ""
TARGET_SEQUENCE = ""
TARGET_CHAIN_ID = "A"
HOTSPOT_RESIDUES = []

# DESIGN STRATEGY
N_DESIGNS = 50
TOP_K = 10
INVERSE_FOLD_SAMPLES = 5
MIN_LENGTH = 80
MAX_LENGTH = 80
LENGTH_STEP = 5

# BEAM SEARCH
BEAM_WIDTH = 4
N_BRANCH = 4
STEP_CHECKPOINTS = [0, 100, 200, 300, 400]
N_STEPS = 400
SEED = 42

# LOSS WEIGHTS (beam scoring — no MPNN)
WEIGHT_IPTM = 1.0
WEIGHT_BINDER_TARGET_IPSAE = 0.5
WEIGHT_TARGET_BINDER_IPSAE = 0.5
WEIGHT_BINDER_TARGET_CONTACT = 0.0
WEIGHT_PLDDT = 0.0
WEIGHT_WITHIN_BINDER_PAE = 0.0
```

### Pipeline

1. Load denoiser + decoder from jproteina-complexa (`load_denoiser()`, `load_decoder()`)
2. Load target from PDB via `load_target_cond(chain, hotspots)`
3. Build loss function from `WEIGHT_*` constants using Mosaic's composable `LossTerm` API
4. For each binder_length in range:
   a. Create mask: `jnp.ones(binder_length, dtype=jnp.bool_)`
   b. Run `beam_search(model, decoder, loss_fn, mask, key, target, step_checkpoints, beam_width, n_branch)`
   c. Sort designs by loss, take top-K backbones
5. For each top-K backbone: inverse fold (resample latents `INVERSE_FOLD_SAMPLES` times, decode new sequences)
6. Score all variants with the same loss function
7. Write `designs.csv` with same columns as Boltz-2 template
8. Write PDB structures for top designs

### Output Format

Same 35-column `designs.csv` as Boltz-2 template. Columns that don't apply to Proteina (e.g., `ranking_loss` from Boltz-2 optimization) are written as the beam search loss value. Aux metrics (`bt_ipsae`, `tb_ipsae`, `ipsae_min`, `iptm`, `plddt_binder_mean`, etc.) come from the Mosaic loss term aux dict — same source as Boltz-2.

### Inverse Folding

For each top-K backbone from beam search:
1. Fix backbone coordinates (`bb` from `ScoredDesign`)
2. Resample latents N times with different random keys
3. Decode each latent → new `DecoderOutput` with different `aatype`
4. Score each variant with the loss function
5. Keep best sequence for each backbone

This uses jproteina-complexa's decoder directly — no external inverse folding tool needed.

---

## Section B: Loss Tuner Changes

### New `--engine` flag

`loss_tuner.py` gets `--engine {boltz2,proteina}` (default: `boltz2`).

When `proteina`:
- Base template: `agent/design_template_proteina.py`
- Grid search: `PROTEINA_GRID_SEARCH` (separate predefined experiments)
- All other machinery unchanged (read constants, edit, run, score, commit/revert)

### Proteina Grid Search

```python
PROTEINA_GRID_SEARCH = [
    ("Baseline: ipTM + ipSAE scoring", {}),
    ("Wider beam: BEAM_WIDTH 4 → 8", {"BEAM_WIDTH": 8}),
    ("More branching: N_BRANCH 4 → 8", {"N_BRANCH": 8}),
    ("Larger binder: 120 aa", {"MIN_LENGTH": 120, "MAX_LENGTH": 120}),
    ("Much larger binder: 200 aa", {"MIN_LENGTH": 200, "MAX_LENGTH": 200}),
    ("Add contact loss", {"WEIGHT_BINDER_TARGET_CONTACT": 1.0}),
    ("Add pLDDT scoring", {"WEIGHT_PLDDT": 0.3}),
    ("Add binder PAE scoring", {"WEIGHT_WITHIN_BINDER_PAE": 0.4}),
    ("More inverse folding: 10 per backbone", {"INVERSE_FOLD_SAMPLES": 10}),
    ("Fewer checkpoints: [0, 200, 400]", {"STEP_CHECKPOINTS": [0, 200, 400]}),
]
```

---

## Section C: Orchestrator Changes

`orchestrator.py` gets `--engine {boltz2,proteina}` flag:

- Phase 2 (tuning): passes `--engine` to `loss_tuner.py`
- Phase 3 (production): copies the correct template and runs it at scale
- Phase 1 (analysis): unchanged

---

## Section D: Mosaic Launcher Changes

`bm2/tools/mosaic.py` adds `engine` to `extra_settings`:

- `engine="boltz2"` (default): templates `hallucinate_bindmaster.py`
- `engine="proteina"`: templates `hallucinate_proteina.py`

Template search adds second path:
```python
_PROTEINA_SEARCH_PATHS = [
    "examples/bindmaster_examples/hallucinate_proteina.py",
]
```

Parameter injection unchanged — same `_inject_parameters()` pattern, different constant names for Proteina-specific params (beam_width, n_branch, etc.).

---

## Section E: Strategy Document

Add Proteina-specific section to `agent/strategy.md`:

### Proteina Beam Search Guidance

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| BEAM_WIDTH | 4 | 1-8 | More beams = more candidates explored, slower |
| N_BRANCH | 4 | 1-8 | More branches = wider tree, more diverse |
| STEP_CHECKPOINTS | [0,100,200,300,400] | 2-10 intervals | More = finer beam pruning, slower |
| INVERSE_FOLD_SAMPLES | 5 | 1-20 | More = better chance of good sequence per backbone |

### When to Use Proteina vs Boltz-2

- **Proteina beam search:** Fast iteration (~1-3s/design), good for length/loss exploration. Flow matching generates diverse backbones. Best for initial exploration.
- **Boltz-2 hallucination:** Gradient-based, slower (~2min/design), but tighter control over loss optimization. ProteinMPNN sequence recovery improves expressibility. Best for production after finding good parameters.

---

## Files Changed Summary

| File | Change | New/Modify |
|------|--------|------------|
| `agent/design_template_proteina.py` | Proteina beam search + inverse folding | New (~400 LOC) |
| `agent/loss_tuner.py` | `--engine` flag, `PROTEINA_GRID_SEARCH` | Modify |
| `agent/orchestrator.py` | `--engine` flag routing | Modify |
| `agent/strategy.md` | Proteina tuning guidance | Modify |
| `bm2/tools/mosaic.py` | `engine="proteina"` template support | Modify |

**Total: 1 new file, 4 modified files.**
