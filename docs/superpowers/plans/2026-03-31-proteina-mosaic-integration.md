# Proteina-Complexa Mosaic Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Proteina-Complexa flow matching as a second design engine (beam search + inverse folding) in BM2's loss tuning system, with same `designs.csv` output format.

**Architecture:** New `design_template_proteina.py` uses Mosaic's `beam_search()` wrapper for jproteina-complexa. Loss tuner gets `--engine` flag to select between Boltz-2 and Proteina grid searches. Mosaic launcher supports `engine="proteina"` for BM2 tool pipeline.

**Tech Stack:** jproteina-complexa, mosaic.models.proteina (beam_search), mosaic.losses.structure_prediction, JAX, gemmi

**Spec:** `docs/superpowers/specs/2026-03-31-proteina-mosaic-integration-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `agent/design_template_proteina.py` | Proteina beam search + inverse folding design script | New (~350 LOC) |
| `agent/loss_tuner.py` | Karpathy loop with engine selection | Modify — add `--engine`, `PROTEINA_GRID_SEARCH` |
| `agent/orchestrator.py` | Pipeline orchestrator | Modify — add `--engine` routing |
| `agent/strategy.md` | Tuning knowledge | Modify — add Proteina section |
| `bm2/tools/mosaic.py` | BM2 tool launcher | Modify — add proteina template support |

---

### Task 1: Create design_template_proteina.py

**Files:**
- Create: `agent/design_template_proteina.py`

This is the core deliverable — a self-contained Proteina beam search + inverse folding script with the same output format as the Boltz-2 template.

- [ ] **Step 1: Create the template**

Create `agent/design_template_proteina.py`. The script must:
- Have all tunables as constants at top (matching BM2_AGENT_INSTRUCTIONS pattern)
- Use `mosaic.models.proteina.beam_search()` for generation
- Use Mosaic's composable `LossTerm` for beam scoring
- Include inverse folding step (resample latents on top-K backbones)
- Write `designs.csv` in same 35-column format as Boltz-2 template
- Write PDB structures and PAE arrays for top designs

Key implementation details from the Mosaic wrapper:
- `beam_search()` returns `list[ScoredDesign]` with `.sequence` (mosaic token order), `.loss`, `.aux`, `.bb` (CA coords in Angstroms), `.lat` (latents)
- Binder length is controlled by `mask = jnp.ones(length, dtype=jnp.bool_)`
- Decoder output sequence is in jproteina-complexa AA order — use `_JPC_TO_MOSAIC` permutation for mosaic token ordering
- Loss function signature: `loss_fn(seq_hard_one_hot [N, 20], *, key) -> (scalar, dict)`
- For inverse folding: fix backbone (`bb_ca` in Angstroms), reinitialize latent noise, run denoiser with `bb` clamped, decode new sequence
- Coordinates: public API uses Angstroms, internal flow matching uses nanometers (the wrapper handles this)
- Target loading: `jproteina_complexa.pdb.load_target_cond(chain: gemmi.Chain, hotspots: list[int] | None)`

The loss function for beam search scoring should be built from the `WEIGHT_*` constants:
```python
loss_fn = folder.build_multisample_loss(
    loss=(
        WEIGHT_IPTM * sp.IPTMLoss()
        + WEIGHT_BINDER_TARGET_IPSAE * sp.BinderTargetIPSAE()
        + WEIGHT_TARGET_BINDER_IPSAE * sp.TargetBinderIPSAE()
        + (WEIGHT_BINDER_TARGET_CONTACT * sp.BinderTargetContact() if WEIGHT_BINDER_TARGET_CONTACT > 0 else 0)
        + (WEIGHT_PLDDT * sp.PLDDTLoss() if WEIGHT_PLDDT > 0 else 0)
        + (WEIGHT_WITHIN_BINDER_PAE * sp.WithinBinderPAE() if WEIGHT_WITHIN_BINDER_PAE > 0 else 0)
    ),
    features=features,
    recycling_steps=1,
    num_samples=1,
)
```

Note: The loss_fn for beam search needs Boltz-2 features for evaluation. Use `Boltz2().binder_features()` to create features, then `build_multisample_loss()` to compile the scorer. The beam search uses this scorer to evaluate decoded sequences from Proteina's decoder.

For the `designs.csv` output, the Stage 2 metrics (bt_ipsae, tb_ipsae, ipsae_min, etc.) come from the loss function's aux dict — same mechanism as the Boltz-2 template. For designs that only have beam search scoring (not top-K), use the beam loss aux values.

The inverse folding function should follow the exact pattern from `Mosaic/examples/proteina.py` lines 456-537:
1. Fix backbone in nanometers (`bb_ca / 10.0`)
2. Initialize fresh latent noise
3. Run `fori_loop` over denoising steps (backbone clamped, only latents evolve)
4. Decode final latents → new sequence

For each top-K backbone, run inverse folding `INVERSE_FOLD_SAMPLES` times with different keys, score each, keep the best.

- [ ] **Step 2: Verify imports work**

Run: `cd /home/david/BindMaster2 && /home/david/BindMaster/Mosaic/.venv/bin/python -c "from mosaic.models.proteina import beam_search, ScoredDesign; from jproteina_complexa.hub import load_denoiser, load_decoder; from jproteina_complexa.pdb import load_target_cond; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 3: Validate constant parsing**

Run: `cd /home/david/BindMaster2 && .venv/bin/python -c "from agent.loss_tuner import read_constants; c = read_constants('agent/design_template_proteina.py'); print(f'Constants: {len(c)}'); assert 'BEAM_WIDTH' in c; assert 'WEIGHT_IPTM' in c; assert 'INVERSE_FOLD_SAMPLES' in c; print('Parsing OK')"`
Expected: "Constants: N" and "Parsing OK"

- [ ] **Step 4: Commit**

```bash
git add agent/design_template_proteina.py
git commit -m "feat: add Proteina-Complexa design template with beam search + inverse folding

Mosaic-wrapped jproteina-complexa flow matching with composable loss scoring.
Same designs.csv output format as Boltz-2 template. Includes inverse folding
(latent resampling on top-K backbones for sequence diversity).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add Proteina Grid Search to Loss Tuner

**Files:**
- Modify: `agent/loss_tuner.py`

- [ ] **Step 1: Add PROTEINA_GRID_SEARCH and --engine flag**

In `agent/loss_tuner.py`, add after `GRID_SEARCH_EXPERIMENTS` (after line 57):

```python
PROTEINA_GRID_SEARCH = [
    ("Baseline: ipTM + ipSAE beam scoring", {}),
    ("Wider beam: BEAM_WIDTH 4 → 8 — more candidates explored",
     {"BEAM_WIDTH": 8}),
    ("More branching: N_BRANCH 4 → 8 — wider tree search",
     {"N_BRANCH": 8}),
    ("Larger binder: 120 aa",
     {"MIN_LENGTH": 120, "MAX_LENGTH": 120}),
    ("Much larger binder: 200 aa",
     {"MIN_LENGTH": 200, "MAX_LENGTH": 200}),
    ("Add contact loss to beam scoring",
     {"WEIGHT_BINDER_TARGET_CONTACT": 1.0}),
    ("Add pLDDT to beam scoring",
     {"WEIGHT_PLDDT": 0.3}),
    ("Add binder PAE to beam scoring",
     {"WEIGHT_WITHIN_BINDER_PAE": 0.4}),
    ("More inverse folding: 10 per backbone",
     {"INVERSE_FOLD_SAMPLES": 10}),
    ("Fewer checkpoints (faster): [0, 200, 400]",
     {"STEP_CHECKPOINTS": [0, 200, 400]}),
]
```

Update `decide_simple()` to accept an `engine` parameter:

```python
def decide_simple(
    experiment_id: int,
    defaults: dict,
    experiments: list[dict],
    engine: str = "boltz2",
) -> tuple[str, dict] | None:
    """Simple grid search — return next experiment from predefined list."""
    grid = PROTEINA_GRID_SEARCH if engine == "proteina" else GRID_SEARCH_EXPERIMENTS
    if experiment_id >= len(grid):
        return None
    hypothesis, changes = grid[experiment_id]
    return hypothesis, changes
```

Update `run_tuning_loop()` signature to accept `engine` parameter and pass it to `decide_fn`.

Update `main()` to add the `--engine` argument:

```python
    parser.add_argument(
        "--engine", choices=["boltz2", "proteina"], default="boltz2",
        help="Design engine (default: boltz2)",
    )
```

And pass it through to `run_tuning_loop()`.

- [ ] **Step 2: Run tests**

Run: `cd /home/david/BindMaster2 && .venv/bin/python -m pytest bm2/tests/ tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Verify engine flag works**

Run: `cd /home/david/BindMaster2 && .venv/bin/python agent/loss_tuner.py --help 2>&1 | grep engine`
Expected: Shows `--engine {boltz2,proteina}`

- [ ] **Step 4: Commit**

```bash
git add agent/loss_tuner.py
git commit -m "feat: add --engine flag and Proteina grid search to loss tuner

Separate tuning tracks: Boltz-2 (MPNN, optimizer params) vs Proteina
(beam width, branching, inverse folding, checkpoints).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update Orchestrator with Engine Routing

**Files:**
- Modify: `agent/orchestrator.py`

- [ ] **Step 1: Add --engine flag and template routing**

In `agent/orchestrator.py`:

1. Add `engine: str = "boltz2"` parameter to `phase2_tune()` and `phase3_production()`
2. In `phase2_tune()`, select the correct template based on engine:
   ```python
   if engine == "proteina":
       template = Path(__file__).parent / "design_template_proteina.py"
   else:
       template = Path(__file__).parent / "design_template.py"
   ```
3. Pass `--engine` to the loss_tuner subprocess call
4. In `run_pipeline()`, add `engine` parameter and thread it through
5. In `main()`, add `--engine` argument:
   ```python
   parser.add_argument("--engine", choices=["boltz2", "proteina"],
                       default="boltz2", help="Design engine")
   ```

- [ ] **Step 2: Verify**

Run: `cd /home/david/BindMaster2 && .venv/bin/python agent/orchestrator.py --help 2>&1 | grep engine`
Expected: Shows `--engine {boltz2,proteina}`

- [ ] **Step 3: Commit**

```bash
git add agent/orchestrator.py
git commit -m "feat: add --engine routing to orchestrator for Proteina support

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Update Strategy Document and Mosaic Launcher

**Files:**
- Modify: `agent/strategy.md`
- Modify: `bm2/tools/mosaic.py`

- [ ] **Step 1: Add Proteina section to strategy.md**

Append to the end of `agent/strategy.md` (before the Lab-Specific Knowledge section):

```markdown
## Proteina-Complexa Beam Search

### When to Use

- **Fast exploration:** ~1-3 seconds per design vs ~2 minutes for Boltz-2
- **Length/loss screening:** Quickly test which binder lengths and loss compositions work
- **Backbone diversity:** Flow matching generates structurally diverse solutions
- **After Boltz-2 tuning:** Use Proteina to rapidly validate if the same loss weights work with a different generation method

### Beam Search Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| BEAM_WIDTH | 4 | 1-8 | Beams kept after pruning. More = wider search, slower. |
| N_BRANCH | 4 | 1-8 | Branches per beam per checkpoint. More = more diverse. |
| STEP_CHECKPOINTS | [0,100,200,300,400] | 2-10 intervals | Finer pruning schedule. More = better but slower. |
| INVERSE_FOLD_SAMPLES | 5 | 1-20 | Sequence variants per backbone. More = better chance of good sequence. |
| N_STEPS | 400 | 100-1000 | ODE integration steps. 400 is standard. |

### Loss Function (No MPNN)

Proteina generates sequences end-to-end via its decoder — no ProteinMPNN needed.
The loss function scores decoded designs for beam selection:

| Term | Default Weight | Purpose |
|------|---------------|---------|
| IPTMLoss | 1.0 | Interface confidence (primary) |
| BinderTargetIPSAE | 0.5 | Binding quality (binder→target) |
| TargetBinderIPSAE | 0.5 | Binding quality (target→binder) |
| BinderTargetContact | 0.0 | Contact count (optional) |
| PLDDTLoss | 0.0 | Per-residue confidence (optional) |
| WithinBinderPAE | 0.0 | Binder fold confidence (optional) |

### Known Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| All designs identical | BEAM_WIDTH too small | Increase to 6-8 |
| Poor sequences despite good backbone | Decoder variance | Increase INVERSE_FOLD_SAMPLES to 10-20 |
| OOM on GPU | Long binder + wide beam | Reduce BEAM_WIDTH or binder length |
| Slow beam search | Too many checkpoints | Use [0, 200, 400] instead of 5 intervals |
```

- [ ] **Step 2: Add proteina template support to Mosaic launcher**

In `bm2/tools/mosaic.py`, add a second search path constant:

```python
_PROTEINA_SEARCH_PATHS = [
    "examples/bindmaster_examples/hallucinate_proteina.py",
]
```

Update `prepare_config()` to check `extra_settings.get("engine", "boltz2")`:
- If `"proteina"`: search `_PROTEINA_SEARCH_PATHS` for the template
- If `"boltz2"` (default): existing behavior

The `_inject_parameters()` method works unchanged — it does text replacement on named constants, and both templates use the same pattern.

Add `_find_proteina_template()` method following the same pattern as `_find_hallucinate_template()`.

- [ ] **Step 3: Run tests**

Run: `cd /home/david/BindMaster2 && .venv/bin/python -m pytest bm2/tests/ tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add agent/strategy.md bm2/tools/mosaic.py
git commit -m "feat: add Proteina tuning guidance to strategy.md, update Mosaic launcher

Strategy: beam search params, loss composition without MPNN, failure modes.
Launcher: engine='proteina' selects hallucinate_proteina.py template.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: GPU Validation (when available)

**Files:**
- No code changes — validation only

**Prerequisite:** GPU must be free (~21 GB VRAM needed for Proteina models).

- [ ] **Step 1: Test design template directly**

Run a minimal test — 1 binder length, beam_width=2, n_branch=2, 1 inverse fold sample:

```bash
cd /tmp && mkdir -p proteina_test && cd proteina_test
# Create a minimal test config
cp /home/david/BindMaster2/agent/design_template_proteina.py test_proteina.py
# Edit: set TARGET_PDB, TARGET_SEQUENCE (PDL1), N_DESIGNS=4, TOP_K=2,
# INVERSE_FOLD_SAMPLES=1, BEAM_WIDTH=2, N_BRANCH=2, MIN/MAX_LENGTH=60
/home/david/BindMaster/Mosaic/.venv/bin/python test_proteina.py
```

Expected: `designs.csv` created with expected columns, PDB files in `structures_*/`

- [ ] **Step 2: Test loss tuner with Proteina engine**

```bash
cd /home/david/BindMaster2
.venv/bin/python agent/loss_tuner.py \
    --design /tmp/proteina_test/test_proteina.py \
    --output /tmp/proteina_tune \
    --mosaic-venv /home/david/BindMaster/Mosaic/.venv \
    --engine proteina \
    --mode simple \
    --max-experiments 2 \
    --timeout 300
```

Expected: 2 experiments run, experiment_log.json created, scores recorded

- [ ] **Step 3: Test orchestrator Phase 1 + 2**

```bash
.venv/bin/python agent/orchestrator.py \
    --target targets/PDL1.pdb --chain A \
    --output /tmp/proteina_campaign \
    --tune --tune-mode simple --tune-experiments 2 --tune-designs 4 \
    --engine proteina
```

Expected: target_analysis.json + tuning experiments + tuned design.py
