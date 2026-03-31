# Mosaic Loss Tuning Strategy

> This document guides the loss-tuning agent. Read it before every experiment.

## Target Context

[PLACEHOLDER — filled in by target_analyzer.py]

- Target name: {target_name}
- PDB code: {pdb_code}
- Organism: {organism}
- Chain: {chain}, {n_residues} residues
- Oligomeric state: {oligomeric_state}
- Surface analysis: {surface_summary}
- Difficulty: {difficulty}
- Recommended starting strategy: {strategy}

## Loss Function Reference

Each loss term drives Boltz-2 hallucination in a specific direction. Weights control the balance.

### Primary Terms (always active)

| Term | Default | Range | What it does |
|------|---------|-------|-------------|
| BinderTargetContact | 1.0 | 0.5–2.0 | Drives binder to make contacts with target. THE binding signal. Never set to 0. |
| WithinBinderContact | 1.0 | 0.5–2.0 | Binder internal packing. Without this, binders are floppy loops. |
| InverseFoldingSequenceRecovery | 10.0 | 0.5–20.0 | ProteinMPNN redesigns the predicted structure and checks if the designed sequence is recovered. Correlates with expression and stability. The Escalante innovation. |

### Confidence Terms

| Term | Default | Range | What it does |
|------|---------|-------|-------------|
| TargetBinderPAE | 0.05 | 0.01–0.2 | Penalizes high PAE from target→binder. Model confidence in where binder sits relative to target. |
| BinderTargetPAE | 0.05 | 0.01–0.2 | Same but binder→target direction. Usually set equal to TargetBinderPAE. |
| IPTMLoss | 0.025 | 0.01–0.1 | Interface predicted TM-score. Global interface confidence. |
| WithinBinderPAE | 0.4 | 0.1–1.0 | Binder internal confidence. "Does the binder know its own fold?" If too high, binder becomes over-rigid. |
| pTMEnergy | 0.025 | 0.01–0.1 | Overall complex pTM. |
| PLDDTLoss | 0.1 | 0.05–0.5 | Per-residue confidence. |

### Optional Terms

| Term | Default | Range | Trigger |
|------|---------|-------|---------|
| HelixLoss | 0.1 | 0.05–0.3 | SS_BIAS = "helix" |
| DistogramRadiusOfGyration | 0.1 | 0.05–0.3 | SS_BIAS = "compact" |

### ProteinMPNN Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| MPNN_TEMPERATURE | 0.001 | 0.001–0.1 | Lower = stricter sequence recovery. Too low → homopolymeric. Too high → unnatural sequences. |
| MPNN_MODEL | "soluble" | soluble/standard | "soluble" pretrained on soluble proteins — better for E. coli expression. |

## Optimizer Parameters

Three-stage simplex_APGM optimization with escalating sharpness:

| Stage | Steps | StepSize | Momentum | Scale | Logspace | Purpose |
|-------|-------|----------|----------|-------|----------|---------|
| 1 | 100 | 0.2√L | 0.3 | 1.0 | False | Soft exploration — find promising region |
| 2 | 50 | 0.5√L | 0.0 | 1.25 | True | Sharpening — refine the solution |
| 3 | 15 | 0.5√L | 0.0 | 1.4 | True | Final discretization — converge to sequence |

**Scale** controls L2 regularization. Higher values (>1.5) push toward discrete/sparse solutions faster. Lower values (<0.8) keep solutions softer longer. Range: 0.5–3.0.

**Logspace** switches from simplex to Bregman proximal method variant. Stages 2-3 use logspace for better convergence to discrete sequences.

## Design Strategy Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| BINDER_LENGTH | 80 | 30–400 | Escalante won Nipah with 220. Larger binders have more surface for binding but are slower to optimize. |
| EPITOPE_MODE | "none" | none/hotspot | "none" lets model choose binding site. "hotspot" uses EPITOPE_RESIDUES. Trust the model — Escalante's declarative approach beat all human epitope selections. |
| USE_MSA | False | True/False | MSA for target sequence in Boltz-2. False uses single-sequence mode (faster, no network dependency). True can be better for well-studied targets with many homologs. |
| N_DESIGNS | 50 | 10–500 | Per-experiment batch size. Keep small (50) for fast tuning iteration. Scale up for production. |

## Known Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| All sequences identical (diversity = 0) | SCALE too high, or dominant loss term | Reduce SCALE, increase N_STEPS, lower the dominant weight |
| Homopolymeric sequences (AAAA... or LLLL...) | WEIGHT_SEQ_RECOVERY > 15 | Reduce to 8–12 range |
| Adversarial unnatural sequences | WEIGHT_SEQ_RECOVERY < 2 | Increase to at least 5 |
| hit_rate drops to 0 | Change too aggressive | Immediately revert. Try smaller change. |
| All designs have high loss | BINDER_LENGTH too small for target | Try 120, 180, 250 |
| Good metrics but poor ranking_loss | Ranking loss uses ipSAE+ipTM which may differ from training loss | Expected — ranking loss is a separate quality check |
| OOM on GPU | BINDER_LENGTH too large for available VRAM | Reduce length or reduce num_samples |

## Exploration Strategy

1. **Start with ONE change per experiment** (scientific method)
2. **Try weights in powers of 2**: if current is 0.05, try 0.1 then 0.025
3. **For BINDER_LENGTH**, try discrete jumps: 60, 80, 120, 180, 250
4. **For MPNN_TEMPERATURE**, try: 0.001, 0.01, 0.05, 0.1
5. **After finding a good region**, do fine-grained search around it
6. **If hit_rate drops to 0**: immediately revert, the change was too aggressive
7. **If all sequences identical (diversity = 0)**: reduce SCALE, increase N_STEPS, or lower dominant loss
8. **After 3 consecutive failures**: try "big jump" — change 2-3 parameters at once
9. **After 3 big jump failures**: STOP — you've converged

## Safety Constraints — NEVER Violate

- Never set WEIGHT_BINDER_TARGET_CONTACT to 0 (no binding signal)
- Never set BINDER_LENGTH below 30 or above 400
- Never set N_STEPS below 50 (total across all stages)
- Never remove both PAE loss terms simultaneously
- Never set WEIGHT_SEQ_RECOVERY below 0.5 (adversarial sequences)
- Always keep at least one contact loss and one confidence loss active

## Lessons from Nipah Competition (Escalante Bio)

- Won with 9/10 binders binding (90% hit rate)
- Key factors: (a) Boltz-2 as loss model, (b) 10.0-weighted sequence recovery, (c) NO epitope specification (declarative design), (d) large 220 AA binders, (e) no filtering
- The model chose to bind the stalk domain (coiled-coil), not the head domain (small, recessed, glycan-occluded). Better than any human choice.
- Multiple epitope engagement (avidity) may have contributed with the tetrameric target
- **Takeaway: trust the model. Large binders can be better. Don't over-filter.**

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

## Lab-Specific Knowledge (Loschmidt Laboratories)

- Expression system: E. coli BL21(DE3), pET21b, 0.1 mM IPTG, 18°C overnight
- Best result: 29B binder for CBG at 10-100nM, Tm=65°C, 20 mg/L yield
- BindCraft on CBG: ~40% expression success
- BoltzGen on CBG: 0/7 expression (sequences too charged, didn't fold)
- Buffer: 500 mM NaCl improves solubility
- SPR (P4SPR/P4PRO) most reliable for binding measurement
