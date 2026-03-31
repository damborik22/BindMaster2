"""Proteina-Complexa beam search + inverse folding binder design template.

Uses NVIDIA's Proteina-Complexa flow matching model (via Mosaic wrapper)
for backbone generation, with Boltz-2 cross-model scoring for Stage 2
refolding.  Same designs.csv output format as the Boltz-2 template.

Run with:  /home/david/BindMaster/Mosaic/.venv/bin/python agent/design_template_proteina.py
"""

import uuid
import os
import csv
import signal
import json
import sys
import time

import gemmi
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from jproteina_complexa.hub import load_denoiser, load_decoder
from jproteina_complexa.pdb import load_target_cond, make_structure
from jproteina_complexa.flow_matching import (
    DenoiseState,
    init_noise,
    PRODUCTION_SAMPLING,
    predict_x1_from_v,
)
from jproteina_complexa.types import (
    DenoiserBatch,
    NoisyState,
    Timesteps,
    DecoderBatch,
)
from jproteina_complexa.constants import AA_CODES, AA_3LETTER

from mosaic.models.proteina import beam_search, ScoredDesign
from mosaic.models.boltz2 import Boltz2
from mosaic.common import TOKENS as MOSAIC_ORDER
import mosaic.losses.structure_prediction as sp
from mosaic.structure_prediction import TargetChain


# ============================
# TARGET PARAMETERS
# ============================
TARGET_PDB = ""
TARGET_SEQUENCE = ""
TARGET_CHAIN_ID = "A"
HOTSPOT_RESIDUES = []           # 0-indexed residue indices on the target chain

# ============================
# DESIGN STRATEGY
# ============================
N_DESIGNS = 50                  # beam search runs per binder length
TOP_K = 10                      # how many top designs get full Boltz-2 refolding
INVERSE_FOLD_SAMPLES = 5        # inverse folding variants per top-K backbone
MIN_LENGTH = 80                 # minimum binder length (aa)
MAX_LENGTH = 80                 # maximum binder length (aa); set equal for single-length
LENGTH_STEP = 5                 # step between lengths if scanning

# ============================
# BEAM SEARCH
# ============================
BEAM_WIDTH = 4                  # beams kept after each pruning
N_BRANCH = 4                    # branches explored per beam per checkpoint
STEP_CHECKPOINTS = [0, 100, 200, 300, 400]
N_STEPS = 400                   # total denoising steps (must equal last checkpoint)
SEED = 42

# ============================
# LOSS WEIGHTS (beam scoring — no MPNN)
# ============================
WEIGHT_IPTM = 1.0
WEIGHT_BINDER_TARGET_IPSAE = 0.5
WEIGHT_TARGET_BINDER_IPSAE = 0.5
WEIGHT_BINDER_TARGET_CONTACT = 0.0
WEIGHT_PLDDT = 0.0
WEIGHT_WITHIN_BINDER_PAE = 0.0

# ============================
# RANKING & REFOLDING (Boltz-2 Stage 2)
# ============================
RANKING_RECYCLES = 3            # Boltz-2 recycling for ranking loss
RANKING_SAMPLES = 6             # number of samples for ranking
REFOLD_RECYCLES = 3             # Boltz-2 recycling for Stage 2 full predict
USE_MSA = False                 # whether target uses MSA in Boltz-2
MIN_RANKING_LOSS = None         # filter threshold (None = no filter)
MIN_HAMMING = 0                 # minimum Hamming distance for diversity filter
MIN_IPTM_AUX = None            # gate for Stage 2 entry (None = no gate)


# ============================
# INTERNAL STATE
# ============================

_interrupt_state = {
    "candidates": [],
    "checkpoint_path": None,
}


# ============================
# HELPER FUNCTIONS
# ============================


def _check_gpu():
    devices = jax.devices()
    if all(d.platform == "cpu" for d in devices):
        print("WARNING: No GPU detected — JAX is running on CPU only.")
        print("         This will be very slow. Consider running on a GPU machine.")
    else:
        print(f"GPU detected: {[str(d) for d in devices]}")


def _hamming_distance(seq_a, seq_b):
    """Character-wise Hamming distance between two equal-length strings."""
    return sum(a != b for a, b in zip(seq_a, seq_b))


def _diversity_filter(candidates, min_hamming):
    """Greedy diversity filter: keep a candidate only if it is at least
    min_hamming away (Hamming distance) from every already-accepted candidate.
    Input list is assumed to be sorted best->worst (lower loss first).
    """
    accepted = []
    for seq, loss_val, source in candidates:
        if all(_hamming_distance(seq, acc_seq) >= min_hamming for acc_seq, _, _ in accepted):
            accepted.append((seq, loss_val, source))
    return accepted


def _nan_safe(obj):
    """Recursively replace float nan/inf with None for JSON serialisation."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_safe(x) for x in obj]
    return obj


def _save_checkpoint(path, data):
    with open(path, "w") as f:
        json.dump(_nan_safe(data), f, indent=2)
    print(f"  [checkpoint] Saved -> {path}")


def _load_checkpoint(path):
    with open(path) as f:
        data = json.load(f)
    raw_candidates = data.get("candidates", [])
    candidates = []
    for item in raw_candidates:
        seq, lv, source = item
        if lv is None:
            lv = float("nan")
        candidates.append((seq, float(lv), source))
    data["candidates"] = candidates
    return data


def _install_signal_handler(get_candidates_fn, checkpoint_path_fn):
    """Install SIGINT handler that saves a checkpoint then exits cleanly."""

    def _handler(signum, frame):
        print("\n\nInterrupted! Saving checkpoint before exit...")
        candidates = get_candidates_fn()
        checkpoint_path = checkpoint_path_fn()
        if checkpoint_path and candidates is not None:
            _save_checkpoint(
                checkpoint_path,
                {
                    "interrupted": True,
                    "candidates": candidates,
                },
            )
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)


def _print_length_summary(summary_rows):
    """Print a summary table after a multi-length scan."""
    print("\n" + "=" * 60)
    print("=== Length Scan Summary ===")
    print(f"{'Length':>8}  {'Best ranking_loss':>18}  {'N designs':>10}")
    print("-" * 60)
    for row in summary_rows:
        length = row["binder_length"]
        best = row["best_ranking_loss"]
        n = row["n_designs"]
        best_str = f"{best:.4f}" if best is not None else "  (filtered)"
        print(f"{length:>8}  {best_str:>18}  {n:>10}")
    print("=" * 60)


# ============================
# INVERSE FOLDING
# ============================


@eqx.filter_jit
def _inverse_fold(denoiser, decoder, bb_ca, mask, target, key):
    """Denoise latents from noise while keeping backbone fixed, then decode.

    Args:
        denoiser: denoiser model (LocalLatentsTransformer)
        decoder: decoder model (DecoderTransformer)
        bb_ca: backbone CA coordinates in Angstroms [N, 3]
        mask: boolean residue mask [N]
        target: TargetCond for the target protein
        key: PRNG key

    Returns:
        DecoderOutput with the inverse-folded sequence and coordinates.
    """
    bb_ca_nm = bb_ca / 10.0  # Angstroms -> nanometers
    cfg = PRODUCTION_SAMPLING
    nsteps = cfg.nsteps
    ts_lat = cfg.local_latents.time_schedule(nsteps)
    mask_f = mask.astype(jnp.float32)

    k_noise, k_run = jax.random.split(key)
    state = init_noise(k_noise, 8, mask, cfg)
    state = DenoiseState(
        bb=bb_ca_nm,
        lat=state.lat,
        sc_bb=bb_ca_nm,
        sc_lat=state.sc_lat,
        key=k_run,
    )

    def body(carry):
        state, key, i = carry
        t_lat = ts_lat[i]
        dt_lat = ts_lat[i + 1] - t_lat

        out = denoiser(
            DenoiserBatch(
                x_t=NoisyState(bb_ca=bb_ca_nm, local_latents=state.lat),
                t=Timesteps(bb_ca=jnp.array(1.00), local_latents=t_lat),
                mask=mask,
                x_sc=NoisyState(bb_ca=state.sc_bb, local_latents=state.sc_lat),
                target=target,
            )
        )

        sc_lat = predict_x1_from_v(state.lat, out.local_latents, t_lat)

        key, k_step = jax.random.split(key)
        lat = cfg.local_latents.step(
            state.lat,
            out.local_latents,
            t_lat,
            dt_lat,
            mask_f,
            k_step,
        )

        return (
            DenoiseState(
                bb=bb_ca_nm,
                lat=lat,
                sc_bb=bb_ca_nm,
                sc_lat=sc_lat,
                key=key,
            ),
            key,
            i + 1,
        )

    state, _, _ = jax.lax.fori_loop(
        0,
        nsteps,
        lambda i, carry: body(carry),
        (state, k_run, jnp.int32(0)),
    )
    return decoder(
        DecoderBatch(
            z_latent=state.lat,
            ca_coors=state.bb * 10.0,
            mask=mask,
        )
    )


# ============================
# PDB WRITING
# ============================


def _make_complex_pdb(decoder_output, target_cond):
    """Build a gemmi Structure with binder (chain A) + target (chain B)."""
    seq = "".join(AA_CODES[j] for j in np.array(decoder_output.aatype))
    return make_structure(
        [
            (
                "A",
                [AA_3LETTER[aa] for aa in seq],
                np.array(decoder_output.coors),
                np.array(decoder_output.atom_mask).astype(np.float32),
            ),
            (
                "B",
                [AA_3LETTER[AA_CODES[i]] for i in np.array(target_cond.seq)],
                np.array(target_cond.coords),
                np.array(target_cond.atom_mask),
            ),
        ]
    )


# ============================
# METRICS EXTRACTION
# ============================


def _merge_aux_entries(aux):
    merged = {}
    for entry in aux:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            merged.setdefault(key, []).append(value)
    return merged


def _flatten_numeric_values(value):
    if value is None:
        return []

    stack = [value]
    out = []

    while stack:
        item = stack.pop()

        if item is None:
            continue
        if isinstance(item, dict):
            stack.extend(item.values())
            continue
        if isinstance(item, (list, tuple)):
            stack.extend(item)
            continue

        arr = np.asarray(item)
        if arr.dtype == object:
            stack.extend(arr.tolist())
            continue

        for x in np.ravel(arr):
            try:
                v = float(x)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                out.append(v)

    return out


def _mean_aux_metric(aux_dict, key, aliases=()):
    for candidate_key in (key, *aliases):
        values = _flatten_numeric_values(aux_dict.get(candidate_key))
        if values:
            return float(np.mean(values)), candidate_key, len(values)
    return float("nan"), None, 0


def _extract_prediction_metrics(prediction, binder_length):
    """Slice PAE and pLDDT arrays into binder/target regions and compute statistics."""
    plddt = np.array(prediction.plddt)
    pae = np.array(prediction.pae)

    L_b = binder_length
    plddt_b = plddt[:L_b]
    plddt_t = plddt[L_b:]
    pae_bb = pae[:L_b, :L_b]
    pae_bt = pae[:L_b, L_b:]
    pae_tb = pae[L_b:, :L_b]
    pae_tt = pae[L_b:, L_b:]

    return {
        "iptm": float(prediction.iptm),
        "plddt_binder_mean": float(plddt_b.mean()),
        "plddt_binder_min": float(plddt_b.min()),
        "plddt_binder_max": float(plddt_b.max()),
        "plddt_binder_std": float(plddt_b.std()),
        "plddt_target_mean": float(plddt_t.mean()) if len(plddt_t) > 0 else float("nan"),
        "plddt_target_min": float(plddt_t.min()) if len(plddt_t) > 0 else float("nan"),
        "pae_bb_mean": float(pae_bb.mean()),
        "pae_bt_mean": float(pae_bt.mean()),
        "pae_tb_mean": float(pae_tb.mean()),
        "pae_tt_mean": float(pae_tt.mean()) if pae_tt.size > 0 else float("nan"),
        "pae_overall_mean": float(pae.mean()),
        "pae_max": float(pae.max()),
    }


# ============================
# DESIGN LOOP
# ============================


def design(
    n_designs: int,
    top_k: int,
    binder_length: int,
    target_sequence: str,
    output_dir: str = "structures",
    *,
    checkpoint_path=None,
    resume_from=None,
    min_ranking_loss=None,
    min_hamming=0,
    min_iptm_aux=None,
):
    """Run a Proteina-Complexa binder design campaign for one binder_length.

    Pipeline:
      1. Beam search over flow matching trajectories (N_DESIGNS runs)
      2. Sort by loss, take top-K
      3. Inverse fold each top-K backbone INVERSE_FOLD_SAMPLES times
      4. Score all variants (beam + inverse-folded) via Boltz-2 ranking loss
      5. Full Boltz-2 refolding for final top-K

    Returns a dict with keys:
        best_ranking_loss : float | None
        n_designs         : int
    """
    worker_id = str(uuid.uuid4())[:8]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nInitializing Proteina-Complexa design run:")
    print(f"  Worker ID: {worker_id}")
    print(f"  Binder length: {binder_length} aa")
    print(f"  Target length: {len(target_sequence)} aa")
    print(f"  Beam search runs: {n_designs}")
    print(f"  Beam width: {BEAM_WIDTH}, Branch factor: {N_BRANCH}")
    print(f"  Checkpoints: {STEP_CHECKPOINTS}")
    print(f"  Top designs to refold: {top_k}")
    print(f"  Inverse fold samples per top-K: {INVERSE_FOLD_SAMPLES}")
    print(f"  Output directory: {output_dir}")
    if checkpoint_path:
        print(f"  Checkpoint path: {checkpoint_path}")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
    if min_ranking_loss is not None:
        print(f"  Min ranking_loss threshold: {min_ranking_loss}")
    if min_hamming > 0:
        print(f"  Min Hamming distance (diversity): {min_hamming}")
    if min_iptm_aux is not None:
        print(f"  Min iptm_aux gate: {min_iptm_aux}")

    _checkpoint_file = checkpoint_path or f"checkpoint_{worker_id}.json"
    candidates_ref = []
    _interrupt_state["candidates"] = candidates_ref
    _interrupt_state["checkpoint_path"] = _checkpoint_file

    # ------------------------------------------------------------------
    # Load Proteina-Complexa models
    # ------------------------------------------------------------------
    print("\nLoading Proteina-Complexa denoiser + decoder...")
    t0 = time.perf_counter()
    denoiser = load_denoiser()
    decoder = load_decoder()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Load target from PDB
    # ------------------------------------------------------------------
    print("Loading target from PDB...")
    st = gemmi.read_structure(TARGET_PDB)
    st.setup_entities()
    target_chain = None
    for chain in st[0]:
        if chain.name == TARGET_CHAIN_ID:
            target_chain = chain
            break
    if target_chain is None:
        target_chain = st[0][0]
        print(f"  WARNING: Chain {TARGET_CHAIN_ID} not found, using chain {target_chain.name}")

    hotspots = HOTSPOT_RESIDUES if HOTSPOT_RESIDUES else None
    target_cond = load_target_cond(target_chain, hotspots=hotspots)
    print(f"  Target chain: {TARGET_CHAIN_ID} ({len(target_cond.seq)} residues)")
    if hotspots:
        print(f"  Hotspot residues (0-indexed): {hotspots}")

    mask = jnp.ones(binder_length, dtype=jnp.bool_)

    # ------------------------------------------------------------------
    # Build Boltz-2 beam scoring loss function
    # ------------------------------------------------------------------
    print("Building Boltz-2 beam scoring loss...")
    folder = Boltz2()

    beam_sp_loss = (
        WEIGHT_IPTM * sp.IPTMLoss()
        + WEIGHT_BINDER_TARGET_IPSAE * sp.BinderTargetIPSAE()
        + WEIGHT_TARGET_BINDER_IPSAE * sp.TargetBinderIPSAE()
    )
    if WEIGHT_BINDER_TARGET_CONTACT > 0:
        beam_sp_loss = beam_sp_loss + WEIGHT_BINDER_TARGET_CONTACT * sp.BinderTargetContact()
    if WEIGHT_PLDDT > 0:
        beam_sp_loss = beam_sp_loss + WEIGHT_PLDDT * sp.PLDDTLoss()
    if WEIGHT_WITHIN_BINDER_PAE > 0:
        beam_sp_loss = beam_sp_loss + WEIGHT_WITHIN_BINDER_PAE * sp.WithinBinderPAE()

    features, _ = folder.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence, use_msa=USE_MSA)],
    )
    loss_fn = folder.build_multisample_loss(
        loss=beam_sp_loss,
        features=features,
        recycling_steps=1,
        num_samples=1,
    )

    @eqx.filter_jit
    def evaluate_loss(loss_fn, pssm, key):
        return loss_fn(pssm, key=key)

    # ------------------------------------------------------------------
    # Stage 1: Beam search
    # ------------------------------------------------------------------
    print(f"\n=== Stage 1: Beam search ({n_designs} runs) ===")

    if resume_from is not None:
        ckpt = _load_checkpoint(resume_from)
        candidates = ckpt["candidates"]
        print(f"  Loaded {len(candidates)} candidates from checkpoint: {resume_from}")
        for i, (seq, lv, src) in enumerate(candidates[:5]):
            print(f"    [{i + 1}] loss={lv:.4f}  source={src}  seq={seq}")
        if len(candidates) > 5:
            print(f"    ... and {len(candidates) - 5} more")
    else:
        candidates = candidates_ref

        for run_idx in range(n_designs):
            seed = SEED + run_idx
            print(f"\n[Run {run_idx + 1}/{n_designs}] beam search (seed={seed})...")
            t0 = time.perf_counter()

            designs = beam_search(
                model=denoiser,
                decoder=decoder,
                loss_fn=loss_fn,
                mask=mask,
                key=jax.random.PRNGKey(seed),
                target=target_cond,
                step_checkpoints=STEP_CHECKPOINTS,
                beam_width=BEAM_WIDTH,
                n_branch=N_BRANCH,
            )

            # Sort and extract best designs from this run
            designs_sorted = sorted(designs, key=lambda d: float(d.loss))

            elapsed = time.perf_counter() - t0
            print(f"  {len(designs)} designs evaluated in {elapsed:.1f}s")

            # Take all unique designs from this run
            seen_seqs = {seq for seq, _, _ in candidates}
            for d in designs_sorted:
                seq_str = "".join(MOSAIC_ORDER[i] for i in np.array(d.sequence))
                if seq_str not in seen_seqs:
                    candidates.append((seq_str, float(d.loss), "beam"))
                    seen_seqs.add(seq_str)

            best_in_run = designs_sorted[0]
            best_seq = "".join(MOSAIC_ORDER[i] for i in np.array(best_in_run.sequence))
            print(f"  Best: loss={float(best_in_run.loss):.4f}  seq={best_seq}")

        candidates = sorted(candidates, key=lambda x: x[1])
        _interrupt_state["candidates"] = candidates

        # ------------------------------------------------------------------
        # Stage 1b: Inverse folding on top-K backbones
        # ------------------------------------------------------------------
        if INVERSE_FOLD_SAMPLES > 0:
            # Re-run beam search for the single best seed to get backbone access
            # (we need the ScoredDesign objects with .bb for inverse folding)
            print(f"\n=== Stage 1b: Inverse folding (top {top_k} backbones x {INVERSE_FOLD_SAMPLES} samples) ===")

            # Collect the top-K unique beam sequences and re-run their seeds to get backbones
            # More efficient: run one final beam search with the best seed and use those backbones
            best_seed = SEED  # Use first seed's beam search for backbone extraction
            print(f"  Re-running beam search (seed={best_seed}) for backbone extraction...")
            t0 = time.perf_counter()

            designs_for_inv = beam_search(
                model=denoiser,
                decoder=decoder,
                loss_fn=loss_fn,
                mask=mask,
                key=jax.random.PRNGKey(best_seed),
                target=target_cond,
                step_checkpoints=STEP_CHECKPOINTS,
                beam_width=BEAM_WIDTH,
                n_branch=N_BRANCH,
            )
            designs_for_inv = sorted(designs_for_inv, key=lambda d: float(d.loss))
            print(f"  Backbone extraction done in {time.perf_counter() - t0:.1f}s")

            # Take top-K unique backbones for inverse folding
            inv_fold_count = min(top_k, len(designs_for_inv))
            for bb_idx in range(inv_fold_count):
                d = designs_for_inv[bb_idx]
                orig_seq = "".join(MOSAIC_ORDER[i] for i in np.array(d.sequence))
                print(f"\n  Inverse folding backbone {bb_idx + 1}/{inv_fold_count} (loss={float(d.loss):.4f})...")

                for sample_idx in range(INVERSE_FOLD_SAMPLES):
                    inv_key = jax.random.PRNGKey(SEED + 1000 + bb_idx * INVERSE_FOLD_SAMPLES + sample_idx)
                    t0 = time.perf_counter()
                    inv_output = _inverse_fold(
                        denoiser, decoder,
                        jnp.array(d.bb),  # already in Angstroms from ScoredDesign
                        mask,
                        target_cond,
                        inv_key,
                    )
                    jax.block_until_ready(inv_output.aatype)

                    # Convert jpc token order to one-letter sequence
                    inv_seq = "".join(AA_CODES[j] for j in np.array(inv_output.aatype))

                    # Convert to mosaic order for Boltz-2 scoring
                    inv_seq_mosaic = jnp.array(inv_output.seq_logits[..., jnp.array([AA_CODES.index(aa) for aa in MOSAIC_ORDER])].argmax(-1))
                    inv_seq_str = "".join(MOSAIC_ORDER[i] for i in np.array(inv_seq_mosaic))

                    # Score with Boltz-2 ranking loss
                    inv_pssm = jax.nn.one_hot(inv_seq_mosaic, 20)

                    boltz_features_inv, _ = folder.target_only_features(
                        chains=[
                            TargetChain(sequence=inv_seq_str, use_msa=USE_MSA),
                            TargetChain(sequence=target_sequence, use_msa=USE_MSA),
                        ]
                    )
                    ranking_loss_inv = folder.build_multisample_loss(
                        loss=1.00 * sp.IPTMLoss() + 0.5 * sp.TargetBinderIPSAE() + 0.5 * sp.BinderTargetIPSAE(),
                        features=boltz_features_inv,
                        recycling_steps=RANKING_RECYCLES,
                        num_samples=RANKING_SAMPLES,
                    )
                    loss_val_inv, _ = evaluate_loss(ranking_loss_inv, inv_pssm, key=jax.random.key(0))

                    n_mut = _hamming_distance(orig_seq, inv_seq_str)
                    elapsed = time.perf_counter() - t0
                    print(f"    Sample {sample_idx + 1}: loss={float(loss_val_inv):.4f}  mutations={n_mut}/{binder_length}  ({elapsed:.1f}s)  seq={inv_seq_str}")

                    if inv_seq_str not in {s for s, _, _ in candidates}:
                        candidates.append((inv_seq_str, float(loss_val_inv), f"inv_bb{bb_idx + 1}"))

            candidates = sorted(candidates, key=lambda x: x[1])
            _interrupt_state["candidates"] = candidates

        _save_checkpoint(
            _checkpoint_file,
            {
                "worker_id": worker_id,
                "binder_length": binder_length,
                "n_designs": n_designs,
                "top_k": top_k,
                "target_sequence": target_sequence,
                "output_dir": output_dir,
                "candidates": candidates,
                "interrupted": False,
            },
        )

    candidates = sorted(candidates, key=lambda x: x[1])

    print(f"\n=== Design ranking (beam + inverse-folded) ===")
    for i, (seq, loss_val, source) in enumerate(candidates[:min(10, len(candidates))]):
        print(f"  Rank {i + 1}: loss={loss_val:.4f}  source={source}  seq={seq}")
    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more designs")

    if min_ranking_loss is not None:
        candidates = [(s, lv, src) for s, lv, src in candidates if lv <= min_ranking_loss]
        print(f"  Threshold gate (<= {min_ranking_loss}): {len(candidates)} candidates pass")
        if not candidates:
            print("  No candidates passed the threshold gate — skipping Stage 2.")
            return {"best_ranking_loss": None, "n_designs": len(candidates)}

    if min_hamming > 0:
        before = len(candidates)
        candidates = _diversity_filter(candidates, min_hamming)
        print(f"  Diversity filter (Hamming >= {min_hamming}): {before} -> {len(candidates)} candidates")

    # ------------------------------------------------------------------
    # Stage 2: Boltz-2 refolding for top-K
    # ------------------------------------------------------------------
    top_k = max(0, min(top_k, len(candidates)))
    print(f"\n=== Stage 2: Boltz-2 refolding top {top_k} designs ===")

    final_lines = []
    csv_rows = []

    for rank, (seq_str, fast_loss, source) in enumerate(candidates):
        is_top = rank < top_k

        ranking_loss_value = float(fast_loss)
        iptm_aux = float("nan")
        bt_ipsae = float("nan")
        tb_ipsae = float("nan")
        ipsae_min = float("nan")
        bt_iptm = float("nan")
        binder_ptm = float("nan")
        plddt_aux = float("nan")
        bb_pae = float("nan")
        bt_pae_aux = float("nan")
        tb_pae = float("nan")
        intra_contact = float("nan")
        target_contact = float("nan")
        pTMEnergy_val = float("nan")
        iptm = float("nan")
        plddt_binder_mean = float("nan")
        plddt_binder_min = float("nan")
        plddt_binder_max = float("nan")
        plddt_binder_std = float("nan")
        plddt_target_mean = float("nan")
        plddt_target_min = float("nan")
        pae_bb_mean = float("nan")
        pae_bt_mean = float("nan")
        pae_tb_mean = float("nan")
        pae_tt_mean = float("nan")
        pae_overall_mean = float("nan")
        pae_max = float("nan")
        pdb_path = ""
        pae_file = ""
        plddt_file = ""

        if is_top:
            print(f"\n[Rank {rank + 1}] refolding  source={source}  seq={seq_str}")

            seq = jnp.array([MOSAIC_ORDER.index(c) for c in seq_str])

            boltz_features, boltz_writer = folder.target_only_features(
                chains=[
                    TargetChain(sequence=seq_str, use_msa=USE_MSA),
                    TargetChain(sequence=target_sequence, use_msa=USE_MSA),
                ]
            )

            metrics_loss = folder.build_multisample_loss(
                loss=(
                    sp.IPTMLoss()
                    + sp.BinderTargetIPSAE()
                    + sp.TargetBinderIPSAE()
                    + sp.IPSAE_min()
                    + sp.BinderTargetIPTM()
                    + sp.BinderPTMLoss()
                    + sp.PLDDTLoss()
                    + sp.WithinBinderPAE()
                    + sp.BinderTargetPAE()
                    + sp.TargetBinderPAE()
                    + sp.WithinBinderContact()
                    + sp.BinderTargetContact()
                    + sp.pTMEnergy()
                ),
                features=boltz_features,
                recycling_steps=REFOLD_RECYCLES,
                num_samples=RANKING_SAMPLES,
            )
            _, aux = evaluate_loss(metrics_loss, jax.nn.one_hot(seq, 20), key=jax.random.key(0))

            aux_dict = _merge_aux_entries(aux)

            iptm_aux, _, _ = _mean_aux_metric(aux_dict, "iptm")
            bt_ipsae, bt_key, bt_n = _mean_aux_metric(aux_dict, "bt_ipsae", aliases=("binder_target_ipsae",))
            tb_ipsae, tb_key, tb_n = _mean_aux_metric(aux_dict, "tb_ipsae", aliases=("target_binder_ipsae",))
            ipsae_min, _, _ = _mean_aux_metric(aux_dict, "ipsae_min")
            bt_iptm, _, _ = _mean_aux_metric(aux_dict, "bt_iptm")
            binder_ptm, _, _ = _mean_aux_metric(aux_dict, "binder_ptm")
            plddt_aux, _, _ = _mean_aux_metric(aux_dict, "plddt")
            bb_pae, _, _ = _mean_aux_metric(aux_dict, "bb_pae")
            bt_pae_aux, _, _ = _mean_aux_metric(aux_dict, "bt_pae")
            tb_pae, _, _ = _mean_aux_metric(aux_dict, "tb_pae")
            intra_contact, _, _ = _mean_aux_metric(aux_dict, "intra_contact")
            target_contact, _, _ = _mean_aux_metric(aux_dict, "target_contact")
            pTMEnergy_val, _, _ = _mean_aux_metric(aux_dict, "pTMEnergy")

            if rank == 0:
                print(f"  [debug] aux keys: {sorted(aux_dict.keys())}")
                print(f"  [debug] bt source={bt_key} n={bt_n}  tb source={tb_key} n={tb_n}")

            if min_iptm_aux is not None and iptm_aux < min_iptm_aux:
                print(f"  [gate] iptm_aux={iptm_aux:.4f} < {min_iptm_aux} — skipping full predict")
                is_top = False

        if is_top:
            prediction = folder.predict(
                PSSM=jax.nn.one_hot(seq, 20),
                features=boltz_features,
                writer=boltz_writer,
                recycling_steps=REFOLD_RECYCLES,
                key=jax.random.key(0),
            )

            pred_metrics = _extract_prediction_metrics(prediction, binder_length)
            iptm = pred_metrics["iptm"]
            plddt_binder_mean = pred_metrics["plddt_binder_mean"]
            plddt_binder_min = pred_metrics["plddt_binder_min"]
            plddt_binder_max = pred_metrics["plddt_binder_max"]
            plddt_binder_std = pred_metrics["plddt_binder_std"]
            plddt_target_mean = pred_metrics["plddt_target_mean"]
            plddt_target_min = pred_metrics["plddt_target_min"]
            pae_bb_mean = pred_metrics["pae_bb_mean"]
            pae_bt_mean = pred_metrics["pae_bt_mean"]
            pae_tb_mean = pred_metrics["pae_tb_mean"]
            pae_tt_mean = pred_metrics["pae_tt_mean"]
            pae_overall_mean = pred_metrics["pae_overall_mean"]
            pae_max = pred_metrics["pae_max"]

            pdb_path = f"{output_dir}/top{rank + 1}_{worker_id}.pdb"
            pae_file = f"{output_dir}/top{rank + 1}_{worker_id}_pae.npy"
            plddt_file = f"{output_dir}/top{rank + 1}_{worker_id}_plddt.csv"

            with open(pdb_path, "w") as f:
                f.write(prediction.st.make_pdb_string())

            np.save(pae_file, np.array(prediction.pae))

            plddt_full = np.array(prediction.plddt)
            with open(plddt_file, "w", newline="") as f:
                plddt_writer = csv.writer(f)
                plddt_writer.writerow(["residue_idx", "chain", "residue_in_chain", "plddt"])
                for i, v in enumerate(plddt_full):
                    chain_label = "binder" if i < binder_length else "target"
                    res_in_chain = i if i < binder_length else i - binder_length
                    plddt_writer.writerow([i, chain_label, res_in_chain, f"{v:.6f}"])

            print(
                f"  Interface:      iptm={iptm:.4f}  bt_ipsae={bt_ipsae:.4f}  tb_ipsae={tb_ipsae:.4f}  ipsae_min={ipsae_min:.4f}  bt_iptm={bt_iptm:.4f}"
            )
            print(
                f"  Binder quality: binder_ptm={binder_ptm:.4f}  plddt_mean={plddt_binder_mean:.4f}  plddt_min={plddt_binder_min:.4f}  pae_bb={pae_bb_mean:.4f}  intra_contact={intra_contact:.4f}"
            )
            print(
                f"  PAE overview:   pae_bt={pae_bt_mean:.4f}  pae_tb={pae_tb_mean:.4f}  pae_bb={pae_bb_mean:.4f}  pae_overall={pae_overall_mean:.4f}  pae_max={pae_max:.4f}"
            )
            print(f"  Energy/contacts: pTMEnergy={pTMEnergy_val:.4f}  target_contact={target_contact:.4f}")
            print(f"  Files:  pdb={pdb_path}  pae={pae_file}  plddt={plddt_file}")

            header = (
                f">rank{rank + 1}_{worker_id}"
                f"  source={source}"
                f"  binder_length={binder_length}"
                f"  ranking_loss={ranking_loss_value:.4f}"
                f"  iptm={iptm:.4f}"
                f"  bt_ipsae={bt_ipsae:.4f}"
                f"  tb_ipsae={tb_ipsae:.4f}"
                f"  ipsae_min={ipsae_min:.4f}"
                f"  bt_iptm={bt_iptm:.4f}"
                f"  binder_ptm={binder_ptm:.4f}"
                f"  plddt_mean={plddt_binder_mean:.4f}"
                f"  plddt_min={plddt_binder_min:.4f}"
                f"  pae_bb={pae_bb_mean:.4f}"
                f"  pTMEnergy={pTMEnergy_val:.4f}"
                f"  intra_contact={intra_contact:.4f}"
                f"  target_contact={target_contact:.4f}"
                f"  pdb={pdb_path}"
            )
        else:
            header = (
                f">rank{rank + 1}_{worker_id}  source={source}  binder_length={binder_length}  ranking_loss={ranking_loss_value:.4f}"
            )

        final_lines.append(f"{header}\n{seq_str}")

        csv_rows.append(
            {
                "worker_id": worker_id,
                "rank": rank + 1,
                "is_top": int(is_top),
                "sequence": seq_str,
                "target_sequence": target_sequence,
                "binder_length": binder_length,
                "ranking_loss": ranking_loss_value,
                "iptm_aux": iptm_aux,
                "bt_ipsae": bt_ipsae,
                "tb_ipsae": tb_ipsae,
                "ipsae_min": ipsae_min,
                "bt_iptm": bt_iptm,
                "binder_ptm": binder_ptm,
                "plddt_aux": plddt_aux,
                "bb_pae": bb_pae,
                "bt_pae_aux": bt_pae_aux,
                "tb_pae": tb_pae,
                "intra_contact": intra_contact,
                "target_contact": target_contact,
                "pTMEnergy": pTMEnergy_val,
                "iptm": iptm,
                "plddt_binder_mean": plddt_binder_mean,
                "plddt_binder_min": plddt_binder_min,
                "plddt_binder_max": plddt_binder_max,
                "plddt_binder_std": plddt_binder_std,
                "plddt_target_mean": plddt_target_mean,
                "plddt_target_min": plddt_target_min,
                "pae_bb_mean": pae_bb_mean,
                "pae_bt_mean": pae_bt_mean,
                "pae_tb_mean": pae_tb_mean,
                "pae_tt_mean": pae_tt_mean,
                "pae_overall_mean": pae_overall_mean,
                "pae_max": pae_max,
                "pdb": pdb_path,
                "pae_file": pae_file,
                "plddt_file": plddt_file,
            }
        )

    with open("designs.txt", "a") as f:
        if os.path.exists("designs.txt") and os.path.getsize("designs.txt") > 0:
            f.write("\n")
        f.write("\n".join(final_lines) + "\n")

    csv_path = "designs.csv"
    csv_columns = [
        "worker_id",
        "rank",
        "is_top",
        "sequence",
        "target_sequence",
        "binder_length",
        "ranking_loss",
        "iptm_aux",
        "bt_ipsae",
        "tb_ipsae",
        "ipsae_min",
        "bt_iptm",
        "binder_ptm",
        "plddt_aux",
        "bb_pae",
        "bt_pae_aux",
        "tb_pae",
        "intra_contact",
        "target_contact",
        "pTMEnergy",
        "iptm",
        "plddt_binder_mean",
        "plddt_binder_min",
        "plddt_binder_max",
        "plddt_binder_std",
        "plddt_target_mean",
        "plddt_target_min",
        "pae_bb_mean",
        "pae_bt_mean",
        "pae_tb_mean",
        "pae_tt_mean",
        "pae_overall_mean",
        "pae_max",
        "pdb",
        "pae_file",
        "plddt_file",
    ]
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        if write_header:
            writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n=== Run Complete ===")
    print(f"Appended {len(csv_rows)} sequences to designs.txt and designs.csv.")
    print(f"PDB files       -> {output_dir}/top*_{worker_id}.pdb")
    print(f"PAE matrices    -> {output_dir}/top*_{worker_id}_pae.npy")
    print(f"pLDDT per-res   -> {output_dir}/top*_{worker_id}_plddt.csv")
    print(f"Worker ID: {worker_id} (for tracking this run)")

    best_loss = candidates[0][1] if candidates else None
    return {"best_ranking_loss": best_loss, "n_designs": len(csv_rows)}


# ============================
# MAIN
# ============================


def main():
    print("=== Proteina-Complexa Binder Design (BindMaster non-interactive) ===\n")

    _check_gpu()
    print()

    # All parameters come from injected constants — no interactive prompts
    target_sequence = TARGET_SEQUENCE
    n_designs = N_DESIGNS
    top_k = TOP_K

    if not TARGET_PDB:
        print("ERROR: TARGET_PDB is not set. Set it to the path of the target PDB file.")
        sys.exit(1)
    if not TARGET_SEQUENCE:
        print("ERROR: TARGET_SEQUENCE is not set. Set it to the target amino acid sequence.")
        sys.exit(1)

    if MIN_LENGTH == MAX_LENGTH:
        binder_lengths = [MIN_LENGTH]
    else:
        binder_lengths = list(range(MIN_LENGTH, MAX_LENGTH + 1, LENGTH_STEP))
        if MAX_LENGTH not in binder_lengths:
            binder_lengths.append(MAX_LENGTH)

    print(f"Parameters:")
    print(
        f"  Target sequence : {target_sequence[:60]}{'...' if len(target_sequence) > 60 else ''} ({len(target_sequence)} aa)"
    )
    print(f"  Target PDB      : {TARGET_PDB}")
    print(f"  Target chain    : {TARGET_CHAIN_ID}")
    print(f"  Hotspot residues: {HOTSPOT_RESIDUES if HOTSPOT_RESIDUES else '(none — model chooses)'}")
    print(f"  Beam search runs: {n_designs}")
    print(f"  Beam width      : {BEAM_WIDTH}")
    print(f"  Branch factor   : {N_BRANCH}")
    print(f"  Checkpoints     : {STEP_CHECKPOINTS}")
    print(f"  Refold (TOP_K)  : {top_k}")
    print(f"  Inv fold samples: {INVERSE_FOLD_SAMPLES}")
    print(f"  Binder lengths  : {binder_lengths}")
    print(f"  Use MSA         : {USE_MSA}")
    print()

    _install_signal_handler(
        get_candidates_fn=lambda: _interrupt_state["candidates"],
        checkpoint_path_fn=lambda: _interrupt_state["checkpoint_path"],
    )

    summary_rows = []
    for binder_length in binder_lengths:
        output_dir = f"structures_{binder_length}aa_{n_designs}_top{top_k}"
        ckpt_path = f"checkpoint_{binder_length}aa.json"

        result = design(
            n_designs,
            top_k,
            binder_length,
            target_sequence,
            output_dir,
            checkpoint_path=ckpt_path,
            min_ranking_loss=MIN_RANKING_LOSS,
            min_hamming=MIN_HAMMING,
            min_iptm_aux=MIN_IPTM_AUX,
        )

        summary_rows.append(
            {
                "binder_length": binder_length,
                "best_ranking_loss": result["best_ranking_loss"] if result else None,
                "n_designs": result["n_designs"] if result else n_designs,
            }
        )

    if len(binder_lengths) > 1:
        _print_length_summary(summary_rows)


if __name__ == "__main__":
    main()
