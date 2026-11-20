"""Learning-rate schedule shared by every trainer."""

# ─── Tri-stage LR schedule ──────────────────────────────────────────────
#
# Phase 1: linear warmup from 0 → peak_lr         (0 → warmup_updates)
# Phase 2: hold at peak_lr                        (warmup_updates → hold_end)
# Phase 3: exponential decay from peak_lr → 0     (hold_end → max_updates)
#
# hold_end = warmup_updates + hold_ratio * (max_updates - warmup_updates)
#
# This is fairseq's tri_stage scheduler. fairseq uses it for wav2vec2/HuBERT/WavLM
# *fine-tuning*; their *pre-training* configs use polynomial decay with a long warmup.
# It is kept here as a deliberate CoPT choice (short warmup, a hold, then decay to ~1%).


def get_tri_stage_lr(
    step: int,
    peak_lr: float,
    warmup_updates: int,
    hold_ratio: float,
    max_updates: int,
) -> float:
    """Compute learning rate at a given step using the tri-stage schedule."""
    hold_updates = int(hold_ratio * (max_updates - warmup_updates))
    hold_end = warmup_updates + hold_updates
    decay_updates = max_updates - hold_end

    if step < warmup_updates:
        # Phase 1: linear warmup
        if warmup_updates == 0:
            return peak_lr
        return peak_lr * step / warmup_updates

    elif step < hold_end:
        # Phase 2: hold
        return peak_lr

    else:
        # Phase 3: exponential decay
        if decay_updates == 0:
            return peak_lr
        decay_step = step - hold_end
        # Exponential decay to ~0 at max_updates
        # lr = peak_lr * gamma^decay_step, where gamma chosen so final lr ≈ peak_lr * 0.01
        gamma = (0.01) ** (1.0 / max(decay_updates, 1))
        return peak_lr * (gamma**decay_step)
