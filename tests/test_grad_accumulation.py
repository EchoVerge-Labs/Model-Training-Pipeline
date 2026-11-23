"""Contrastive gradient accumulation: one update over several micro-batches must use ALL of
them, normalised once by the masked-position count of the whole update.

The contrastive loss is an unnormalised sum over masked positions. Rescaling the gradient
after every micro-batch (the HF example's per-step recipe) shrinks the micro-batches already
accumulated, so only the last one would count. This compares the trainer's logged gradient
norm against an independent reference computed from the raw per-micro-batch gradients.
"""

import math

import torch
from transformers import Wav2Vec2ForPreTraining

from pipeline.train_contrastive import (
    compute_mask_and_negatives,
    get_gumbel_temperature,
)
from tests.test_resume import DEVICE, _batches, _make_base, _params, _run

ACCUM = 4


def _reference_grad_norm(base, cfg, batches):
    """Norm of (sum of raw micro-batch gradients) / (total masked positions)."""
    model = Wav2Vec2ForPreTraining.from_pretrained(base).to(DEVICE)
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()
    model.train()
    total_masked = 0.0
    for batch in batches:
        x = batch["input_values"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        model.set_gumbel_temperature(
            get_gumbel_temperature(
                0,
                cfg.max_gumbel_temperature,
                cfg.min_gumbel_temperature,
                cfg.gumbel_temperature_decay,
            )
        )
        masked, negatives, _ = compute_mask_and_negatives(
            model, x, mask, cfg.mask_time_prob, cfg.mask_time_length, DEVICE
        )
        out = model(
            x,
            attention_mask=mask,
            mask_time_indices=masked,
            sampled_negative_indices=negatives,
        )
        out.loss.backward()  # raw sum: gradients add up in .grad
        total_masked += float(masked.sum())
    grads = [p.grad / total_masked for p in model.parameters() if p.grad is not None]
    return math.sqrt(sum(float(g.pow(2).sum()) for g in grads))


def _seed():
    import numpy as np

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def test_one_update_over_several_micro_batches_uses_every_micro_batch(tmp_path):
    base = _make_base(tmp_path)
    # target_batch_seconds / per_device_max_seconds = 4 micro-batches per update
    params = _params(
        tmp_path,
        base,
        max_updates=1,
        warmup_updates=1,
        target_batch_seconds=2.0 * ACCUM,
        per_device_max_seconds=2.0,
        precision="fp32",
    )
    cfg = params.pretrain
    batches = _batches(ACCUM)

    # Deterministic tiny model: no dropout / layerdrop, so the only randomness is the mask
    # sampling and Gumbel noise, which both runs draw in the same order from the same seed.
    model = Wav2Vec2ForPreTraining.from_pretrained(base)
    model.config.hidden_dropout = 0.0
    model.config.activation_dropout = 0.0
    model.config.attention_dropout = 0.0
    model.config.feat_proj_dropout = 0.0
    model.config.feat_quantizer_dropout = 0.0
    model.config.layerdrop = 0.0
    model.save_pretrained(base)

    _seed()
    reference = _reference_grad_norm(base, cfg, batches)

    _seed()
    _run(params, tmp_path)
    rows = (tmp_path / "reports" / "pretrain_curves-xlsr.csv").read_text().splitlines()
    logged = float(rows[-1].split(",")[5])  # grad_norm: the normalised gradient, pre-clipping

    assert logged == __import__("pytest").approx(reference, rel=2e-2, abs=1e-3), (
        f"trainer grad_norm {logged} != reference {reference}: micro-batch gradients are not "
        "being accumulated and normalised once per update"
    )
