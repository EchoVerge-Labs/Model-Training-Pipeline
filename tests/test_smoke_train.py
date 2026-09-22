"""Smoke test: 3 training steps with a tiny random model on CPU."""

import torch
from transformers import HubertConfig, HubertForPreTraining

from pipeline.train import (
    compute_mask_and_negatives,
    get_gumbel_temperature,
    get_tri_stage_lr,
    multiply_grads,
)


def _tiny_model() -> HubertForPreTraining:
    config = HubertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(32, 32),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        num_codevectors_per_group=32,
        num_codevector_groups=2,
        codevector_dim=32,
        proj_codevector_dim=32,
    )
    return HubertForPreTraining(config)


def test_tri_stage_lr():
    peak = 5e-5
    warmup = 100
    hold_ratio = 0.4
    max_updates = 1000

    # At step 0: should be 0
    assert get_tri_stage_lr(0, peak, warmup, hold_ratio, max_updates) == 0.0

    # At warmup: should be peak
    lr_warmup = get_tri_stage_lr(warmup, peak, warmup, hold_ratio, max_updates)
    assert abs(lr_warmup - peak) < 1e-10

    # During hold: should be peak
    hold_end = warmup + int(hold_ratio * (max_updates - warmup))
    lr_hold = get_tri_stage_lr(hold_end - 1, peak, warmup, hold_ratio, max_updates)
    assert abs(lr_hold - peak) < 1e-10

    # At max_updates: should be near 0
    lr_end = get_tri_stage_lr(max_updates, peak, warmup, hold_ratio, max_updates)
    assert lr_end < peak * 0.05  # at least 20x reduction


def test_mask_and_negatives_shape():
    """compute_mask_and_negatives wraps transformers' own _compute_mask_indices /
    _sample_negative_indices against a real model's feature-extractor output
    length -- exercise it against the tiny model, not a bare shape tuple."""
    model = _tiny_model()
    batch_size = 4
    input_values = torch.randn(batch_size, 16000)
    attention_mask = torch.ones(batch_size, 16000, dtype=torch.long)

    mask_time_indices, sampled_negative_indices, sub_attention_mask = compute_mask_and_negatives(
        model,
        input_values,
        attention_mask,
        mask_prob=0.65,
        mask_length=10,
        device=torch.device("cpu"),
    )

    seq_length = int(model._get_feat_extract_output_lengths(input_values.shape[-1]))
    assert mask_time_indices.shape == (batch_size, seq_length)
    assert mask_time_indices.dtype == torch.long
    # Should have some masked and some unmasked
    assert mask_time_indices.bool().any()
    assert not mask_time_indices.bool().all()

    assert sampled_negative_indices.shape[:2] == (batch_size, seq_length)
    assert sub_attention_mask.shape == (batch_size, seq_length)


def test_gumbel_temperature():
    t0 = get_gumbel_temperature(0, max_temp=1.0, min_temp=0.5, decay=0.999)
    t100 = get_gumbel_temperature(100, max_temp=1.0, min_temp=0.5, decay=0.999)
    t_big = get_gumbel_temperature(100000, max_temp=1.0, min_temp=0.5, decay=0.999)

    assert t0 == 1.0
    assert t100 < 1.0
    assert t_big == 0.5  # clamped to min


def test_smoke_forward():
    """Run 3 forward+backward steps on a tiny random HuBERT.

    mask_time_indices and sampled_negative_indices are both required --
    HubertForPreTraining.forward() returns loss=None without
    sampled_negative_indices (see its docstring: "Required input for
    pre-training"), so this exercises the same path train.py's training loop
    depends on, not just a bare forward call.
    """
    model = _tiny_model()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(3):
        # ~1 second of 16kHz audio
        x = torch.randn(2, 16000)
        mask = torch.ones(2, 16000, dtype=torch.long)

        model.set_gumbel_temperature(get_gumbel_temperature(step, 1.0, 0.5, 0.999))
        mask_time_indices, sampled_negative_indices, _ = compute_mask_and_negatives(
            model,
            x,
            mask,
            mask_prob=0.65,
            mask_length=10,
            device=torch.device("cpu"),
        )

        out = model(
            x,
            attention_mask=mask,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        )
        assert out.loss is not None
        assert out.loss.requires_grad
        out.loss.backward()

        num_losses = mask_time_indices.sum().clamp(min=1)
        multiply_grads([p for p in model.parameters() if p.requires_grad], 1.0 / num_losses)

        optimizer.step()
        optimizer.zero_grad()
