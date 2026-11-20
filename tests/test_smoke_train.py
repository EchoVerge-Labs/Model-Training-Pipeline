"""Smoke test: 3 masked-prediction training steps with a tiny random WavLM on CPU."""

import torch
from torch import nn
from transformers import WavLMConfig

from pipeline.train import compute_mask, get_tri_stage_lr
from pipeline.wavlm_model import WavLMForMaskedPrediction

NUM_CLUSTERS = 8


def _tiny_model() -> WavLMForMaskedPrediction:
    config = WavLMConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(32, 32),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
    )
    return WavLMForMaskedPrediction.from_config(config, num_clusters=NUM_CLUSTERS)


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


def test_compute_mask_shape():
    """compute_mask wraps transformers' own _compute_mask_indices against a
    real model's feature-extractor output length -- exercise it against the
    tiny model, not a bare shape tuple. Must return a bool tensor: WavLMModel
    indexes hidden_states[mask_time_indices] directly with no dtype cast."""
    model = _tiny_model()
    batch_size = 4
    input_values = torch.randn(batch_size, 16000)
    attention_mask = torch.ones(batch_size, 16000, dtype=torch.long)

    mask_time_indices, sub_attention_mask = compute_mask(
        model,
        input_values,
        attention_mask,
        mask_prob=0.65,
        mask_length=10,
        device=torch.device("cpu"),
    )

    seq_length = int(model._get_feat_extract_output_lengths(input_values.shape[-1]))
    assert mask_time_indices.shape == (batch_size, seq_length)
    assert mask_time_indices.dtype == torch.bool
    # Should have some masked and some unmasked
    assert mask_time_indices.any()
    assert not mask_time_indices.all()

    assert sub_attention_mask.shape == (batch_size, seq_length)


def test_smoke_forward():
    """Run 3 forward+backward steps on a tiny random WavLM against synthetic
    cluster-id targets -- the actual objective WavLM pretrains on (masked
    cross-entropy against k-means pseudo-labels), not wav2vec2's contrastive
    loss."""
    model = _tiny_model()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for _step in range(3):
        # ~1 second of 16kHz audio
        x = torch.randn(2, 16000)
        mask = torch.ones(2, 16000, dtype=torch.long)

        mask_time_indices, _ = compute_mask(
            model,
            x,
            mask,
            mask_prob=0.65,
            mask_length=10,
            device=torch.device("cpu"),
        )

        logits = model(x, attention_mask=mask, mask_time_indices=mask_time_indices)
        assert logits.shape == (*mask_time_indices.shape, NUM_CLUSTERS)

        target = torch.randint(0, NUM_CLUSTERS, mask_time_indices.shape)
        target[~mask_time_indices] = -100

        loss = loss_fn(logits.transpose(1, 2), target)
        assert loss.requires_grad
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
