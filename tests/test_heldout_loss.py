"""Held-out loss scoring: windowing, sampling and a tiny-model CPU run."""

import json

import numpy as np
import torch
from transformers import WavLMConfig

from pipeline.heldout_loss import evaluate, make_batches, sample_manifest, split_windows
from pipeline.schema import Params
from pipeline.wavlm_model import WavLMForMaskedPrediction

K = 6


def test_windows_are_equal_and_cover_the_clip():
    wave = torch.arange(16000 * 70, dtype=torch.float32)
    parts = split_windows(wave, 30.0)
    assert len(parts) == 3 and sum(p.numel() for p in parts) == wave.numel()
    assert max(p.numel() for p in parts) <= 30 * 16000
    assert len(split_windows(wave[: 16000 * 5], 30.0)) == 1


def test_sample_manifest_is_seeded_and_capped(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": i, "filename": f"{i}.wav", "duration": 60.0}) + "\n"
            for i in range(50)
        )
    )
    a, b = sample_manifest(path, 0.1, 3), sample_manifest(path, 0.1, 3)
    assert [r["id"] for r in a] == [r["id"] for r in b] and sum(r["duration"] for r in a) >= 360
    assert len(a) < 50 and [r["id"] for r in a] != [r["id"] for r in sample_manifest(path, 0.1, 4)]


def test_evaluate_is_deterministic_and_finite():
    torch.manual_seed(0)
    config = WavLMConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(32, 32),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
    )
    model = WavLMForMaskedPrediction.from_config(config, num_clusters=K)
    rng = np.random.default_rng(0)
    windows = [
        torch.from_numpy((rng.standard_normal(n) * 0.1).astype("float32"))
        for n in (16000, 20000, 24000)
    ]
    labels = [
        rng.integers(0, K, int(model._get_feat_extract_output_lengths(w.shape[0]))).tolist()
        for w in windows
    ]
    data = make_batches(windows, labels, normalize=True, max_padded_seconds=3)
    pretrain = Params.from_yaml("params.yaml").pretrain
    first = evaluate(model, data, pretrain, torch.device("cpu"), seed=1)
    again = evaluate(model, data, pretrain, torch.device("cpu"), seed=1)
    assert first == again and all(np.isfinite(v) for v in first.values())
    assert 0.0 <= first["masked_accuracy"] <= 1.0
