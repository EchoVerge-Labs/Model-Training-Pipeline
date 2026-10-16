"""Held-out loss scoring: windowing, sampling and a tiny-model CPU run."""

import json

import numpy as np
import soundfile as sf
import torch

from pipeline.heldout_loss import batches, evaluate, load_windows, sample_manifest, split_windows
from pipeline.schema import Params
from tests.test_smoke_train import _tiny_model


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


def test_evaluate_is_deterministic_and_finite(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        sf.write(tmp_path / f"{i}.wav", (rng.standard_normal(20000) * 0.1).astype("float32"), 16000)
        rows.append({"filename": f"{i}.wav"})
    data = batches(load_windows(rows, tmp_path, normalize=True), max_padded_seconds=3)
    pretrain = Params.from_yaml("params.yaml").pretrain
    model = _tiny_model()
    first = evaluate(model, data, pretrain, step=0, device=torch.device("cpu"), seed=1)
    again = evaluate(model, data, pretrain, step=0, device=torch.device("cpu"), seed=1)
    assert first == again and all(np.isfinite(v) for v in first.values())
