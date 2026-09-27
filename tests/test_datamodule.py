"""datamodule: seconds-budgeted batches from Shar shards, no clip dropped or duplicated."""

import random

import numpy as np
import pytest

pytest.importorskip("lhotse")
import soundfile as sf
import torch
from lhotse import CutSet, MonoCut, Recording

from pipeline.datamodule import SAMPLE_RATE, create_dataloader, normalize_waveform, pack_batches
from pipeline.shard import shard_cuts


def _wave(seconds):
    return torch.zeros(int(seconds * SAMPLE_RATE))


def _make_shars(tmp_path, durations, shard_size, noise_std=None):
    """One clip per duration; clip i is a constant signal of (i+1)/200, so the clip
    can be identified from its audio after batching -- or, with noise_std, Gaussian
    noise at that scale (a constant clip normalises to all zeros)."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    raw.mkdir()
    cuts = []
    for i, seconds in enumerate(durations):
        path = raw / f"{i}.wav"
        n = int(seconds * SAMPLE_RATE)
        if noise_std is None:
            audio = np.full(n, (i + 1) / 200, dtype="float32")
        else:
            audio = (0.01 + rng.normal(0, noise_std, n)).astype("float32")
        sf.write(path, audio, SAMPLE_RATE)
        rec = Recording.from_file(path, recording_id=f"c{i}")
        cuts.append(MonoCut(id=f"c{i}", start=0, duration=rec.duration, channel=0, recording=rec))
    out = tmp_path / "shars"
    shard_cuts(CutSet.from_cuts(cuts), str(out), shard_size=shard_size, audio_format="flac")
    return out


def _consume(loader):
    """-> (clip indices seen, list of (n_items, padded_seconds) per batch)."""
    seen, shapes = [], []
    for batch in loader:
        x, mask = batch["input_values"], batch["attention_mask"]
        shapes.append((x.shape[0], x.shape[1] / SAMPLE_RATE))
        for row in range(x.shape[0]):
            n = int(mask[row].sum())
            seen.append(round(float(x[row, :n].mean()) * 200) - 1)
    return seen, shapes


DURATIONS = [1.0 + (i % 6) for i in range(48)]  # 1s..6s, 48 clips


def test_pack_batches_never_drops_an_item_and_respects_the_padded_budget():
    waves = [_wave(s) for s in (1, 2, 2, 3, 5, 5, 6, 9)]
    batches = pack_batches(waves, max_padded_seconds=10.0, rng=random.Random(0))
    assert sum(len(b) for b in batches) == len(waves)
    assert all(len(b) * max(w.shape[0] for w in b) <= 10.0 * SAMPLE_RATE for b in batches)


def test_pack_batches_puts_an_over_budget_item_in_its_own_batch_rather_than_dropping_it():
    batches = pack_batches(
        [_wave(3), _wave(20), _wave(3)], max_padded_seconds=10.0, rng=random.Random(0)
    )
    assert sorted(len(b) for b in batches) == [1, 2]
    assert any(len(b) == 1 and b[0].shape[0] == 20 * SAMPLE_RATE for b in batches)


def test_one_epoch_yields_every_clip_exactly_once_within_the_budget(tmp_path):
    shars = _make_shars(tmp_path, DURATIONS, shard_size=6)  # 8 shards
    loader = create_dataloader(str(shars), per_device_max_seconds=12.0, num_workers=0, seed=1)
    seen, shapes = _consume(loader)
    assert sorted(seen) == list(range(len(DURATIONS)))  # none dropped, none repeated
    assert all(padded <= 12.0 + 1e-6 for _, padded in shapes) or all(n == 1 for n, _ in shapes)
    assert max(n for n, _ in shapes) > 1  # actually batching, not 1 clip each


def test_workers_read_disjoint_shards_so_nothing_is_duplicated(tmp_path):
    shars = _make_shars(tmp_path, DURATIONS, shard_size=6)  # 8 shards >= 2 workers
    loader = create_dataloader(str(shars), per_device_max_seconds=12.0, num_workers=2, seed=1)
    seen, _ = _consume(loader)
    assert sorted(seen) == list(range(len(DURATIONS)))


def test_each_epoch_is_reshuffled_but_covers_the_same_clips(tmp_path):
    shars = _make_shars(tmp_path, DURATIONS, shard_size=6)
    loader = create_dataloader(str(shars), per_device_max_seconds=12.0, num_workers=0, seed=1)
    first, _ = _consume(loader)
    second, _ = _consume(loader)
    assert sorted(first) == sorted(second) == list(range(len(DURATIONS)))
    assert first != second


def test_too_few_shards_for_the_workers_is_a_clear_error(tmp_path):
    shars = _make_shars(tmp_path, DURATIONS[:6], shard_size=6)  # a single shard
    with pytest.raises(ValueError, match="at least 4"):
        create_dataloader(str(shars), per_device_max_seconds=12.0, num_workers=4)


def test_normalize_waveform_matches_the_wav2vec2_feature_extractor():
    transformers = pytest.importorskip("transformers")
    w = torch.from_numpy(np.random.default_rng(0).normal(0.01, 0.066, 16000).astype("float32"))
    fe = transformers.Wav2Vec2FeatureExtractor(do_normalize=True)
    expected = fe(w.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"][0]
    torch.testing.assert_close(normalize_waveform(w), expected, atol=1e-4, rtol=1e-4)


def test_normalize_scales_each_clip_on_its_own_samples_and_leaves_padding_zero(tmp_path):
    shars = _make_shars(tmp_path, DURATIONS[:12], shard_size=6, noise_std=0.066)
    loader = create_dataloader(
        str(shars), per_device_max_seconds=12.0, num_workers=0, seed=1, normalize=True
    )
    rows = 0
    for batch in loader:
        x, mask = batch["input_values"], batch["attention_mask"]
        for row in range(x.shape[0]):
            n = int(mask[row].sum())
            assert abs(float(x[row, :n].mean())) < 1e-3
            assert abs(float(x[row, :n].std(unbiased=False)) - 1.0) < 1e-3
            assert not x[row, n:].any()
            rows += 1
    assert rows == 12
