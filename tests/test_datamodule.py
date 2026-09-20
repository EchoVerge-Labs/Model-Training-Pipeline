"""datamodule: seconds-budgeted batches from Shar shards, no clip dropped or duplicated."""
import random

import numpy as np
import pytest

pytest.importorskip("lhotse")
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from lhotse import CutSet, MonoCut, Recording  # noqa: E402

from pipeline.datamodule import SAMPLE_RATE, create_dataloader, pack_batches  # noqa: E402
from pipeline.shard import shard_cuts  # noqa: E402


def _wave(seconds):
    return torch.zeros(int(seconds * SAMPLE_RATE))


def _make_shars(tmp_path, durations, shard_size):
    """One clip per duration; clip i is a constant signal of (i+1)/200, so the clip
    can be identified from its audio after batching."""
    raw = tmp_path / "raw"
    raw.mkdir()
    cuts = []
    for i, seconds in enumerate(durations):
        path = raw / f"{i}.wav"
        sf.write(path, np.full(int(seconds * SAMPLE_RATE), (i + 1) / 200, dtype="float32"), SAMPLE_RATE)
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


DURATIONS = [1.0 + (i % 6) for i in range(48)]        # 1s..6s, 48 clips


def test_pack_batches_never_drops_an_item_and_respects_the_padded_budget():
    waves = [_wave(s) for s in (1, 2, 2, 3, 5, 5, 6, 9)]
    batches = pack_batches(waves, max_padded_seconds=10.0, rng=random.Random(0))
    assert sum(len(b) for b in batches) == len(waves)
    assert all(len(b) * max(w.shape[0] for w in b) <= 10.0 * SAMPLE_RATE for b in batches)


def test_pack_batches_puts_an_over_budget_item_in_its_own_batch_rather_than_dropping_it():
    batches = pack_batches([_wave(3), _wave(20), _wave(3)], max_padded_seconds=10.0, rng=random.Random(0))
    assert sorted(len(b) for b in batches) == [1, 2]
    assert any(len(b) == 1 and b[0].shape[0] == 20 * SAMPLE_RATE for b in batches)


def test_one_epoch_yields_every_clip_exactly_once_within_the_budget(tmp_path):
    shars = _make_shars(tmp_path, DURATIONS, shard_size=6)             # 8 shards
    loader = create_dataloader(str(shars), per_device_max_seconds=12.0, num_workers=0, seed=1)
    seen, shapes = _consume(loader)
    assert sorted(seen) == list(range(len(DURATIONS)))                 # none dropped, none repeated
    assert all(padded <= 12.0 + 1e-6 for _, padded in shapes) or all(n == 1 for n, _ in shapes)
    assert max(n for n, _ in shapes) > 1                               # actually batching, not 1 clip each


def test_workers_read_disjoint_shards_so_nothing_is_duplicated(tmp_path):
    shars = _make_shars(tmp_path, DURATIONS, shard_size=6)             # 8 shards >= 2 workers
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
    shars = _make_shars(tmp_path, DURATIONS[:6], shard_size=6)         # a single shard
    with pytest.raises(ValueError, match="at least 4"):
        create_dataloader(str(shars), per_device_max_seconds=12.0, num_workers=4)
