"""datamodule: cluster labels and WavLM utterance mixing (the masked-prediction models)."""

import gzip
import json
import random

import numpy as np
import pytest

pytest.importorskip("lhotse")
import soundfile as sf
import torch
from lhotse import CutSet, MonoCut, Recording

from pipeline.datamodule import SAMPLE_RATE, create_dataloader, load_labels, pad_batch
from pipeline.shard import shard_cuts


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
        sf.write(
            path, np.full(int(seconds * SAMPLE_RATE), (i + 1) / 200, dtype="float32"), SAMPLE_RATE
        )
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


def test_pad_batch_labels_are_padded_with_ignore_index():
    waves = [torch.zeros(3), torch.zeros(5)]
    label_seqs = [[1, 2], [3, 4, 5]]
    batch = pad_batch(waves, label_seqs)
    assert batch["labels"].shape == (2, 3)
    assert batch["labels"][0].tolist() == [1, 2, -100]
    assert batch["labels"][1].tolist() == [3, 4, 5]


def test_load_labels_reads_gzip_jsonl(tmp_path):
    path = tmp_path / "labels.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "labels": [1, 2, 3]}) + "\n")
        f.write(json.dumps({"id": "b", "labels": [4, 5]}) + "\n")
    labels_by_id = load_labels(str(path))
    assert labels_by_id == {"a": [1, 2, 3], "b": [4, 5]}


def test_dataloader_attaches_labels_and_missing_id_raises_clearly(tmp_path):
    durations = [1.0, 1.0, 1.0, 1.0]
    shars = _make_shars(tmp_path, durations, shard_size=1)  # 4 shards
    labels_path = tmp_path / "labels.jsonl.gz"
    with gzip.open(labels_path, "wt", encoding="utf-8") as f:
        for i in range(len(durations)):
            f.write(json.dumps({"id": f"c{i}", "labels": [i, i, i]}) + "\n")

    loader = create_dataloader(
        str(shars), per_device_max_seconds=12.0, num_workers=0, seed=1, labels_path=str(labels_path)
    )
    for batch in loader:
        assert "labels" in batch
        assert batch["labels"].shape[0] == batch["input_values"].shape[0]

    # a labels file missing one id should fail loudly when that cut is read, not silently
    with gzip.open(labels_path, "wt", encoding="utf-8") as f:
        for i in range(len(durations) - 1):  # drop the last id
            f.write(json.dumps({"id": f"c{i}", "labels": [i, i, i]}) + "\n")
    loader = create_dataloader(
        str(shars), per_device_max_seconds=12.0, num_workers=0, seed=1, labels_path=str(labels_path)
    )
    with pytest.raises(KeyError, match="no cluster labels"):
        list(loader)


def test_normalize_uses_only_the_real_samples_and_leaves_padding_zero():
    from pipeline.datamodule import normalize_waveform, pad_batch

    short = torch.randn(4000) * 3 + 5
    long = torch.randn(8000)
    batch = pad_batch([short, long], normalize=True)
    x, mask = batch["input_values"], batch["attention_mask"]

    row = x[0, : short.shape[0]]
    assert abs(float(row.mean())) < 1e-4
    assert abs(float(row.var(unbiased=False)) - 1.0) < 1e-3
    assert torch.equal(x[0, short.shape[0] :], torch.zeros(4000))  # padding stays zero
    assert torch.allclose(row, normalize_waveform(short))
    assert int(mask[0].sum()) == 4000

    raw = pad_batch([short, long])
    assert torch.equal(raw["input_values"][0, :4000], short)  # off by default


def _rms(x):
    return float(x.pow(2).mean().sqrt())


def test_mixing_with_prob_zero_or_a_single_utterance_is_a_no_op():
    from pipeline.datamodule import mix_utterances

    waves = [torch.randn(4000), torch.randn(6000)]
    assert all(
        torch.equal(a, b) for a, b in zip(mix_utterances(waves, 0.0, random.Random(0)), waves)
    )
    assert torch.equal(mix_utterances(waves[:1], 1.0, random.Random(0))[0], waves[0])


def test_mixing_adds_one_bounded_window_of_another_utterance_at_the_right_level():
    from pipeline.datamodule import mix_utterances

    rng = random.Random(3)
    for _ in range(50):
        waves = [torch.randn(8000), torch.randn(8000) * 0.1 + 2.0, torch.randn(9000)]
        originals = [w.clone() for w in waves]
        mixed = mix_utterances(waves, 1.0, rng)

        for w, o in zip(waves, originals):  # inputs are never modified in place
            assert torch.equal(w, o)
        for m, o in zip(mixed, originals):
            assert m.shape == o.shape  # length, hence the labels' frame count, unchanged
            changed = (m != o).nonzero().flatten()
            assert len(changed) > 0
            window = int(changed[-1] - changed[0]) + 1
            assert window <= 0.5 * o.shape[0] + 1  # at most half the primary
            added = (m - o)[changed[0] : changed[-1] + 1]
            # secondary's RMS relative to the primary's is within +-5 dB
            ratio_db = 20 * torch.log10(torch.tensor(_rms(added) / _rms(o)))
            assert -5.5 <= float(ratio_db) <= 5.5


def test_mixing_never_uses_an_utterance_as_its_own_secondary():
    from pipeline.datamodule import mix_utterances

    a = torch.ones(4000)
    b = -torch.ones(4000)  # a mixed-in `a` would be positive-only, `b` negative-only
    for seed in range(20):
        mixed_a, mixed_b = mix_utterances([a, b], 1.0, random.Random(seed))
        assert (mixed_a - a).max() <= 0  # only b (negative) was added to a
        assert (mixed_b - b).min() >= 0  # only a (positive) was added to b


def test_pad_batch_mixes_after_normalising_and_keeps_padding_and_labels():
    from pipeline.datamodule import normalize_waveform, pad_batch

    short, long = torch.randn(4000) * 3 + 5, torch.randn(8000)
    labels = [[1, 2, 3], [4, 5, 6, 7]]
    batch = pad_batch([short, long], labels, True, 1.0, random.Random(0))
    assert batch["labels"].tolist() == [[1, 2, 3, -100], [4, 5, 6, 7]]  # clean labels
    assert torch.equal(batch["input_values"][0, 4000:], torch.zeros(4000))  # padding stays zero
    assert not torch.allclose(batch["input_values"][0, :4000], normalize_waveform(short))  # mixed
    assert int(batch["attention_mask"][0].sum()) == 4000
    with pytest.raises(ValueError, match="rng"):
        pad_batch([short, long], mix_prob=0.5)
