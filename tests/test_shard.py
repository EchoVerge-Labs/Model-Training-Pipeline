"""shard: manifest -> cuts -> Lhotse Shar, on synthetic audio (needs lhotse)."""

import json

import numpy as np
import pytest

pytest.importorskip("lhotse")
import soundfile as sf
from lhotse import CutSet

from pipeline.shard import (
    concatenate_short_cuts,
    load_manifest_as_cuts,
    shard_cuts,
    shuffle_cuts,
    split_long_cuts,
)


def _make(tmp_path, clips):
    """clips: [(id, relpath, seconds, channel_id)] -> writes wavs + manifest, returns manifest path."""
    raw = tmp_path / "raw"
    manifest = tmp_path / "train.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for cid, rel, seconds, channel in clips:
            (raw / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                raw / rel, (np.random.randn(int(16000 * seconds)) * 0.1).astype("float32"), 16000
            )
            f.write(
                json.dumps(
                    {
                        "id": cid,
                        "filename": rel,
                        "language": "sinhala",
                        "genre": "drama",
                        "channel_id": channel,
                    }
                )
                + "\n"
            )
    return manifest, raw


def test_manifest_to_cuts_to_shar_round_trip(tmp_path):
    manifest, raw = _make(
        tmp_path,
        [
            ("long-1", "Sinhala/drama/vidA/a.wav", 6.0, "src1"),
            ("long-2", "Sinhala/drama/vidB/a.wav", 7.0, "src2"),  # same basename as long-1
            *[(f"short-{i}", f"Sinhala/drama/vidC/s{i}.wav", 1.0, "src3") for i in range(6)],
        ],
    )
    cuts = load_manifest_as_cuts(str(manifest), str(raw))
    assert len(cuts) == 8

    cuts = concatenate_short_cuts(
        cuts, min_sec=5.0, target_sec=15.0
    )  # 6 x 1s of src3 -> one 6s cut
    assert len(cuts) == 3
    assert sorted(round(c.duration) for c in cuts) == [6, 6, 7]

    out = tmp_path / "shars"
    shard_cuts(cuts, str(out), shard_size=2, audio_format="flac")
    assert len(list(out.glob("*.tar"))) >= 1

    back = CutSet.from_shar(in_dir=str(out))
    assert len(back) == 3
    assert sorted(round(c.duration) for c in back) == [6, 6, 7]
    # the merged cut got a deterministic id (not a random UUID) and its own audio
    assert sorted(c.id for c in back if c.id.startswith("concat-")) == ["concat-short-0-x6"]
    assert all(c.load_audio().shape[-1] > 0 for c in back)


def test_repeated_clip_id_is_an_error(tmp_path):
    manifest, raw = _make(
        tmp_path,
        [
            ("dup", "Sinhala/drama/vidA/a.wav", 1.0, "s"),
            ("dup", "Sinhala/drama/vidB/a.wav", 1.0, "s"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate clip id"):
        load_manifest_as_cuts(str(manifest), str(raw))


def test_missing_file_is_an_error_not_a_warning(tmp_path):
    manifest, raw = _make(tmp_path, [("a", "Sinhala/drama/vidA/a.wav", 1.0, "s")])
    (raw / "Sinhala/drama/vidA/a.wav").unlink()
    with pytest.raises(FileNotFoundError, match="materialise"):
        load_manifest_as_cuts(str(manifest), str(raw))


def test_long_clips_are_split_into_equal_windows_no_larger_than_the_cap(tmp_path):
    manifest, raw = _make(
        tmp_path,
        [
            ("ok", "Sinhala/drama/vidA/ok.wav", 20.0, "s1"),  # under the cap: untouched
            ("long", "Sinhala/drama/vidB/long.wav", 70.0, "s2"),  # -> 3 windows of ~23.3s
            ("edge", "Sinhala/drama/vidC/edge.wav", 31.0, "s3"),  # just over: 2 x 15.5s, not 30 + 1
        ],
    )
    cuts = split_long_cuts(load_manifest_as_cuts(str(manifest), str(raw)), max_sec=30.0)

    by_id = {c.id: c for c in cuts}
    assert sorted(by_id) == ["edge-p0", "edge-p1", "long-p0", "long-p1", "long-p2", "ok"]
    assert all(c.duration <= 30.0 for c in cuts)
    assert min(c.duration for c in cuts if c.id.startswith("edge")) > 15.0  # no tiny remainder
    assert sum(c.duration for c in cuts if c.id.startswith("long")) == pytest.approx(70.0, abs=1e-3)
    assert by_id["long-p1"].custom["channel_id"] == "s2"  # metadata carried over

    # windows are consecutive, non-overlapping slices of the source
    starts = sorted((c.start, c.duration) for c in cuts if c.id.startswith("long"))
    assert starts[1][0] == pytest.approx(starts[0][0] + starts[0][1], abs=1e-3)
    assert starts[2][0] == pytest.approx(starts[1][0] + starts[1][1], abs=1e-3)


def test_split_windows_survive_sharding_with_the_right_audio(tmp_path):
    manifest, raw = _make(tmp_path, [("long", "Sinhala/drama/vidB/long.wav", 70.0, "s2")])
    cuts = split_long_cuts(load_manifest_as_cuts(str(manifest), str(raw)), max_sec=30.0)
    out = tmp_path / "shars"
    shard_cuts(cuts, str(out), shard_size=10, audio_format="flac")

    back = CutSet.from_shar(in_dir=str(out))
    assert sorted(c.id for c in back) == ["long-p0", "long-p1", "long-p2"]
    total_samples = 0
    for c in back:
        n = c.load_audio().shape[-1]
        assert n == pytest.approx(
            c.duration * 16000, abs=2
        )  # audio matches the window, not the whole file
        total_samples += n
    assert total_samples == pytest.approx(70 * 16000, abs=8)  # nothing lost, nothing duplicated


def test_shuffle_is_seeded_and_mixes_sources_without_losing_clips(tmp_path):
    clips = [(f"a{i}", f"Sinhala/drama/vidA/a{i}.wav", 1.0, "srcA") for i in range(8)] + [
        (f"b{i}", f"Sinhala/drama/vidB/b{i}.wav", 1.0, "srcB") for i in range(8)
    ]
    manifest, raw = _make(tmp_path, clips)
    cuts = load_manifest_as_cuts(str(manifest), str(raw))  # grouped: all A, then all B

    shuffled = shuffle_cuts(cuts, seed=7)
    assert sorted(c.id for c in shuffled) == sorted(c.id for c in cuts)
    assert [c.id for c in shuffled] == [c.id for c in shuffle_cuts(cuts, seed=7)]  # reproducible
    assert [c.id for c in shuffled] != [c.id for c in shuffle_cuts(cuts, seed=8)]
    first_shard = [c.custom["channel_id"] for c in list(shuffled)[:8]]
    assert set(first_shard) == {"srcA", "srcB"}  # no longer one source per shard
