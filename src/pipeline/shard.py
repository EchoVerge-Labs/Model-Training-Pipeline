"""Shard WAV files into Lhotse Shar tarballs for efficient training IO."""

import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path

from pipeline.schema import Params

try:
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.cut import MixedCut
    from lhotse.shar import SharWriter

    HAS_LHOTSE = True
except ImportError:
    HAS_LHOTSE = False


def load_manifest_as_cuts(manifest_path: str, raw_dir: str) -> "CutSet":
    """Load our JSONL manifest and convert to Lhotse CutSet.

    entry["filename"] is the clip's path under raw_dir (<Lang>/<genre>/<source-dir>/<file>)
    and entry["id"] is unique per clip. A missing file or a repeated id is an
    error, not a warning: sharding a silently partial or ambiguous dataset would
    only surface much later, mid-training.
    """
    cuts, seen_ids, missing = [], set(), []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] in seen_ids:
                raise ValueError(f"duplicate clip id in {manifest_path}: {entry['id']}")
            seen_ids.add(entry["id"])
            wav_path = Path(raw_dir) / entry["filename"]
            if not wav_path.exists():
                missing.append(entry["filename"])
                continue

            recording = Recording.from_file(wav_path, recording_id=entry["id"])
            cut = MonoCut(
                id=entry["id"],
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
                custom={
                    "language": entry.get("language", "unknown"),
                    "genre": entry.get("genre", "unknown"),
                    "channel_id": entry.get("channel_id", "unknown"),
                },
            )
            cuts.append(cut)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(seen_ids)} clips in {manifest_path} are not under {raw_dir} "
            f"(first: {missing[0]}) -- run the materialise step"
        )
    return CutSet.from_cuts(cuts)


def split_long_cuts(cuts: "CutSet", max_sec: float) -> "CutSet":
    """Split every cut longer than max_sec into ceil(duration/max_sec) EQUAL windows.

    Equal windows (70s -> 3 x 23.3s, not 30 + 30 + 10) mean no window is shorter
    than max_sec/2, so none falls below the minimum cut length. Bounds each
    clip so one batch item can't blow the per-device seconds budget, while
    keeping all the audio. Window ids are derived from the source id, so
    they're stable across runs.
    """
    out = []
    for cut in cuts:
        if cut.duration <= max_sec:
            out.append(cut)
            continue
        n = math.ceil(cut.duration / max_sec)
        piece = cut.duration / n
        for i in range(n):
            dur = cut.duration - i * piece if i == n - 1 else piece  # last window: no float overrun
            out.append(cut.truncate(offset=i * piece, duration=dur).with_id(f"{cut.id}-p{i}"))
    return CutSet.from_cuts(out)


def shuffle_cuts(cuts: "CutSet", seed: int) -> "CutSet":
    """Seeded shuffle, so each shard mixes many sources. The manifest lists
    clips grouped by source; unshuffled, a shard (and so a training batch) would
    hold consecutive clips from the same video and speaker."""
    return cuts.shuffle(rng=random.Random(seed))


def _merge_buffer(buffer: list, min_sec: float) -> list:
    """Concatenate buffer's cuts in time into one cut with a deterministic id
    (Lhotse would otherwise give it a random UUID, making shards irreproducible).
    If merging fails, fall back to keeping the individually long-enough cuts."""
    if len(buffer) == 1:
        return buffer
    try:
        merged = buffer[0]
        for c in buffer[1:]:
            merged = merged.append(c)
        return [merged.with_id(f"concat-{buffer[0].id}-x{len(buffer)}")]
    except Exception as e:  # noqa: BLE001 -- Lhotse's append can raise assorted errors; fall back and warn
        print(f"WARNING: could not merge {len(buffer)} cuts starting at {buffer[0].id}: {e}")
        return [c for c in buffer if c.duration >= min_sec]


def concatenate_short_cuts(cuts: "CutSet", min_sec: float, target_sec: float) -> "CutSet":
    """Concatenate short cuts from the same channel into longer windows."""
    # Group by channel
    from collections import defaultdict

    by_channel = defaultdict(list)
    long_enough = []

    for cut in cuts:
        if cut.duration >= min_sec:
            long_enough.append(cut)
        else:
            ch = cut.custom.get("channel_id", "unknown")
            by_channel[ch].append(cut)

    # Concatenate shorts within each channel
    for short_cuts in by_channel.values():
        buffer = []
        buffer_dur = 0.0
        for cut in short_cuts:
            buffer.append(cut)
            buffer_dur += cut.duration
            if buffer_dur >= target_sec:
                long_enough.extend(_merge_buffer(buffer, min_sec))
                buffer = []
                buffer_dur = 0.0

        # Remaining buffer — keep if above minimum
        if buffer:
            total = sum(c.duration for c in buffer)
            if total >= min_sec:
                long_enough.extend(_merge_buffer(buffer, min_sec))

    return CutSet.from_cuts(long_enough)


def shard_cuts(cuts: "CutSet", output_dir: str, shard_size: int, audio_format: str):
    """Write CutSet to Shar tarballs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Merged cuts are MixedCuts, which SharWriter can't write (they have no single
    # `recording`). Render each to a temp audio file, giving a plain single-recording
    # cut; the temp dir only has to outlive the writes.
    with (
        tempfile.TemporaryDirectory(prefix="shar-concat-") as scratch,
        SharWriter(
            output_dir=str(output_path),
            shard_size=shard_size,
            fields={"recording": audio_format},
        ) as writer,
    ):
        for cut in cuts:
            if isinstance(cut, MixedCut):
                cut = cut.save_audio(
                    Path(scratch) / f"{cut.id}.{audio_format}", format=audio_format
                )
            writer.write(cut)

    # Count shards
    n_shards = len(list(output_path.glob("*.tar")))
    total_dur = sum(c.duration for c in cuts)
    print(f"Wrote {n_shards} shards, {len(cuts)} cuts, {total_dur / 3600:.1f}h total")


def main():
    if not HAS_LHOTSE:
        print("ERROR: lhotse is not installed. Install with: pip install lhotse[webdataset]")
        print("On ARM64 DGX Spark, install inside the NGC PyTorch container.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    cfg = params.shard

    cuts = load_manifest_as_cuts("data/manifests/train.jsonl", "data/raw")
    print(f"Loaded {len(cuts)} cuts")

    # Split long clips first, then concatenate short ones
    n_long = sum(1 for c in cuts if c.duration > cfg.max_cut_seconds)
    cuts = split_long_cuts(cuts, cfg.max_cut_seconds)
    print(f"After splitting {n_long} clips longer than {cfg.max_cut_seconds}s: {len(cuts)} cuts")
    cuts = concatenate_short_cuts(cuts, cfg.min_cut_seconds, cfg.target_cut_seconds)
    print(f"After concatenation: {len(cuts)} cuts")

    cuts = shuffle_cuts(cuts, params.select.seed)

    longest = max(c.duration for c in cuts)
    if longest > cfg.max_cut_seconds + 0.01:
        raise RuntimeError(
            f"a {longest:.1f}s cut survived splitting (max_cut_seconds={cfg.max_cut_seconds})"
        )
    print(f"Longest cut: {longest:.1f}s")

    shard_cuts(cuts, cfg.output_dir, cfg.shard_size, cfg.format)


if __name__ == "__main__":
    main()
