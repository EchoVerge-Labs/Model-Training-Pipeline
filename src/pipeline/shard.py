"""Shard WAV files into Lhotse Shar tarballs for efficient training IO."""
import argparse
import json
import sys
from pathlib import Path

from pipeline.schema import Params

try:
    import lhotse
    from lhotse import CutSet, Recording, MonoCut
    from lhotse.shar import SharWriter
    HAS_LHOTSE = True
except ImportError:
    HAS_LHOTSE = False


def load_manifest_as_cuts(manifest_path: str, raw_dir: str) -> "CutSet":
    """Load our JSONL manifest and convert to Lhotse CutSet."""
    cuts = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            wav_path = Path(raw_dir) / entry["filename"]
            if not wav_path.exists():
                print(f"WARNING: missing {wav_path}, skipping")
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

    return CutSet.from_cuts(cuts)


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
    for ch_id, short_cuts in by_channel.items():
        buffer = []
        buffer_dur = 0.0
        for cut in short_cuts:
            buffer.append(cut)
            buffer_dur += cut.duration
            if buffer_dur >= target_sec:
                if len(buffer) > 1:
                    try:
                        merged = buffer[0]
                        for c in buffer[1:]:
                            merged = merged.append(c)
                        long_enough.append(merged)
                    except Exception:
                        # If append fails, keep individuals if they're long enough
                        long_enough.extend(c for c in buffer if c.duration >= min_sec)
                else:
                    long_enough.extend(buffer)
                buffer = []
                buffer_dur = 0.0

        # Remaining buffer — keep if above minimum
        if buffer:
            total = sum(c.duration for c in buffer)
            if total >= min_sec:
                if len(buffer) > 1:
                    try:
                        merged = buffer[0]
                        for c in buffer[1:]:
                            merged = merged.append(c)
                        long_enough.append(merged)
                    except Exception:
                        long_enough.extend(c for c in buffer if c.duration >= min_sec)
                else:
                    long_enough.extend(buffer)

    return CutSet.from_cuts(long_enough)


def shard_cuts(cuts: "CutSet", output_dir: str, shard_size: int, audio_format: str):
    """Write CutSet to Shar tarballs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with SharWriter(
        output_dir=str(output_path),
        shard_size=shard_size,
        fields={"recording": audio_format},
    ) as writer:
        for cut in cuts:
            writer.write(cut)

    # Count shards
    n_shards = len(list(output_path.glob("*.tar")))
    total_dur = sum(c.duration for c in cuts)
    print(f"Wrote {n_shards} shards, {len(cuts)} cuts, {total_dur/3600:.1f}h total")


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

    # Concatenate short segments
    cuts = concatenate_short_cuts(cuts, cfg.min_cut_seconds, cfg.target_cut_seconds)
    print(f"After concatenation: {len(cuts)} cuts")

    shard_cuts(cuts, cfg.output_dir, cfg.shard_size, cfg.format)


if __name__ == "__main__":
    main()
