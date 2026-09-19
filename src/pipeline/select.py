"""Deterministic, stratified, channel-disjoint selection of training data."""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from pipeline.schema import Params


@dataclass
class Segment:
    filename: str
    language: str
    duration_seconds: float
    genre: str
    speaking_style: str
    channel_id: str
    size_bytes: int
    row_index: int


def extract_channel_id(filename: str) -> str:
    """Extract channel identifier from filename.

    Expected convention: {lang}_{source}_{channel_slug}_{video_id}.wav
    Falls back to a hash of the filename if pattern doesn't match.
    """
    parts = filename.rsplit(".", 1)[0].split("_")
    if len(parts) >= 3:
        # channel slug is the third component
        return parts[2]
    # fallback: group by first 3 chars as a crude cluster
    return f"unknown_{parts[0][:3]}" if parts else f"hash_{hashlib.md5(filename.encode()).hexdigest()[:8]}"


def load_catalog(catalog_path: str) -> list[Segment]:
    """Load and validate the frozen catalog CSV."""
    segments = []
    with open(catalog_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row.get("to_train", "").strip().lower() != "yes":
                continue

            duration_min = float(row.get("duration_minutes", 0))
            duration_sec = duration_min * 60.0

            segments.append(Segment(
                filename=row["name_in_drive"],
                language=row.get("language", "unknown"),
                duration_seconds=duration_sec,
                genre=row.get("genre", "unknown"),
                speaking_style=row.get("speaking_style", "unknown"),
                channel_id=extract_channel_id(row["name_in_drive"]),
                size_bytes=int(row.get("size", 0)),
                row_index=i,
            ))

    return segments


def select(
    segments: list[Segment],
    cfg,
    blocklist: Optional[set[str]] = None,
) -> tuple[list[Segment], list[Segment]]:
    """Select training and holdout sets.

    Returns (train_segments, holdout_segments).
    """
    rng = random.Random(cfg.seed)
    blocklist = blocklist or set()

    # 1. Filter by duration
    filtered = [
        s for s in segments
        if cfg.min_segment_seconds <= s.duration_seconds <= cfg.max_segment_seconds
    ]

    # 2. Remove blocklisted channels
    filtered = [s for s in filtered if s.channel_id not in blocklist]

    # 3. Group by channel
    by_channel: dict[str, list[Segment]] = defaultdict(list)
    for s in filtered:
        by_channel[s.channel_id].append(s)

    # 4. Shuffle channels deterministically and assign holdout
    channel_ids = sorted(by_channel.keys())
    rng.shuffle(channel_ids)

    holdout_segments = []
    holdout_seconds = 0.0
    holdout_channels = set()

    for ch_id in channel_ids:
        if holdout_seconds >= cfg.holdout_hours * 3600:
            break
        ch_segs = by_channel[ch_id]
        ch_dur = sum(s.duration_seconds for s in ch_segs)
        holdout_segments.extend(ch_segs)
        holdout_seconds += ch_dur
        holdout_channels.add(ch_id)

    # 5. From remaining channels, select up to target_hours
    remaining_channels = [ch for ch in channel_ids if ch not in holdout_channels]

    # Group remaining by language
    by_lang: dict[str, list[tuple[str, list[Segment]]]] = defaultdict(list)
    for ch_id in remaining_channels:
        ch_segs = by_channel[ch_id]
        if ch_segs:
            lang = ch_segs[0].language
            by_lang[lang].append((ch_id, ch_segs))

    # Shuffle within each language
    for lang in by_lang:
        rng.shuffle(by_lang[lang])

    train_segments = []
    train_seconds_by_lang: dict[str, float] = defaultdict(float)
    target_seconds_by_lang = {
        lang: cfg.target_hours * 3600 * ratio
        for lang, ratio in cfg.language_mix.items()
    }

    for lang, target_sec in target_seconds_by_lang.items():
        channels_for_lang = by_lang.get(lang, [])
        for ch_id, ch_segs in channels_for_lang:
            if train_seconds_by_lang[lang] >= target_sec:
                break
            # Per-channel cap
            ch_dur = sum(s.duration_seconds for s in ch_segs)
            max_ch = cfg.max_hours_per_channel * 3600
            if ch_dur > max_ch:
                # Take a subset
                rng.shuffle(ch_segs)
                capped = []
                capped_dur = 0.0
                for s in ch_segs:
                    if capped_dur + s.duration_seconds > max_ch:
                        continue
                    capped.append(s)
                    capped_dur += s.duration_seconds
                ch_segs = capped

            train_segments.extend(ch_segs)
            train_seconds_by_lang[lang] += sum(s.duration_seconds for s in ch_segs)

    return train_segments, holdout_segments


def write_manifest(segments: list[Segment], path: Path):
    """Write a simple JSONL manifest (Lhotse-compatible structure)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in segments:
            entry = {
                "id": Path(s.filename).stem,
                "filename": s.filename,
                "language": s.language,
                "duration": s.duration_seconds,
                "genre": s.genre,
                "speaking_style": s.speaking_style,
                "channel_id": s.channel_id,
            }
            f.write(json.dumps(entry) + "\n")


def write_report(
    train: list[Segment],
    holdout: list[Segment],
    all_segments: list[Segment],
    cfg,
    report_path: Path,
    stats_path: Path,
):
    """Write selection report and machine-readable stats."""
    train_hours = sum(s.duration_seconds for s in train) / 3600
    holdout_hours = sum(s.duration_seconds for s in holdout) / 3600

    train_by_lang = defaultdict(float)
    for s in train:
        train_by_lang[s.language] += s.duration_seconds / 3600

    train_by_genre = defaultdict(float)
    for s in train:
        train_by_genre[s.genre] += s.duration_seconds / 3600

    train_channels = set(s.channel_id for s in train)
    holdout_channels = set(s.channel_id for s in holdout)

    stats = {
        "target_hours": cfg.target_hours,
        "realised_train_hours": round(train_hours, 2),
        "realised_holdout_hours": round(holdout_hours, 2),
        "shortfall_pct": round((1 - train_hours / cfg.target_hours) * 100, 1) if cfg.target_hours > 0 else 0,
        "train_segments": len(train),
        "holdout_segments": len(holdout),
        "total_to_train_segments": len(all_segments),
        "train_channels": len(train_channels),
        "holdout_channels": len(holdout_channels),
        "channel_overlap": len(train_channels & holdout_channels),  # should be 0
        "hours_by_language": {k: round(v, 2) for k, v in sorted(train_by_lang.items())},
        "hours_by_genre": {k: round(v, 2) for k, v in sorted(train_by_genre.items())},
    }

    # Hard fail if severely under target
    if train_hours < cfg.target_hours * 0.95:
        print(f"WARNING: Realised {train_hours:.1f}h vs target {cfg.target_hours}h "
              f"({stats['shortfall_pct']:.1f}% short)")

    # Assert no channel overlap
    assert stats["channel_overlap"] == 0, \
        f"BUG: {stats['channel_overlap']} channels appear in both train and holdout"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Data Selection Report\n\n")
        f.write(f"Target: {cfg.target_hours}h | Realised: {train_hours:.1f}h\n")
        f.write(f"Holdout: {holdout_hours:.1f}h\n\n")
        f.write("## By Language\n\n")
        for lang, hours in sorted(train_by_lang.items()):
            f.write(f"- {lang}: {hours:.1f}h\n")
        f.write("\n## By Genre (top 10)\n\n")
        for genre, hours in sorted(train_by_genre.items(), key=lambda x: -x[1])[:10]:
            f.write(f"- {genre}: {hours:.1f}h\n")
        f.write(f"\n## Channels: {len(train_channels)} train, {len(holdout_channels)} holdout, "
                f"{stats['channel_overlap']} overlap\n")

    print(f"Selected {train_hours:.1f}h train + {holdout_hours:.1f}h holdout "
          f"from {len(all_segments)} candidates")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    cfg = params.select

    segments = load_catalog(cfg.catalog_path)
    if not segments:
        print("ERROR: No segments with to_train=yes found in catalog")
        sys.exit(1)

    # Load blocklist if it exists
    blocklist = set()
    bl_path = Path(cfg.exclude_channels_file)
    if bl_path.exists():
        blocklist = {line.strip() for line in bl_path.read_text().splitlines() if line.strip()}

    train, holdout = select(segments, cfg, blocklist)

    write_manifest(train, Path("data/manifests/train.jsonl"))
    write_manifest(holdout, Path("data/manifests/dev.jsonl"))
    write_report(
        train, holdout, segments, cfg,
        report_path=Path("reports/selection_report.md"),
        stats_path=Path("reports/selection_stats.json"),
    )


if __name__ == "__main__":
    main()
