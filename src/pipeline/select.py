"""Deterministic, stratified, channel-disjoint selection of training data."""
import argparse
import csv
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
    language_formality: str
    code_switching: str
    channel_id: str  # the catalog's source_id (one YouTube source, many files)
    title: str
    source_url: str
    size_bytes: int
    row_index: int


def load_catalog(catalog_path: str) -> list[Segment]:
    """Load the frozen catalog CSV, keeping only rows marked to_train=yes.

    source_id is required and used directly as the channel/source identifier
    (channel-disjoint train/holdout splitting depends on it); a to_train row
    with a blank source_id raises rather than silently lumping every such row
    into one giant "channel".
    """
    segments = []
    with open(catalog_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row.get("to_train", "").strip().lower() != "yes":
                continue

            source_id = row["source_id"].strip()
            if not source_id:
                raise ValueError(
                    f"{catalog_path}: row {i} ({row.get('name_in_drive')!r}) is marked "
                    f"to_train=yes but has an empty source_id"
                )

            duration_min = float(row.get("duration_minutes", 0))
            duration_sec = duration_min * 60.0

            segments.append(Segment(
                filename=row["name_in_drive"],
                language=row.get("language", "unknown"),
                duration_seconds=duration_sec,
                genre=row.get("genre", "unknown"),
                speaking_style=row.get("speaking_style", "unknown"),
                language_formality=row.get("language_formality", "unknown"),
                code_switching=row.get("code_switching", "unknown"),
                channel_id=source_id,
                title=row.get("title", ""),
                source_url=row.get("source_url", ""),
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
                "language_formality": s.language_formality,
                "code_switching": s.code_switching,
                "channel_id": s.channel_id,
            }
            f.write(json.dumps(entry) + "\n")


def _top_sources(segments: list[Segment], n: int) -> list[dict]:
    """Group segments by source_id; return the n sources with the most hours.

    A source's title is its first non-empty title in catalog order (stable, so
    the report is reproducible)."""
    by_source: dict[str, dict] = {}
    for s in segments:
        entry = by_source.setdefault(
            s.channel_id, {"source_id": s.channel_id, "hours": 0.0, "segments": 0, "title": ""}
        )
        entry["hours"] += s.duration_seconds / 3600
        entry["segments"] += 1
        if not entry["title"] and s.title:
            entry["title"] = s.title
    return sorted(by_source.values(), key=lambda e: (-e["hours"], e["source_id"]))[:n]


def _md_cell(text: str) -> str:
    """Make free text (YouTube titles) safe inside a markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


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
        f.write("\n## Top 10 sources by hours\n\n")
        f.write("Check for a single source dominating the training set.\n\n")
        f.write("| Rank | source_id | Hours | % of train | Segments | Title |\n")
        f.write("|---|---|---|---|---|---|\n")
        for rank, src in enumerate(_top_sources(train, 10), start=1):
            pct = 100 * src["hours"] / train_hours if train_hours > 0 else 0.0
            f.write(f"| {rank} | {src['source_id']} | {src['hours']:.2f} | {pct:.1f}% | "
                    f"{src['segments']} | {_md_cell(src['title'])} |\n")
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
