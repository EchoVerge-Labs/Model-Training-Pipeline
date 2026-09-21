"""Deterministic, stratified, channel-disjoint selection of training data."""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pipeline.drive_index import load_index
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
    relpath: str = ""  # <Lang>/<genre>/<source-dir>/<file> on the Drive; set by resolve_paths

    @property
    def clip_id(self) -> str:
        """Unique per clip. The sheet's name_in_drive repeats across different
        source videos, so the id also carries a hash of the (unique) relative path."""
        digest = hashlib.sha1(self.relpath.encode("utf-8")).hexdigest()[:10]
        return f"{Path(self.filename).stem}-{digest}"


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

            segments.append(
                Segment(
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
                )
            )

    return segments


def dedupe_segments(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Drop repeated rows for the same clip. The sheet logs some clips twice
    (same language / source_id / name / size, differing only in stored_date);
    counting both would double-weight them. Keeps the first occurrence."""
    seen, kept = set(), []
    for s in segments:
        key = (s.language.lower(), s.channel_id, s.filename, s.size_bytes)
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)
    return kept, len(segments) - len(kept)


def build_path_lookup(index: list[tuple[str, int]]) -> dict[tuple, list[str]]:
    """(language, genre, filename, size) -> Drive relpaths. name_in_drive alone
    isn't unique on the Drive; adding genre folder and byte size is."""
    lookup: dict[tuple, list[str]] = defaultdict(list)
    for rel, size in index:
        parts = rel.split("/")
        if len(parts) < 4:  # <Lang>/<genre>/<source-dir>/<file>
            continue
        lookup[(parts[0].lower(), parts[1], parts[-1], size)].append(rel)
    return lookup


def resolve_paths(
    segments: list[Segment], lookup: dict[tuple, list[str]]
) -> tuple[list[Segment], list[tuple[Segment, str]]]:
    """Attach each segment's Drive relpath. Rows that can't be pinned to exactly
    one file are set aside with a reason -- never guessed:
      no_match        no file with that (language, genre, name, size)
      ambiguous       several files match (same name+size in different videos)
      duplicate_path  another row already resolved to this same file
    """
    resolved, unresolved, used = [], [], set()
    for s in segments:
        candidates = lookup.get((s.language.lower(), s.genre, s.filename, s.size_bytes), [])
        if not candidates:
            unresolved.append((s, "no_match"))
        elif len(candidates) > 1:
            unresolved.append((s, "ambiguous"))
        elif candidates[0] in used:
            unresolved.append((s, "duplicate_path"))
        else:
            s.relpath = candidates[0]
            used.add(candidates[0])
            resolved.append(s)
    return resolved, unresolved


def write_unresolved(unresolved: list[tuple[Segment, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reason", "language", "genre", "name_in_drive", "size", "source_id", "title"])
        for s, reason in unresolved:
            w.writerow(
                [reason, s.language, s.genre, s.filename, s.size_bytes, s.channel_id, s.title]
            )


def select(
    segments: list[Segment],
    cfg,
    blocklist: set[str] | None = None,
) -> tuple[list[Segment], list[Segment]]:
    """Select training and holdout sets.

    Returns (train_segments, holdout_segments).
    """
    rng = random.Random(cfg.seed)
    blocklist = blocklist or set()

    # 1. Filter by duration
    filtered = [
        s
        for s in segments
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
    for lang_channels in by_lang.values():
        rng.shuffle(lang_channels)

    train_segments = []
    train_seconds_by_lang: dict[str, float] = defaultdict(float)
    target_seconds_by_lang = {
        lang: cfg.target_hours * 3600 * ratio for lang, ratio in cfg.language_mix.items()
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
    ids: set[str] = set()
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            if not s.relpath:
                raise ValueError(f"{s.filename}: no resolved Drive path -- run resolve_paths first")
            if s.clip_id in ids:
                raise ValueError(f"duplicate clip id {s.clip_id} for {s.relpath}")
            ids.add(s.clip_id)
            entry = {
                "id": s.clip_id,
                "filename": s.relpath,  # path under drive_root and under data/raw
                "name_in_drive": s.filename,
                "language": s.language,
                "duration": s.duration_seconds,
                "genre": s.genre,
                "speaking_style": s.speaking_style,
                "language_formality": s.language_formality,
                "code_switching": s.code_switching,
                "channel_id": s.channel_id,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
    cleaning: dict | None = None,
):
    """Write selection report and machine-readable stats. `cleaning` (from main)
    describes what was dropped before selection: repeats and unresolvable rows."""
    train_hours = sum(s.duration_seconds for s in train) / 3600
    holdout_hours = sum(s.duration_seconds for s in holdout) / 3600

    train_by_lang = defaultdict(float)
    for s in train:
        train_by_lang[s.language] += s.duration_seconds / 3600

    train_by_genre = defaultdict(float)
    for s in train:
        train_by_genre[s.genre] += s.duration_seconds / 3600

    train_channels = {s.channel_id for s in train}
    holdout_channels = {s.channel_id for s in holdout}

    stats = {
        "target_hours": cfg.target_hours,
        "realised_train_hours": round(train_hours, 2),
        "realised_holdout_hours": round(holdout_hours, 2),
        "shortfall_pct": round((1 - train_hours / cfg.target_hours) * 100, 1)
        if cfg.target_hours > 0
        else 0,
        "train_segments": len(train),
        "holdout_segments": len(holdout),
        "candidate_segments": len(all_segments),
        "longest_train_segment_seconds": round(
            max((s.duration_seconds for s in train), default=0.0), 1
        ),
        "cleaning": cleaning or {},
        "train_channels": len(train_channels),
        "holdout_channels": len(holdout_channels),
        "channel_overlap": len(train_channels & holdout_channels),  # should be 0
        "hours_by_language": {k: round(v, 2) for k, v in sorted(train_by_lang.items())},
        "hours_by_genre": {k: round(v, 2) for k, v in sorted(train_by_genre.items())},
    }

    # Hard fail if severely under target
    if train_hours < cfg.target_hours * 0.95:
        print(
            f"WARNING: Realised {train_hours:.1f}h vs target {cfg.target_hours}h "
            f"({stats['shortfall_pct']:.1f}% short)"
        )

    # Assert no channel overlap
    assert stats["channel_overlap"] == 0, (
        f"BUG: {stats['channel_overlap']} channels appear in both train and holdout"
    )

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Data Selection Report\n\n")
        f.write(f"Target: {cfg.target_hours}h | Realised: {train_hours:.1f}h\n")
        f.write(f"Holdout: {holdout_hours:.1f}h\n")
        f.write(f"Longest training clip: {stats['longest_train_segment_seconds']}s\n\n")
        if cleaning:
            u = cleaning["unresolved"]
            f.write("## Catalog cleaning\n\n")
            f.write(f"- to_train=yes rows: {cleaning['to_train_rows']:,}\n")
            f.write(f"- exact repeats removed: {cleaning['exact_duplicates_removed']:,}\n")
            f.write(
                f"- unresolvable, set aside (see reports/unresolved_clips.csv): "
                f"{sum(u.values())} (ambiguous {u['ambiguous']}, no_match {u['no_match']}, "
                f"duplicate_path {u['duplicate_path']})\n"
            )
            f.write(f"- candidates for selection: {len(all_segments):,}\n\n")
        f.write("## By Language\n\n")
        f.writelines(f"- {lang}: {hours:.1f}h\n" for lang, hours in sorted(train_by_lang.items()))
        f.write("\n## By Genre (top 10)\n\n")
        f.writelines(
            f"- {genre}: {hours:.1f}h\n"
            for genre, hours in sorted(train_by_genre.items(), key=lambda x: -x[1])[:10]
        )
        f.write("\n## Top 10 sources by hours\n\n")
        f.write("Check for a single source dominating the training set.\n\n")
        f.write("| Rank | source_id | Hours | % of train | Segments | Title |\n")
        f.write("|---|---|---|---|---|---|\n")
        for rank, src in enumerate(_top_sources(train, 10), start=1):
            pct = 100 * src["hours"] / train_hours if train_hours > 0 else 0.0
            f.write(
                f"| {rank} | {src['source_id']} | {src['hours']:.2f} | {pct:.1f}% | "
                f"{src['segments']} | {_md_cell(src['title'])} |\n"
            )
        f.write(
            f"\n## Channels: {len(train_channels)} train, {len(holdout_channels)} holdout, "
            f"{stats['channel_overlap']} overlap\n"
        )

    print(
        f"Selected {train_hours:.1f}h train + {holdout_hours:.1f}h holdout "
        f"from {len(all_segments)} candidates"
    )
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
    n_rows = len(segments)

    # Clean before selecting: collapse repeated rows, then pin every clip to
    # exactly one file on the Drive (see drive_index.py for why the sheet's
    # name_in_drive alone can't).
    segments, n_repeats = dedupe_segments(segments)
    index_path = Path(cfg.drive_index_path)
    if not index_path.exists():
        print(f"ERROR: {index_path} not found -- run `python -m pipeline.drive_index` first")
        sys.exit(1)
    segments, unresolved = resolve_paths(segments, build_path_lookup(load_index(index_path)))
    write_unresolved(unresolved, Path("reports/unresolved_clips.csv"))
    reasons = [reason for _, reason in unresolved]
    cleaning = {
        "to_train_rows": n_rows,
        "exact_duplicates_removed": n_repeats,
        "unresolved": {r: reasons.count(r) for r in ("ambiguous", "no_match", "duplicate_path")},
    }
    print(
        f"Catalog: {n_rows:,} to_train rows -> {n_repeats:,} repeats removed, "
        f"{len(unresolved)} unresolvable -> {len(segments):,} candidates"
    )
    if not segments:
        print("ERROR: no clip could be resolved to a file on the Drive")
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
        train,
        holdout,
        segments,
        cfg,
        report_path=Path("reports/selection_report.md"),
        stats_path=Path("reports/selection_stats.json"),
        cleaning=cleaning,
    )


if __name__ == "__main__":
    main()
