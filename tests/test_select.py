"""Test the data selector against a mock catalog."""
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from pipeline.select import load_catalog, select, write_manifest, Segment


def _write_mock_csv(path: Path, n_si: int = 60, n_ta: int = 40):
    fields = [
        "genre", "speaking_style", "language_form", "accent_or_region",
        "code_switch", "duration_minutes", "name_in_drive", "size",
        "to_train", "tagged", "language",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n_si):
            w.writerow({
                "genre": "drama", "speaking_style": "spontaneous",
                "language_form": "spoken", "accent_or_region": "general",
                "code_switch": "no_code",
                "duration_minutes": str(round(0.1 + (i % 10) * 0.05, 2)),
                "name_in_drive": f"si_yt_ch{i % 5}_{i:04d}.wav",
                "size": str(100000 + i * 1000),
                "to_train": "yes", "tagged": "yes", "language": "sinhala",
            })
        for i in range(n_ta):
            w.writerow({
                "genre": "news", "speaking_style": "read",
                "language_form": "spoken", "accent_or_region": "general",
                "code_switch": "no_code",
                "duration_minutes": str(round(0.15 + (i % 8) * 0.04, 2)),
                "name_in_drive": f"ta_yt_ch{i % 4}_{i:04d}.wav",
                "size": str(120000 + i * 1200),
                "to_train": "yes", "tagged": "yes", "language": "tamil",
            })


def test_load_catalog(tmp_path):
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 50, 30)
    segs = load_catalog(str(csv_path))
    assert len(segs) == 80
    assert all(s.duration_seconds > 0 for s in segs)


def test_channel_disjoint_splits(tmp_path):
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 200, 150)
    segs = load_catalog(str(csv_path))

    cfg = MagicMock()
    cfg.seed = 42
    cfg.target_hours = 0.3
    cfg.language_mix = {"sinhala": 0.6, "tamil": 0.4}
    cfg.min_segment_seconds = 1.0
    cfg.max_segment_seconds = 9999.0
    cfg.max_hours_per_channel = 1.0
    cfg.holdout_hours = 0.02

    train, holdout = select(segs, cfg)
    train_ch = {s.channel_id for s in train}
    hold_ch = {s.channel_id for s in holdout}
    assert train_ch.isdisjoint(hold_ch), f"Overlap: {train_ch & hold_ch}"


def test_deterministic(tmp_path):
    """Same seed + same catalog = same selection."""
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 100, 80)

    cfg = MagicMock()
    cfg.seed = 1337
    cfg.target_hours = 0.2
    cfg.language_mix = {"sinhala": 0.6, "tamil": 0.4}
    cfg.min_segment_seconds = 1.0
    cfg.max_segment_seconds = 9999.0
    cfg.max_hours_per_channel = 1.0
    cfg.holdout_hours = 0.01

    segs1 = load_catalog(str(csv_path))
    train1, _ = select(segs1, cfg)

    segs2 = load_catalog(str(csv_path))
    train2, _ = select(segs2, cfg)

    ids1 = sorted(s.filename for s in train1)
    ids2 = sorted(s.filename for s in train2)
    assert ids1 == ids2, "Selection is not deterministic!"


def test_blocklist(tmp_path):
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 50, 30)
    segs = load_catalog(str(csv_path))

    cfg = MagicMock()
    cfg.seed = 42
    cfg.target_hours = 10.0
    cfg.language_mix = {"sinhala": 0.6, "tamil": 0.4}
    cfg.min_segment_seconds = 0.5
    cfg.max_segment_seconds = 9999.0
    cfg.max_hours_per_channel = 10.0
    cfg.holdout_hours = 0.01

    # Block channel "ch0"
    train, _ = select(segs, cfg, blocklist={"ch0"})
    for s in train:
        assert s.channel_id != "ch0", "Blocklisted channel appeared in selection"
