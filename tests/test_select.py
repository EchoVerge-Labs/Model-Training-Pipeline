"""Test the data selector against a mock catalog."""
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

from pipeline.select import Segment, load_catalog, select, write_report

# The actual catalog schema (Google Sheet tabs, plus `language` as set by
# snapshot_catalog.py from the tab name).
CATALOG_FIELDS = [
    "source_id", "source_url", "language", "title", "genre", "speaking_style",
    "speaker_count", "speaker_gender", "acoustic_condition", "language_formality",
    "accent_or_region", "code_switching", "duration_minutes", "uploaded_date",
    "name_in_drive", "size", "stored_date", "to_train", "tagged",
]


def _row(**overrides) -> dict:
    row = {f: "" for f in CATALOG_FIELDS}
    row.update({
        "speaker_count": "1", "speaker_gender": "mixed", "acoustic_condition": "clean",
        "accent_or_region": "general", "uploaded_date": "2025-01-01",
        "stored_date": "2025-02-01", "to_train": "yes", "tagged": "yes",
    })
    row.update(overrides)
    return row


def _write_mock_csv(path: Path, n_si: int = 60, n_ta: int = 40):
    """Filenames deliberately carry no channel information (no underscores):
    channel identity must come from the source_id column, not the filename."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        w.writeheader()
        for i in range(n_si):
            src = f"si_src{i % 5}"
            w.writerow(_row(
                source_id=src, source_url=f"https://youtube.com/watch?v={src}",
                language="sinhala", title=f"Sinhala video {i % 5}",
                genre="drama", speaking_style="spontaneous",
                language_formality="spoken", code_switching="no_code",
                duration_minutes=str(round(0.1 + (i % 10) * 0.05, 2)),
                name_in_drive=f"rec{i:04d}.wav", size=str(100000 + i * 1000),
            ))
        for i in range(n_ta):
            src = f"ta_src{i % 4}"
            w.writerow(_row(
                source_id=src, source_url=f"https://youtube.com/watch?v={src}",
                language="tamil", title=f"Tamil video {i % 4}",
                genre="news", speaking_style="read",
                language_formality="spoken", code_switching="no_code",
                duration_minutes=str(round(0.15 + (i % 8) * 0.04, 2)),
                name_in_drive=f"rec{n_si + i:04d}.wav", size=str(120000 + i * 1200),
            ))


def _cfg(**overrides):
    cfg = MagicMock()
    cfg.seed = 42
    cfg.target_hours = 0.3
    cfg.language_mix = {"sinhala": 0.5, "tamil": 0.5}
    cfg.min_segment_seconds = 1.0
    cfg.max_segment_seconds = 9999.0
    cfg.max_hours_per_channel = 1.0
    cfg.holdout_hours = 0.02
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_load_catalog(tmp_path):
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 50, 30)
    segs = load_catalog(str(csv_path))
    assert len(segs) == 80
    assert all(s.duration_seconds > 0 for s in segs)


def test_load_catalog_uses_source_id_and_new_columns(tmp_path):
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 50, 30)
    segs = load_catalog(str(csv_path))

    # channel_id comes straight from source_id, not from the filename
    assert {s.channel_id for s in segs} == (
        {f"si_src{i}" for i in range(5)} | {f"ta_src{i}" for i in range(4)}
    )
    first = segs[0]
    assert first.filename == "rec0000.wav"
    assert first.channel_id == "si_src0"
    assert first.title == "Sinhala video 0"
    assert first.source_url == "https://youtube.com/watch?v=si_src0"
    assert first.code_switching == "no_code"
    assert first.language_formality == "spoken"


def test_load_catalog_rejects_blank_source_id(tmp_path):
    csv_path = tmp_path / "cat.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        w.writeheader()
        w.writerow(_row(source_id="ok_src", language="sinhala", duration_minutes="0.1",
                        name_in_drive="a.wav", size="1"))
        w.writerow(_row(source_id="  ", language="sinhala", duration_minutes="0.1",
                        name_in_drive="b.wav", size="1"))
    try:
        load_catalog(str(csv_path))
    except ValueError as e:
        assert "empty source_id" in str(e) and "b.wav" in str(e)
    else:
        raise AssertionError("expected ValueError for blank source_id")


def test_load_catalog_ignores_blank_source_id_when_not_to_train(tmp_path):
    csv_path = tmp_path / "cat.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        w.writeheader()
        w.writerow(_row(source_id="", to_train="no", language="sinhala",
                        duration_minutes="0.1", name_in_drive="a.wav", size="1"))
    assert load_catalog(str(csv_path)) == []


def test_channel_disjoint_splits(tmp_path):
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 200, 150)
    segs = load_catalog(str(csv_path))

    train, holdout = select(segs, _cfg())
    train_ch = {s.channel_id for s in train}
    hold_ch = {s.channel_id for s in holdout}
    assert holdout, "expected a non-empty holdout"
    assert train_ch.isdisjoint(hold_ch), f"Overlap: {train_ch & hold_ch}"


def test_deterministic(tmp_path):
    """Same seed + same catalog = same selection."""
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 100, 80)
    cfg = _cfg(seed=1337, target_hours=0.2, holdout_hours=0.01)

    train1, _ = select(load_catalog(str(csv_path)), cfg)
    train2, _ = select(load_catalog(str(csv_path)), cfg)

    ids1 = sorted(s.filename for s in train1)
    ids2 = sorted(s.filename for s in train2)
    assert ids1 == ids2, "Selection is not deterministic!"


def test_blocklist(tmp_path):
    csv_path = tmp_path / "cat.csv"
    _write_mock_csv(csv_path, 50, 30)
    segs = load_catalog(str(csv_path))

    cfg = _cfg(target_hours=10.0, min_segment_seconds=0.5,
               max_hours_per_channel=10.0, holdout_hours=0.01)

    train, holdout = select(segs, cfg, blocklist={"si_src0"})
    assert train, "expected a non-empty selection"
    for s in train + holdout:
        assert s.channel_id != "si_src0", "Blocklisted source appeared in selection"


def _seg(source_id: str, title: str, seconds: float, filename: str) -> Segment:
    return Segment(
        filename=filename, language="sinhala", duration_seconds=seconds, genre="drama",
        speaking_style="read", language_formality="spoken", code_switching="no_code",
        channel_id=source_id, title=title, source_url="", size_bytes=1, row_index=0,
    )


def test_report_lists_top_sources_with_titles(tmp_path):
    # "big" dominates (3h of 3.5h); titles include a pipe and a newline, which
    # would break a markdown table if not escaped/flattened.
    train = (
        [_seg("big", "Big | Show\nEp 1", 3600, f"b{i}.wav") for i in range(3)]
        + [_seg("small", "Small show", 1800, "s0.wav")]
    )
    report_path = tmp_path / "report.md"
    stats_path = tmp_path / "stats.json"
    write_report(train, [], train, _cfg(target_hours=3.5), report_path, stats_path)

    report = report_path.read_text()
    assert "## Top 10 sources by hours" in report
    rows = [ln for ln in report.splitlines() if ln.startswith("| 1 ") or ln.startswith("| 2 ")]
    assert len(rows) == 2
    assert "| 1 | big | 3.00 | 85.7% | 3 | Big \\| Show Ep 1 |" == rows[0]
    assert "| 2 | small | 0.50 | 14.3% | 1 | Small show |" == rows[1]
    assert json.loads(stats_path.read_text())["train_channels"] == 2


def test_report_top_sources_capped_at_ten(tmp_path):
    train = [_seg(f"src{i:02d}", f"Title {i}", 3600 - i, f"f{i}.wav") for i in range(15)]
    report_path = tmp_path / "report.md"
    write_report(train, [], train, _cfg(target_hours=15.0), report_path, tmp_path / "s.json")

    table_rows = [ln for ln in report_path.read_text().splitlines()
                  if ln.startswith("| ") and ln.split("|")[1].strip().isdigit()]
    assert len(table_rows) == 10
    assert "src00" in table_rows[0] and "src14" not in "".join(table_rows)
