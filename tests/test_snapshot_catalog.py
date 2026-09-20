"""snapshot_catalog: language detection and column handling on the real export filenames."""
import csv

import pytest

from pipeline.snapshot_catalog import REQUIRED_COLUMNS, _language_from_name, snapshot_from_local_csv

SHEET = "Sinhala X Tamil Voice Dataset - Pre-Processed-{}.csv"


def _write_export(path, language_value, n=2, trailing_blank_column=False):
    fields = ["source_id", "source_url", "language", "title", "genre", "speaking_style",
              "speaker_count", "speaker_gender", "acoustic_condition", "language_formality",
              "accent_or_region", "code_switching", "duration_minutes", "uploaded_date",
              "name_in_drive", "size", "stored_date", "to_train", "tagged"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        header = fields + ([""] if trailing_blank_column else [])
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            row = {k: "x" for k in fields}
            row.update(language=language_value, name_in_drive=f"{language_value}_{i}.wav", to_train="Yes")
            w.writerow([row[k] for k in fields] + ([""] if trailing_blank_column else []))


def test_language_is_the_last_one_named_in_the_export_filename():
    # The Tamil export's name also contains "Sinhala" -- a plain substring test
    # labelled every Tamil clip Sinhala.
    assert _language_from_name("Sinhala X Tamil Voice Dataset - Pre-Processed-Tamil") == "tamil"
    assert _language_from_name("Sinhala X Tamil Voice Dataset - Pre-Processed-Sinhala") == "sinhala"
    assert _language_from_name("Pre-Processed-Tamil") == "tamil"


def test_language_needs_a_language_in_the_name():
    with pytest.raises(ValueError, match="cannot tell the language"):
        _language_from_name("catalog")


def test_local_csv_merge_labels_each_file_correctly_and_drops_blank_columns(tmp_path):
    si, ta = tmp_path / SHEET.format("Sinhala"), tmp_path / SHEET.format("Tamil")
    _write_export(si, "Sinhala")
    _write_export(ta, "Tamil", trailing_blank_column=True)   # the real Tamil export has this
    latest = snapshot_from_local_csv([str(si), str(ta)], tmp_path / "out")

    rows = list(csv.DictReader(open(latest, newline="", encoding="utf-8")))
    assert [(r["language"], r["name_in_drive"]) for r in rows] == [
        ("sinhala", "Sinhala_0.wav"), ("sinhala", "Sinhala_1.wav"),
        ("tamil", "Tamil_0.wav"), ("tamil", "Tamil_1.wav"),
    ]
    assert "" not in rows[0]
    assert all(c in rows[0] for c in REQUIRED_COLUMNS)


def test_rows_that_contradict_the_derived_language_are_refused(tmp_path):
    ta = tmp_path / SHEET.format("Tamil")
    _write_export(ta, "Sinhala")       # a "Tamil" export whose rows say Sinhala
    with pytest.raises(ValueError, match="derived language 'tamil'"):
        snapshot_from_local_csv([str(ta)], tmp_path / "out")
