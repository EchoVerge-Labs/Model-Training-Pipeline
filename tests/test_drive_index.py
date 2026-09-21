"""drive_index: index wavs under the Drive mount by relative path + size."""

import pytest

from pipeline.drive_index import build_index, load_index, write_index


def _touch(path, nbytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * nbytes)


def test_index_lists_wavs_with_relative_paths_and_sizes_sorted(tmp_path):
    _touch(tmp_path / "Tamil/news/vidB/b_000.wav", 20)
    _touch(tmp_path / "Sinhala/drama/vidA/a_000.wav", 10)
    _touch(tmp_path / "Sinhala/drama/vidA/clips.jsonl", 99)  # not audio
    _touch(tmp_path / "Sinhala/drama/vidC/a_000.wav", 30)  # same name, other video
    _touch(tmp_path / "Other/x/y/z.wav", 5)  # outside the language folders
    assert build_index(tmp_path) == [
        ("Sinhala/drama/vidA/a_000.wav", 10),
        ("Sinhala/drama/vidC/a_000.wav", 30),
        ("Tamil/news/vidB/b_000.wav", 20),
    ]


def test_index_round_trips_unicode_paths(tmp_path):
    entries = [("Sinhala/drama/2021_නඩගමකරය/a_000.wav", 7)]
    write_index(entries, tmp_path / "idx.tsv")
    assert load_index(tmp_path / "idx.tsv") == entries


def test_missing_language_folder_is_an_error(tmp_path):
    _touch(tmp_path / "Sinhala/drama/vidA/a.wav", 1)
    with pytest.raises(FileNotFoundError, match="Tamil"):
        build_index(tmp_path)


def test_load_rejects_a_file_that_is_not_an_index(tmp_path):
    (tmp_path / "bad.tsv").write_text("something else\n")
    with pytest.raises(ValueError, match="unexpected header"):
        load_index(tmp_path / "bad.tsv")
