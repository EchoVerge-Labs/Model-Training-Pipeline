"""materialise: copy manifest clips from the Drive mount to data/raw, preserving paths."""
import json

from pipeline.materialise import materialise, read_filenames


def _drive(tmp_path):
    root = tmp_path / "drive"
    files = {
        "Sinhala/drama/vidA/a_000.wav": b"A" * 10,
        "Sinhala/drama/vidB/a_000.wav": b"B" * 20,      # same basename, different video
        "Tamil/news/vidC/c_000.wav": b"C" * 30,
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root, files


def test_copies_keep_the_drive_layout_so_same_named_clips_do_not_collide(tmp_path):
    root, files = _drive(tmp_path)
    raw = tmp_path / "raw"
    report = materialise(list(files), root, raw, workers=2)
    assert (report["copied"], report["skipped"], report["missing"], report["failed"]) == (3, 0, 0, 0)
    for rel, data in files.items():
        assert (raw / rel).read_bytes() == data
    assert not list(raw.rglob("*.part"))


def test_rerun_skips_files_already_present_and_recopies_wrong_sized_ones(tmp_path):
    root, files = _drive(tmp_path)
    raw = tmp_path / "raw"
    materialise(list(files), root, raw, workers=2)
    (raw / "Tamil/news/vidC/c_000.wav").write_bytes(b"truncated")        # interrupted earlier copy
    report = materialise(list(files), root, raw, workers=2)
    assert (report["copied"], report["skipped"]) == (1, 2)
    assert (raw / "Tamil/news/vidC/c_000.wav").read_bytes() == files["Tamil/news/vidC/c_000.wav"]


def test_missing_files_are_reported_not_silently_skipped(tmp_path):
    root, files = _drive(tmp_path)
    report = materialise([*files, "Sinhala/drama/vidZ/gone.wav"], root, tmp_path / "raw", workers=2)
    assert report["missing"] == 1 and report["missing_files"] == ["Sinhala/drama/vidZ/gone.wav"]
    assert report["copied"] == 3


def test_read_filenames_merges_train_and_dev_without_repeats(tmp_path):
    for name, paths in (("train", ["b.wav", "a.wav"]), ("dev", ["c.wav", "a.wav"])):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps({"filename": p}) + "\n" for p in paths), encoding="utf-8")
    assert read_filenames([tmp_path / "train.jsonl", tmp_path / "dev.jsonl"]) == ["a.wav", "b.wav", "c.wav"]
