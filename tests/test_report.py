"""report.py turns reports/bench/<run>/metrics.json into the results table."""

import json

from pipeline.eval import report


def _bench(tmp_path, name, entries):
    d = tmp_path / "reports" / "bench" / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"results": entries}))


def test_every_task_family_slsb_runs_is_reported(tmp_path, monkeypatch):
    """ic (intent) and sd (diarization) results used to be dropped silently."""
    monkeypatch.chdir(tmp_path)
    ok = "ok"
    _bench(
        tmp_path,
        "run",
        [
            {"name": "ic_sinhala", "status": ok, "metrics": {"accuracy": 0.8}},
            {"name": "ic_tamil", "status": ok, "metrics": {"accuracy": 0.6}},
            {"name": "sd_sinhala", "status": ok, "metrics": {"der": 0.2}},
            {"name": "asr_tamil", "status": ok, "metrics": {"wer": 0.5}},
            {"name": "ic_bad", "status": "failed", "metrics": {"accuracy": 0.0}},
        ],
    )
    got = report.load_upstream_metrics("run")
    assert got["ic"]["mean"] == 0.7 and got["sd"]["mean"] == 0.2 and got["asr"]["mean"] == 0.5


def test_unbenchmarked_upstream_is_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert report.load_upstream_metrics("nothing") is None
