"""Guards ported from refactor/unified-pipeline: don't benchmark/select an unfinished run,
don't log a partial SLSB run, and never silently drop a configured contamination blocklist."""

import json

import pytest

from pipeline.eval import benchmark


def _metrics(tmp_path, run, completed):
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / f"pretrain_metrics-{run}.json").write_text(
        json.dumps({"final_step": 9000, "completed": completed})
    )


def test_benchmark_refuses_an_unfinished_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _metrics(tmp_path, "r", completed=False)
    with pytest.raises(SystemExit, match="incomplete"):
        benchmark.require_complete_pretraining("r")


def test_benchmark_refuses_when_no_metrics_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="not found"):
        benchmark.require_complete_pretraining("never-trained")


def test_a_finished_run_and_allow_incomplete_both_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _metrics(tmp_path, "r", completed=True)
    benchmark.require_complete_pretraining("r")  # finished -> ok
    _metrics(tmp_path, "r", completed=False)
    benchmark.require_complete_pretraining("r", allow_incomplete=True)  # override -> ok


def test_an_explicit_upstream_skips_the_completion_gate(monkeypatch):
    """Benchmarking a baseline / arbitrary checkpoint (--upstream) is deliberate and ungated;
    runs from the repo root so params-wavlm.yaml resolves, with no metrics file present."""
    captured = {}
    monkeypatch.setattr(
        benchmark.subprocess, "call", lambda cmd: captured.setdefault("cmd", cmd) and 0 or 0
    )
    rc = benchmark.main(["--config", "params-wavlm.yaml", "--upstream", "microsoft/wavlm-large"])
    assert rc == 0 and captured["cmd"][2] == "microsoft/wavlm-large"  # ran without a metrics file


def test_failed_tasks_flags_a_partial_slsb_run(tmp_path):
    p = tmp_path / "metrics.json"
    p.write_text(
        json.dumps(
            {
                "results": [
                    {"task": "asr", "language": "tamil", "seed": 0, "status": "ok"},
                    {"task": "sid", "language": "sinhala", "seed": 0, "status": "error"},
                ]
            }
        )
    )
    assert benchmark.failed_tasks(p) == ["sid/sinhala/seed0"]


def test_missing_blocklist_warns_and_is_recorded(tmp_path, monkeypatch, capsys):
    """A configured-but-missing blocklist must warn and be flagged in the report, not be
    silently treated as empty."""
    from pipeline.common.schema import SelectConfig
    from pipeline.data.select import write_report

    cfg = SelectConfig(
        seed=1,
        target_hours=1.0,
        language_mix={"sinhala": 1.0},
        min_segment_seconds=1.0,
        max_segment_seconds=30.0,
        max_hours_per_channel=10.0,
        holdout_hours=0.1,
        catalog_path="x",
        exclude_channels_file="x",
        drive_root="x",
        drive_index_path="x",
    )
    cleaning = {
        "to_train_rows": 0,
        "exact_duplicates_removed": 0,
        "unresolved": {"ambiguous": 0, "no_match": 0, "duplicate_path": 0},
        "blocklist": {"path": "data/missing.txt", "found": False, "channels": 0},
    }
    report = tmp_path / "r.md"
    write_report(
        [], [], [], cfg, report_path=report, stats_path=tmp_path / "s.json", cleaning=cleaning
    )
    assert "MISSING (data/missing.txt)" in report.read_text()
