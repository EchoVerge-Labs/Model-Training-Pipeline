"""report.py: aggregates reports/bench/<upstream>/metrics.json into the comparison table."""

import csv
import json
from pathlib import Path

import pytest

from pipeline import report

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_metrics(root: Path, name: str, entries: list[dict]) -> None:
    d = root / "reports" / "bench" / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"results": entries}))


def _entry(task_name: str, **metrics) -> dict:
    return {"name": task_name, "status": "ok", "metrics": metrics}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _config(workdir: Path, upstreams: dict, tasks="asr,sid") -> Path:
    import yaml

    path = workdir / "params.yaml"
    path.write_text(
        yaml.safe_dump({"benchmark": {"tasks": tasks, "seeds": "0,1", "upstreams": upstreams}})
    )
    return path


def test_the_primary_metric_is_averaged_over_languages_and_seeds(workdir):
    _write_metrics(
        workdir,
        "m1",
        [
            _entry("asr_sinhala", wer=0.6, cer=0.1),
            _entry("asr_tamil", wer=0.8, cer=0.2),
            _entry("sid_multilingual", accuracy=0.9),
        ],
    )
    metrics = report.load_upstream_metrics("m1")
    assert metrics["asr"]["mean"] == pytest.approx(0.7)
    assert metrics["sid"]["mean"] == pytest.approx(0.9)


def test_a_model_with_no_results_is_not_a_row_and_is_listed_as_not_benchmarked(workdir):
    assert report.load_upstream_metrics("never_benchmarked") is None
    _write_metrics(workdir, "has_results", [_entry("asr_tamil", wer=0.5)])
    cfg = _config(
        workdir, {"never_benchmarked": {"name": "Nobody"}, "has_results": {"name": "Somebody"}}
    )
    report.build_report(str(cfg))
    text = (workdir / "reports" / "results.md").read_text()
    assert "| Somebody |" in text and "| Nobody |" not in text
    assert "Not benchmarked yet: never_benchmarked" in text
    rows = list(csv.reader((workdir / "reports" / "results_table.csv").open()))
    assert [r[0] for r in rows] == ["upstream", "Somebody"]


def _with_protocol(workdir, name, protocol):
    _write_metrics(workdir, name, [_entry("asr_tamil", wer=0.5)])
    path = workdir / "reports" / "bench" / name / "metrics.json"
    data = json.loads(path.read_text())
    if protocol:
        data["protocol"] = protocol
    path.write_text(json.dumps(data))


def test_every_row_shows_its_protocol_and_mixed_protocols_are_flagged(workdir):
    _with_protocol(workdir, "a", "v0.5")
    _with_protocol(workdir, "b", None)  # results from before slsb recorded the protocol
    cfg = _config(workdir, {"a": {"name": "A"}, "b": {"name": "B"}})
    report.build_report(str(cfg))
    text = (workdir / "reports" / "results.md").read_text()
    assert "| A | v0.5 |" in text and "| B | unknown |" in text
    assert "different SLSB protocol versions" in text and "not comparable" in text


def test_a_single_protocol_has_no_warning_and_can_be_selected(workdir):
    _with_protocol(workdir, "a", "v0.5")
    _with_protocol(workdir, "b", "v0.4")
    cfg = _config(workdir, {"a": {"name": "A"}, "b": {"name": "B"}})
    report.build_report(str(cfg), protocol="v0.5")
    text = (workdir / "reports" / "results.md").read_text()
    assert "| A | v0.5 |" in text and "| B |" not in text
    assert "different SLSB protocol versions" not in text
    assert "Left out (not protocol v0.5): b" in text


def test_diarization_and_intent_are_reported_with_their_own_primary_metric(workdir):
    _write_metrics(
        workdir,
        "m",
        [_entry("sd_sinhala", der=0.07), _entry("ic_banking_tamil", accuracy=0.6)],
    )
    metrics = report.load_upstream_metrics("m")
    assert metrics["sd"]["mean"] == pytest.approx(0.07)
    assert metrics["ic"]["mean"] == pytest.approx(0.6)


def test_failed_entries_are_ignored(workdir):
    _write_metrics(
        workdir,
        "m1",
        [_entry("asr_tamil", wer=0.5), {"name": "asr_sinhala", "status": "error", "error": "x"}],
    )
    assert report.load_upstream_metrics("m1")["asr"]["mean"] == pytest.approx(0.5)


def test_all_models_adds_every_models_upstreams_from_its_config(workdir):
    base = _config(workdir, {"only_in_base": {"name": "Base"}})
    only_base = report.load_report_config(str(base))["benchmark"]["upstreams"]
    merged = report.load_report_config(str(base), all_models=True)["benchmark"]["upstreams"]
    assert set(only_base) == {"only_in_base"}
    assert {
        "only_in_base",
        "wavlm-large-copt200h",
        "hubert-large-copt200h",
        "wav2vec2-large-lv60-copt200h",
    } <= set(merged)
    assert (REPO_ROOT / "configs" / "models" / "wavlm.yaml").exists()
