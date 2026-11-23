"""select_checkpoint: one selector for all five models."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from pipeline import select_checkpoint
from pipeline.select_checkpoint import find_milestone_checkpoints


def _backbone(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"")


def test_milestones_need_only_a_backbone_so_contrastive_checkpoints_qualify(tmp_path):
    for step in (4500, 9000):
        _backbone(tmp_path / f"checkpoint-{step}")  # no masked-prediction head, as for XLS-R
    _backbone(tmp_path / "checkpoint-1500")  # rolling, not a milestone
    found = find_milestone_checkpoints(str(tmp_path), [9000, 4500, 13500])
    assert [p.name for p in found] == ["checkpoint-4500", "checkpoint-9000"]


def test_a_milestone_without_weights_is_skipped(tmp_path, capsys):
    (tmp_path / "checkpoint-4500").mkdir()
    (tmp_path / "checkpoint-4500" / "config.json").write_text("{}")
    assert find_milestone_checkpoints(str(tmp_path), [4500]) == []
    assert "checkpoint-4500" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config", "suffix_arg", "expected"),
    [
        ("params-xlsr.yaml", [], "checkpoint_selection-xlsr.json"),
        ("params-wav2vec2.yaml", [], "checkpoint_selection-wav2vec2.json"),
        ("params-wavlm.yaml", [], "checkpoint_selection-wavlm.json"),
        ("params.yaml", ["--report-suffix=-custom"], "checkpoint_selection-custom.json"),
    ],
)
def test_report_suffix_defaults_to_the_model_key(
    tmp_path, monkeypatch, config, suffix_arg, expected
):
    with open(config) as f:
        raw = yaml.safe_load(f)
    raw["pretrain"]["output_dir"] = str(tmp_path / "model")
    raw["checkpoint_selection"]["output_dir"] = str(tmp_path / "selected")
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    for step in raw["pretrain"]["milestone_checkpoints"]:
        _backbone(tmp_path / "model" / f"checkpoint-{step}")

    scores = iter([3.0, 1.0, 2.0])
    monkeypatch.setattr(select_checkpoint, "evaluate_checkpoint", lambda *a, **k: next(scores))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), *suffix_arg])
    select_checkpoint.main()

    report = json.loads((tmp_path / "reports" / expected).read_text())
    assert report["best_checkpoint"] == "checkpoint-9000"
    assert (tmp_path / "selected" / "config.json").exists()


# ── ranking on the dev score, never the test metric ──────────────────────────────────────


@pytest.fixture
def fake_slsb(monkeypatch):
    """Replace `slsb run` by a stub that writes the results file select_checkpoint reads.
    Usage: fake_slsb({"checkpoint-A": [entry, ...], ...}); entries carry metrics (test) and
    details (dev), as slsb 0.5 writes them."""
    import shutil
    import subprocess
    from types import SimpleNamespace

    written = []

    def install(entries_by_ckpt):
        def run(cmd, **_):
            out_dir = Path(cmd[cmd.index("--out") + 1])
            upstream = cmd[cmd.index("--upstream") + 1]
            out_dir.mkdir(parents=True, exist_ok=True)
            written.append(out_dir)
            name = Path(upstream).name
            results = {"results": entries_by_ckpt[name]}
            (out_dir / f"results_{upstream.replace('/', '__')}.json").write_text(
                json.dumps(results)
            )
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(subprocess, "run", run)

    yield install
    for out_dir in written:
        shutil.rmtree(out_dir, ignore_errors=True)


def _entry(metrics, details, status="ok"):
    return {"status": status, "metrics": metrics, "details": details}


def _score(name, task="asr"):
    return select_checkpoint.evaluate_checkpoint(Path("/x") / name, task, 0)


def test_ranking_follows_the_dev_score_even_when_the_test_metric_disagrees(fake_slsb):
    fake_slsb(
        {
            # better dev CER, WORSE test WER
            "checkpoint-990001": [_entry({"wer": 0.50, "cer": 0.2}, {"dev_cer": 0.10})],
            # worse dev CER, BETTER test WER
            "checkpoint-990002": [_entry({"wer": 0.30, "cer": 0.1}, {"dev_cer": 0.20})],
        }
    )
    assert _score("checkpoint-990001") < _score("checkpoint-990002")  # dev wins


def test_each_proxy_task_uses_its_own_dev_score_and_direction(fake_slsb):
    fake_slsb(
        {
            "checkpoint-990011": [
                _entry({}, {"dev_accuracy": 0.9, "dev_eer": 0.05, "dev_der": 0.1})
            ],
            "checkpoint-990012": [
                _entry({}, {"dev_accuracy": 0.6, "dev_eer": 0.15, "dev_der": 0.3})
            ],
        }
    )
    good, bad = "checkpoint-990011", "checkpoint-990012"
    for task in ("sid", "asv", "sd"):  # sid: higher accuracy is better (negated), others lower
        assert _score(good, task) < _score(bad, task), task
    assert _score(good, "sid") == pytest.approx(-0.9)


def test_the_dev_score_is_averaged_over_languages_and_seeds(fake_slsb):
    fake_slsb(
        {
            "checkpoint-990021": [
                _entry({}, {"dev_cer": 0.10}),
                _entry({}, {"dev_cer": 0.30}),
                _entry({}, {"dev_cer": 0.99}, status="error"),  # failed runs are ignored
            ]
        }
    )
    assert _score("checkpoint-990021") == pytest.approx(0.20)


def test_a_result_without_the_dev_score_is_a_failed_eval_not_a_test_fallback(fake_slsb, capsys):
    fake_slsb({"checkpoint-990031": [_entry({"wer": 0.3, "cer": 0.1}, {})]})  # test metric only
    assert _score("checkpoint-990031") == float("inf")
    assert "no dev score (dev_cer)" in capsys.readouterr().out


def test_emotion_recognition_cannot_be_a_proxy_task(tmp_path, monkeypatch):
    import yaml

    with open("params.yaml") as f:
        raw = yaml.safe_load(f)
    raw["checkpoint_selection"] = {"proxy_task": "er", "output_dir": str(tmp_path / "sel")}
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg)])
    with pytest.raises(SystemExit, match="no dev score to rank on"):
        select_checkpoint.main()


def test_the_report_says_which_score_it_ranked_on(tmp_path, monkeypatch):
    import yaml

    with open("params-wavlm.yaml") as f:
        raw = yaml.safe_load(f)
    raw["pretrain"]["output_dir"] = str(tmp_path / "model")
    raw["checkpoint_selection"]["output_dir"] = str(tmp_path / "selected")
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    for step in raw["pretrain"]["milestone_checkpoints"]:
        _backbone(tmp_path / "model" / f"checkpoint-{step}")
    monkeypatch.setattr(select_checkpoint, "evaluate_checkpoint", lambda *a, **k: 1.0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg)])
    select_checkpoint.main()
    report = json.loads((tmp_path / "reports" / "checkpoint_selection-wavlm.json").read_text())
    assert report["ranked_on"] == "dev_cer"
