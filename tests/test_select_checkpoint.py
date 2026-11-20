"""Checkpoint selection: which checkpoints are scored, who wins, where the result goes."""

import json
import sys
from pathlib import Path

import pytest
import torch

from pipeline import select_checkpoint as sc


@pytest.fixture(autouse=True)
def _hermetic_eval_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "EVAL_DIR", tmp_path / "eval")


def _checkpoint(root: Path, step: int, complete: bool = True) -> Path:
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    if complete:
        (d / "model.safetensors").write_bytes(b"x")
        torch.save({"optimizer": {}, "global_step": step}, d / "training_state.pt")
    return d


def test_only_complete_milestone_checkpoints_are_scored(tmp_path, capsys):
    _checkpoint(tmp_path, 3)
    _checkpoint(tmp_path, 6, complete=False)  # a crashed save
    _checkpoint(tmp_path, 4)  # complete, but not a milestone
    found = sc.find_milestone_checkpoints(str(tmp_path), [3, 6, 9])
    assert [p.name for p in found] == ["checkpoint-3"]
    assert "6" in capsys.readouterr().out  # the skipped milestones are named


def _fake_slsb(dev_by_checkpoint: dict[str, float], key: str = "dev_cer", test_by_checkpoint=None):
    """subprocess.run stand-in: writes the results JSON slsb would write, with a dev score in
    `details` and a (deliberately different) test metric in `metrics`."""

    def run(cmd, **_):
        upstream = cmd[cmd.index("--upstream") + 1]
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        name = Path(upstream).name
        test = (test_by_checkpoint or {}).get(name, 0.5)
        entry = {
            "status": "ok",
            "task": "asr",
            "metrics": {"wer": test, "cer": test},
            "details": {key: dev_by_checkpoint[name], "lr": 0.001},
        }
        (out / f"results_{upstream.replace('/', '__')}.json").write_text(
            json.dumps({"results": [entry, entry]})
        )
        return type("R", (), {"returncode": 0, "stderr": ""})()

    return run


@pytest.fixture(autouse=True)
def _slsb_is_found(monkeypatch):
    monkeypatch.setattr(sc, "slsb_command", lambda: ["slsb"])


def test_evaluate_checkpoint_returns_the_mean_dev_score_not_the_test_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sc.subprocess,
        "run",
        _fake_slsb({"checkpoint-3": 0.25}, test_by_checkpoint={"checkpoint-3": 0.99}),
    )
    ckpt = _checkpoint(tmp_path / "m", 3)
    assert sc.evaluate_checkpoint(ckpt, "asr", 0) == pytest.approx(0.25)


def test_higher_is_better_dev_scores_are_negated_so_that_lower_is_always_better(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sc.subprocess, "run", _fake_slsb({"checkpoint-3": 0.8}, "dev_accuracy"))
    ckpt = _checkpoint(tmp_path / "m", 3)
    assert sc.evaluate_checkpoint(ckpt, "sid", 0) == pytest.approx(-0.8)


def test_a_task_without_a_recorded_dev_score_cannot_be_the_proxy(tmp_path):
    ckpt = _checkpoint(tmp_path / "m", 3)
    with pytest.raises(SystemExit, match="no dev score"):
        sc.evaluate_checkpoint(ckpt, "er", 0)


def test_a_result_without_the_dev_score_is_infinity_not_a_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sc.subprocess, "run", _fake_slsb({"checkpoint-3": 0.3}, key="something_else")
    )
    assert sc.evaluate_checkpoint(_checkpoint(tmp_path / "m", 3), "asr", 0) == float("inf")


def test_a_failed_evaluation_scores_infinity_not_zero(tmp_path, monkeypatch):
    def failing(cmd, **_):
        return type("R", (), {"returncode": 1, "stderr": "boom"})()

    monkeypatch.setattr(sc.subprocess, "run", failing)
    ckpt = _checkpoint(tmp_path / "m", 3)
    assert sc.evaluate_checkpoint(ckpt, "asr", 0) == float("inf")


def _finished(reports_dir, step):
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "pretrain_metrics.json").write_text(
        json.dumps({"final_step": step, "final_loss": 1.0, "completed": True})
    )


def test_main_copies_the_best_checkpoint_and_writes_the_report_to_the_models_reports_dir(
    tmp_path, monkeypatch
):
    out, selected, reports = tmp_path / "out", tmp_path / "selected", tmp_path / "reports"
    for step in (3, 6):
        _checkpoint(out, step)
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"model: {{reports_dir: {reports}}}\n"
        f"pretrain: {{output_dir: {out}, milestone_checkpoints: [3, 6], max_updates: 6}}\n"
        f"checkpoint_selection: {{output_dir: {selected}}}\n"
    )
    monkeypatch.setattr(
        sc.subprocess, "run", _fake_slsb({"checkpoint-3": 0.9, "checkpoint-6": 0.4})
    )
    monkeypatch.chdir(tmp_path)
    _finished(reports, 6)
    repo = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(
        sys, "argv", ["x", "--config", str(repo / "params.yaml"), "--overlay", str(overlay)]
    )

    sc.main()

    assert (selected / "config.json").exists()
    report = json.loads((reports / "checkpoint_selection.json").read_text())
    assert report["best_checkpoint"] == "checkpoint-6"
    assert report["all_scores"] == {"checkpoint-3": 0.9, "checkpoint-6": 0.4}
    assert (report["selection_split"], report["selection_metric"]) == ("dev", "dev_cer")


def test_main_refuses_to_select_when_every_evaluation_failed(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _checkpoint(out, 3)
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"model: {{reports_dir: {tmp_path / 'r'}}}\n"
        f"pretrain: {{output_dir: {out}, milestone_checkpoints: [3], max_updates: 3}}\n"
        f"checkpoint_selection: {{output_dir: {tmp_path / 'sel'}}}\n"
    )
    monkeypatch.setattr(
        sc.subprocess, "run", lambda cmd, **_: type("R", (), {"returncode": 1, "stderr": ""})()
    )
    monkeypatch.chdir(tmp_path)
    _finished(tmp_path / "r", 3)
    repo = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(
        sys, "argv", ["x", "--config", str(repo / "params.yaml"), "--overlay", str(overlay)]
    )
    with pytest.raises(SystemExit, match="every proxy eval failed"):
        sc.main()
    assert not (tmp_path / "sel").exists()


def test_checkpoint_selection_waits_for_pretraining_to_finish(tmp_path):
    import json

    from pipeline.common.checkpoints import pretraining_problem

    assert "not found" in pretraining_problem(tmp_path, 13500)
    (tmp_path / "pretrain_metrics.json").write_text(
        json.dumps({"final_step": 9000, "final_loss": 1.0, "completed": False})
    )
    assert "stopped at step 9000 of 13500" in pretraining_problem(tmp_path, 13500)
    (tmp_path / "pretrain_metrics.json").write_text(
        json.dumps({"final_step": 13500, "final_loss": 1.0, "completed": True})
    )
    assert pretraining_problem(tmp_path, 13500) is None
