"""select_checkpoint: one selector for all five models."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from pipeline.common.schema import Params, load_raw_params
from pipeline.common.tracking import claim_run_folder
from pipeline.eval import select_checkpoint
from pipeline.eval.select_checkpoint import find_milestone_checkpoints


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
    ("config", "tag_arg", "run"),
    [
        ("params-xlsr.yaml", [], "xlsr300m-copt-si-ta-200h"),
        ("params-wav2vec2.yaml", [], "wav2vec2-large-lv60-copt-si-ta-200h"),
        ("params-wavlm.yaml", [], "wavlm-large-copt-si-ta-200h"),
        ("params-mhubert147.yaml", ["--tag=custom"], "mhubert147-copt-si-ta-200h-custom"),
    ],
)
def test_the_selection_lands_in_the_runs_own_folders(tmp_path, monkeypatch, config, tag_arg, run):
    """reports/<run>/checkpoint_selection.json and <run folder>/selected/, named by the run."""
    raw = load_raw_params(config)
    raw["pretrain"]["output_dir"] = str(tmp_path / "model")
    cfg = tmp_path / "params-mhubert147.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    params = Params.from_yaml(str(cfg), tag=(tag_arg[0].split("=")[1] if tag_arg else None))
    claim_run_folder(params)  # the folder is this run's (make train writes run.json first)
    for step in raw["pretrain"]["milestone_checkpoints"]:
        _backbone(tmp_path / "model" / f"checkpoint-{step}")

    scores = iter([3.0, 1.0, 2.0])
    monkeypatch.setattr(select_checkpoint, "evaluate_checkpoint", lambda *a, **k: next(scores))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--allow-incomplete", *tag_arg])
    select_checkpoint.main()

    report = json.loads((tmp_path / "reports" / run / "checkpoint_selection.json").read_text())
    assert report["best_checkpoint"] == "checkpoint-9000"
    assert (tmp_path / "model" / "selected" / "config.json").exists()


def test_selection_refuses_a_folder_made_for_another_config(tmp_path, monkeypatch):
    raw = load_raw_params("params-xlsr.yaml")
    raw["pretrain"]["output_dir"] = str(tmp_path / "model")
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    other = Params.from_yaml(str(cfg)).model_copy(deep=True)
    other.pretrain.peak_lr = 1e-3  # a different config claimed the folder first
    claim_run_folder(other)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--allow-incomplete"])
    with pytest.raises(SystemExit, match="different config"):
        select_checkpoint.main()


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

    install.written = written
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

    raw = load_raw_params("params-mhubert147.yaml")
    raw["checkpoint_selection"] = {"proxy_task": "er", "output_dir": str(tmp_path / "sel")}
    cfg = tmp_path / "params-mhubert147.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--allow-incomplete"])
    with pytest.raises(SystemExit, match="no dev score to rank on"):
        select_checkpoint.main()


def test_the_report_says_which_score_it_ranked_on(tmp_path, monkeypatch):
    import yaml

    raw = load_raw_params("params-wavlm.yaml")
    raw["pretrain"]["output_dir"] = str(tmp_path / "model")
    cfg = tmp_path / "params-mhubert147.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    claim_run_folder(Params.from_yaml(str(cfg)))
    for step in raw["pretrain"]["milestone_checkpoints"]:
        _backbone(tmp_path / "model" / f"checkpoint-{step}")
    monkeypatch.setattr(select_checkpoint, "evaluate_checkpoint", lambda *a, **k: 1.0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--allow-incomplete"])
    select_checkpoint.main()
    report = json.loads(
        (
            tmp_path / "reports" / "wavlm-large-copt-si-ta-200h" / "checkpoint_selection.json"
        ).read_text()
    )
    assert report["ranked_on"] == "dev_cer"


def test_each_evaluation_uses_a_private_dir_that_is_removed_afterwards(fake_slsb, tmp_path):
    fake_slsb({"checkpoint-A": [_entry({}, {"dev_cer": 0.1})]})
    assert _score("checkpoint-A") == 0.1
    assert _score("checkpoint-A") == 0.1
    # the stub records every --out it was given (see fake_slsb)
    first, second = fake_slsb.written
    assert first != second
    assert not first.exists() and not second.exists()
