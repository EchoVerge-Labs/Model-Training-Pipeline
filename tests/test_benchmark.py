"""pipeline.benchmark drives the SLSB repository; these check what it asks SLSB to do."""

import json
from pathlib import Path

import pytest

from pipeline import benchmark as bm
from pipeline import slsb

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _fake_slsb(monkeypatch):
    monkeypatch.setattr(bm, "slsb_command", lambda: ["slsb"])
    monkeypatch.setattr(bm, "slsb_python", lambda: "/slsb/python")
    monkeypatch.setattr(bm, "log_results_script", lambda: Path("/slsb/scripts/log_results.py"))
    monkeypatch.setattr(bm, "require_version", lambda expected, cmd=None: expected)


def _finish_pretraining(tmp_path, reports="reports/wavlm", step=13500):
    (tmp_path / reports).mkdir(parents=True, exist_ok=True)
    (tmp_path / reports / "pretrain_metrics.json").write_text(
        json.dumps({"final_step": step, "final_loss": 1.0, "completed": step >= 13500})
    )


def _run(monkeypatch, tmp_path, *argv, finished=True):
    if finished:
        _finish_pretraining(tmp_path)
    monkeypatch.chdir(tmp_path)
    return bm.main(["--config", str(REPO / "params.yaml"), *argv])


def test_a_configured_model_is_benchmarked_on_its_selected_checkpoint_with_every_task_family(
    monkeypatch, tmp_path, capsys
):
    assert _run(monkeypatch, tmp_path, "--model", "wavlm", "--dry-run") == 0
    out = capsys.readouterr().out
    run, log = [line.strip() for line in out.splitlines() if line.startswith("  ")]
    assert "--upstream models/selected_wavlm" in run
    assert "--tasks asr,sid,er,sd,asv,ic" in run and "--seeds 0,1,2" in run
    assert "--mlflow-uri" not in run  # live per-task logging is not leaderboard data
    assert "--name wavlm-large-copt200h --kind adapted --family wavlm" in log
    assert "--base-model microsoft/wavlm-large" in log and "--pretrain-hours 200" in log
    assert "SLSB-benchmark.mlflow" in log  # the leaderboard, not the pipeline's own experiment


def test_the_frozen_baseline_row_uses_the_base_model_and_no_checkpoint(
    monkeypatch, tmp_path, capsys
):
    _run(monkeypatch, tmp_path, "--model", "hubert", "--frozen", "--dry-run")
    run, log = [
        line.strip() for line in capsys.readouterr().out.splitlines() if line.startswith("  ")
    ]
    assert "--upstream facebook/hubert-large-ll60k" in run
    assert "--name hubert-large --kind frozen" in log
    assert "--checkpoint" not in log and "--pretrain-hours" not in log


def test_the_selected_checkpoint_step_comes_from_the_selection_report(
    monkeypatch, tmp_path, capsys
):
    reports = tmp_path / "reports" / "mhubert"
    reports.mkdir(parents=True)
    (reports / "checkpoint_selection.json").write_text(
        json.dumps({"best_checkpoint": "checkpoint-9000"})
    )
    _run(monkeypatch, tmp_path, "--model", "mhubert", "--dry-run")
    assert "--checkpoint 9000" in capsys.readouterr().out


def test_an_experiment_run_needs_its_own_name_so_it_cannot_overwrite_another_models_row(
    monkeypatch, tmp_path
):
    with pytest.raises(SystemExit, match="--upstream needs --name"):
        _run(monkeypatch, tmp_path, "--upstream", "models/some/checkpoint-100", "--dry-run")


def test_an_experiment_checkpoint_can_be_benchmarked_with_explicit_names(
    monkeypatch, tmp_path, capsys
):
    _run(
        monkeypatch,
        tmp_path,
        "--upstream", "models/xlsr300m-si-100h/checkpoint-6500",
        "--name", "xlsr300m-copt-si-100h", "--family", "xlsr",
        "--base-model", "facebook/wav2vec2-xls-r-300m", "--checkpoint", "6500",
        "--dry-run",
    )  # fmt: skip
    run, log = [
        line.strip() for line in capsys.readouterr().out.splitlines() if line.startswith("  ")
    ]
    assert "--out reports/bench/xlsr300m-copt-si-100h" in run
    assert "--name xlsr300m-copt-si-100h" in log and "--checkpoint 6500" in log


def test_results_are_copied_to_the_fixed_metrics_path_and_logged_afterwards(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        if cmd[0] == "slsb":
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True)
            (
                out / f"results_{cmd[cmd.index('--upstream') + 1].replace('/', '__')}.json"
            ).write_text("{}")

    monkeypatch.setattr(bm.subprocess, "run", fake_run)
    monkeypatch.setenv("DAGSHUB_USER", "u")
    monkeypatch.setenv("DAGSHUB_TOKEN", "t")
    (tmp_path / "models" / "selected_wavlm").mkdir(parents=True)
    _run(monkeypatch, tmp_path, "--model", "wavlm")
    assert (tmp_path / "reports" / "bench" / "wavlm-large-copt200h" / "metrics.json").is_file()
    assert [c[0] for c in calls] == ["slsb", "/slsb/python"]  # slsb first, then log_results.py


def test_without_dagshub_credentials_the_results_stay_local_and_say_how_to_log_them(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(bm.subprocess, "run", lambda cmd, check: None)
    monkeypatch.delenv("DAGSHUB_TOKEN", raising=False)
    monkeypatch.setattr(bm, "copy_metrics", lambda plan: plan.out_dir / "metrics.json")
    (tmp_path / "models" / "selected_wavlm").mkdir(parents=True)
    _run(monkeypatch, tmp_path, "--model", "wavlm")
    assert "were NOT logged" in capsys.readouterr().out


def test_an_adapted_run_refuses_to_start_without_the_checkpoint(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="make ckpt"):
        _run(monkeypatch, tmp_path, "--model", "wavlm")


def test_an_adapted_run_waits_for_pretraining_to_finish(monkeypatch, tmp_path):
    (tmp_path / "models" / "selected_wavlm").mkdir(parents=True)
    with pytest.raises(SystemExit, match="pre-training has not finished"):
        _run(monkeypatch, tmp_path, "--model", "wavlm", finished=False)
    _finish_pretraining(tmp_path, step=9000)
    with pytest.raises(SystemExit, match="stopped at step 9000 of 13500"):
        _run(monkeypatch, tmp_path, "--model", "wavlm", finished=False)


def test_allow_incomplete_overrides_the_wait_and_the_frozen_baseline_never_needs_it(
    monkeypatch, tmp_path
):
    (tmp_path / "models" / "selected_wavlm").mkdir(parents=True)
    monkeypatch.setattr(bm.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(bm, "copy_metrics", lambda plan: plan.out_dir / "metrics.json")
    _run(
        monkeypatch, tmp_path, "--model", "wavlm", "--allow-incomplete", "--no-log", finished=False
    )
    _run(monkeypatch, tmp_path, "--model", "wavlm", "--frozen", "--no-log", finished=False)


# ── locating and version-checking slsb ──


def test_slsb_is_found_from_the_environment_and_a_missing_install_is_a_clear_error(
    monkeypatch, tmp_path
):
    fake = tmp_path / "slsb"
    fake.write_text("#!/bin/sh\necho 'slsb 0.5.0'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("SLSB_BIN", str(fake))
    assert slsb.slsb_command() == [str(fake)]
    assert slsb.slsb_version() == "0.5.0"
    assert slsb.require_version("0.5.0") == "0.5.0"
    with pytest.raises(slsb.SlsbNotFound, match="not comparable"):
        slsb.require_version("0.4.0")

    monkeypatch.delenv("SLSB_BIN")
    monkeypatch.setenv("SLSB_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(slsb.SlsbNotFound, match="slsb not found"):
        slsb.slsb_command()
