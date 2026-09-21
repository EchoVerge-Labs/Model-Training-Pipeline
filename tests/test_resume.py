"""Checkpoint resume, auto-resume, and the collapse detector's wiring into train().

Everything here uses a tiny random wav2vec2 on CPU, synthetic batches injected in
place of the shard loader, and pytest's tmp_path for every output -- never the
repo's real reports/ or models/ directories (a real training run may be writing
to them).
"""

from pathlib import Path

import pytest
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining

from pipeline.callbacks import CodebookCollapseDetector
from pipeline.schema import MonitoringConfig, Params
from pipeline.train import (
    CURVES_HEADER,
    TRAINING_STATE_FILE,
    checkpoint_step,
    find_resume,
    list_checkpoints,
    load_training_state,
    prepare_curves,
    prune_checkpoints,
    save_checkpoint,
    train,
)
from tests.test_smoke_train import _tiny_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    """Any stray relative path (reports/, models/, params.yaml) lands in tmp_path, never
    in the repo -- a real training run may be reading and writing those."""
    monkeypatch.chdir(tmp_path)


def _feature_extractor() -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )


def _make_base(tmp_path: Path) -> Path:
    """A tiny random 'pretrained' model directory to train from."""
    base = tmp_path / "base"
    _tiny_model().save_pretrained(base)
    _feature_extractor().save_pretrained(base)
    return base


def _params(tmp_path: Path, base: Path, monitoring: dict | None = None, **pretrain) -> Params:
    params = Params.from_yaml(str(REPO_ROOT / "params.yaml"))
    settings = {
        "base_model": str(base),
        "precision": "fp32",
        "freeze_feature_encoder": True,
        "max_updates": 3,
        "warmup_updates": 1,
        "target_batch_seconds": 2.0,  # with per_device 2.0 -> 1 micro-batch per update
        "per_device_max_seconds": 2.0,
        "num_workers": 0,
        "save_every_updates": 3,
        "eval_every_updates": 1,
        "keep_last_n_checkpoints": 5,
        "milestone_checkpoints": [],
        "output_dir": str(tmp_path / "out"),
    }
    settings.update(pretrain)
    # a floor this low never trips on the tiny model's random perplexity
    monitor = {"perplexity_floor": 0.001, "consecutive_alerts": 5, **(monitoring or {})}
    return params.model_copy(
        update={
            "pretrain": params.pretrain.model_copy(update=settings),
            "monitoring": MonitoringConfig(**monitor),
        }
    )


def _batches(n: int = 4) -> list[dict]:
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "input_values": torch.randn(2, 16000, generator=generator),
            "attention_mask": torch.ones(2, 16000, dtype=torch.long),
        }
        for _ in range(n)
    ]


def _run(params: Params, tmp_path: Path, **kwargs) -> int:
    kwargs.setdefault("use_mlflow", False)
    return train(
        params,
        dataloader=_batches(),
        reports_dir=tmp_path / "reports",
        device="cpu",
        **kwargs,
    )


def _curve_steps(tmp_path: Path) -> list[int]:
    lines = (tmp_path / "reports" / "pretrain_curves.csv").read_text().splitlines()
    assert lines[0] == CURVES_HEADER
    return [int(line.split(",")[0]) for line in lines[1:]]


def _write_checkpoint(out: Path, step: int):
    """A real (tiny) complete checkpoint, written by the production save path."""
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return save_checkpoint(model, _feature_extractor(), optimizer, step, 0.9, out)


# ── the scenario asked for: save at step 3, resume, 2 more steps -> step 5 ──


def test_resume_from_step_3_runs_two_more_updates_and_ends_at_step_5(tmp_path, capsys):
    base = _make_base(tmp_path)
    out = tmp_path / "out"

    assert _run(_params(tmp_path, base, max_updates=3), tmp_path) == 3
    assert (out / "checkpoint-3" / TRAINING_STATE_FILE).exists()

    # Same command, longer schedule, no flags: picks up checkpoint-3 by itself.
    assert _run(_params(tmp_path, base, max_updates=5), tmp_path) == 5

    printed = capsys.readouterr().out
    assert "Resuming automatically from" in printed and "checkpoint-3 at step 3" in printed
    # Run 2 continued (4, 5) instead of starting over (which would re-log 1, 2, ...).
    assert _curve_steps(tmp_path) == [1, 2, 3, 4, 5]
    assert '"final_step": 5' in (tmp_path / "reports" / "pretrain_metrics.json").read_text()


def test_explicit_resume_from_a_checkpoint_directory(tmp_path):
    base = _make_base(tmp_path)
    _run(_params(tmp_path, base, max_updates=3), tmp_path)

    # A different output dir, so nothing could be auto-discovered: only the flag can work.
    params = _params(tmp_path, base, max_updates=5, output_dir=str(tmp_path / "elsewhere"))
    assert _run(params, tmp_path, resume_from=tmp_path / "out" / "checkpoint-3") == 5
    assert _curve_steps(tmp_path) == [1, 2, 3, 4, 5]


def test_explicit_resume_from_something_that_is_not_a_checkpoint_is_an_error(tmp_path):
    base = _make_base(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a complete checkpoint"):
        _run(_params(tmp_path, base), tmp_path, resume_from=tmp_path / "nope")


def test_no_resume_starts_from_the_base_model_even_with_checkpoints_present(tmp_path, capsys):
    base = _make_base(tmp_path)
    _run(_params(tmp_path, base, max_updates=3), tmp_path)
    capsys.readouterr()

    assert _run(_params(tmp_path, base, max_updates=2), tmp_path, auto_resume=False) == 2
    assert "Starting from the base model" in capsys.readouterr().out
    assert _curve_steps(tmp_path) == [1, 2]  # a fresh curve, not appended to the old one


def test_relaunching_a_finished_run_changes_nothing(tmp_path, capsys):
    base = _make_base(tmp_path)
    params = _params(tmp_path, base, max_updates=3)
    _run(params, tmp_path)
    reports = tmp_path / "reports"
    before = {p.name: p.read_bytes() for p in reports.iterdir()}
    capsys.readouterr()

    assert _run(params, tmp_path) == 3
    assert "nothing to train" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in reports.iterdir()} == before


# ── what gets restored ──


def test_model_and_optimizer_state_round_trip_through_a_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model()
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for _ in range(3):  # three real optimizer steps so Adam has moments and a step count
        for p in params:
            p.grad = torch.randn_like(p)
        optimizer.step()
    ckpt = save_checkpoint(model, _feature_extractor(), optimizer, 3, 0.9, tmp_path)

    restored = Wav2Vec2ForPreTraining.from_pretrained(ckpt)
    optimizer2 = torch.optim.AdamW(list(restored.parameters()), lr=1e-3)
    state = load_training_state(ckpt, CPU)
    optimizer2.load_state_dict(state["optimizer"])

    assert state["global_step"] == 3 and state["gumbel_temperature"] == pytest.approx(0.9)
    restored_weights = restored.state_dict()
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, restored_weights[name]), name
    saved, loaded = optimizer.state_dict()["state"], optimizer2.state_dict()["state"]
    assert set(saved) == set(loaded)
    for i in saved:
        assert loaded[i]["step"] == 3
        assert torch.equal(saved[i]["exp_avg"], loaded[i]["exp_avg"])
        assert torch.equal(saved[i]["exp_avg_sq"], loaded[i]["exp_avg_sq"])


# ── choosing what to resume from ──


def test_auto_resume_picks_the_newest_complete_checkpoint_and_ignores_debris(tmp_path):
    out = tmp_path / "out"
    _write_checkpoint(out, 3)
    _write_checkpoint(out, 6)
    incomplete = _write_checkpoint(out, 9)
    (incomplete / TRAINING_STATE_FILE).unlink()  # a save that died before its last file
    (out / ".tmp-checkpoint-12").mkdir()  # a save that never finished renaming
    (out / "checkpoint-final").mkdir()  # not a numbered checkpoint

    assert [checkpoint_step(p) for p in list_checkpoints(out)] == [6, 3]
    path, state = find_resume(None, out, CPU)
    assert path.name == "checkpoint-6" and state["global_step"] == 6


def test_auto_resume_falls_back_when_the_newest_checkpoint_is_unreadable(tmp_path, capsys):
    out = tmp_path / "out"
    _write_checkpoint(out, 3)
    truncated = _write_checkpoint(out, 6)
    (truncated / TRAINING_STATE_FILE).write_bytes(b"cut off mid-write")

    path, state = find_resume(None, out, CPU)
    assert path.name == "checkpoint-3" and state["global_step"] == 3
    assert "skipping unreadable checkpoint" in capsys.readouterr().out


def test_no_checkpoints_means_a_fresh_start(tmp_path):
    assert find_resume(None, tmp_path / "empty", CPU) is None
    _write_checkpoint(tmp_path / "out", 3)
    assert find_resume(None, tmp_path / "out", CPU, auto_resume=False) is None


def test_a_crash_while_saving_leaves_no_checkpoint_that_looks_valid(tmp_path, monkeypatch):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def die(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", die)
    with pytest.raises(RuntimeError, match="disk full"):
        save_checkpoint(model, _feature_extractor(), optimizer, 3, 0.9, tmp_path)

    assert not (tmp_path / "checkpoint-3").exists()
    assert list_checkpoints(tmp_path) == []


def test_a_stale_partial_save_is_cleaned_up_at_startup(tmp_path):
    base = _make_base(tmp_path)
    stale = tmp_path / "out" / ".tmp-checkpoint-1500"
    stale.mkdir(parents=True)
    (stale / "half_written.bin").write_bytes(b"x")
    _run(_params(tmp_path, base, max_updates=1), tmp_path)
    assert not stale.exists()


# ── bookkeeping around a resume ──


def test_pruning_keeps_milestones_plus_the_newest_few_others(tmp_path):
    for step in (1, 2, 3, 4, 5, 6):
        (tmp_path / f"checkpoint-{step}").mkdir()
    prune_checkpoints(tmp_path, milestones=[2, 5], keep_last_n=2)
    kept = sorted(checkpoint_step(p) for p in tmp_path.glob("checkpoint-*"))
    assert kept == [2, 4, 5, 6]  # milestones 2 and 5, plus the newest two others (4 and 6)


def test_resuming_trims_the_loss_curve_so_it_has_no_duplicates(tmp_path):
    curves = tmp_path / "pretrain_curves.csv"
    rows = [f"{s},1,1,1,1,1,1,1,1" for s in (500, 1000, 1500, 2000)]
    curves.write_text("\n".join([CURVES_HEADER, *rows]) + "\n")

    prepare_curves(curves, resume_step=1500)  # crashed after logging 2000, last checkpoint at 1500
    steps = [line.split(",")[0] for line in curves.read_text().splitlines()]
    assert steps == ["step", "500", "1000", "1500"]

    prepare_curves(curves, resume_step=0)  # a fresh run starts over
    assert curves.read_text() == CURVES_HEADER + "\n"
    prepare_curves(tmp_path / "missing.csv", resume_step=1500)  # resuming with no history file
    assert (tmp_path / "missing.csv").read_text() == CURVES_HEADER + "\n"


# ── the collapse detector ──


def test_detector_stops_after_the_configured_number_of_consecutive_low_readings():
    detector = CodebookCollapseDetector(alert_threshold=10.0, consecutive_alerts=5)
    with pytest.warns(UserWarning, match="low reading"):
        outcomes = [detector.check(3.0, step) for step in (500, 1000, 1500, 2000, 2500)]
    assert outcomes == [True, True, True, True, False]


def test_detector_defaults_to_five_readings_not_five_hundred():
    detector = CodebookCollapseDetector()
    with pytest.warns(UserWarning):
        assert [detector.check(0.5, s) for s in range(5)] == [True, True, True, True, False]


def test_a_healthy_reading_resets_the_count():
    detector = CodebookCollapseDetector(alert_threshold=10.0, consecutive_alerts=5)
    with pytest.warns(UserWarning):
        assert all(detector.check(3.0, s) for s in range(4))
    assert detector.check(50.0, 5)  # recovers
    with pytest.warns(UserWarning):
        assert all(detector.check(3.0, s) for s in range(6, 10))  # 4 more lows: still fine


def test_monitoring_defaults_and_validation():
    assert MonitoringConfig().perplexity_floor == 10.0
    assert MonitoringConfig().consecutive_alerts == 5
    real = Params.from_yaml(str(REPO_ROOT / "params.yaml")).monitoring
    assert (real.perplexity_floor, real.consecutive_alerts) == (10.0, 5)
    with pytest.raises(ValueError):
        MonitoringConfig(consecutive_alerts=0)


def test_train_uses_the_monitoring_settings_to_stop_on_collapse(tmp_path, capsys):
    base = _make_base(tmp_path)
    # A floor far above any perplexity makes every reading "low"; two in a row must stop the run.
    params = _params(
        tmp_path,
        base,
        max_updates=10,
        save_every_updates=100,
        monitoring={"perplexity_floor": 1e9, "consecutive_alerts": 2},
    )
    with pytest.warns(UserWarning):
        assert _run(params, tmp_path) == 2
    assert "CODEBOOK COLLAPSE DETECTED" in capsys.readouterr().out
    assert '"completed": false' in (tmp_path / "reports" / "pretrain_metrics.json").read_text()
