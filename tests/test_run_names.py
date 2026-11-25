"""A run is <model>-<experiment>; everything it writes carries that name, so two
experiments (or two models) never share a folder, report or benchmark directory."""

import argparse

import pytest

from pipeline.common.schema import Params, load_raw_params
from pipeline.eval.benchmark import bench_name, log_results_command, make_plan, slsb_run_command


def test_default_run_name_is_model_prefix_plus_experiment():
    assert Params.from_yaml("params-wavlm.yaml").run == "wavlm-large-si-ta-200h"
    assert Params.from_yaml("params-mhubert147.yaml").run == "mhubert147-si-ta-200h"


def test_experiment_flag_renames_every_output_path():
    p = Params.from_yaml("params-wavlm.yaml", "si-100h")
    assert p.run == "wavlm-large-si-100h"
    assert p.pretrain.output_dir == "models/wavlm-large-si-100h"
    raw = load_raw_params("params-wavlm.yaml", "si-100h")
    assert raw["checkpoint_selection"]["output_dir"] == "models/selected-wavlm-large-si-100h"
    assert "wavlm-large-si-100h" in raw["benchmark"]["upstreams"]
    assert "{run}" not in str(raw)


def test_two_experiments_and_two_models_share_no_output_path():
    a = Params.from_yaml("params-wavlm.yaml", "si-ta-200h")
    b = Params.from_yaml("params-wavlm.yaml", "si-100h")
    c = Params.from_yaml("params-hubert.yaml", "si-100h")
    assert len({a.pretrain.output_dir, b.pretrain.output_dir, c.pretrain.output_dir}) == 3


@pytest.mark.parametrize("bad", ["a/b", "x{run}"])
def test_experiment_must_be_a_plain_name(bad):
    with pytest.raises(ValueError, match="plain name"):
        load_raw_params("params-wavlm.yaml", bad)


def _args(**kw):
    base = {
        "frozen": False,
        "upstream": None,
        "name": None,
        "out": None,
        "checkpoint": None,
        "leaderboard_uri": None,
    }
    return argparse.Namespace(**{**base, **kw})


def test_model_name_follows_the_leaderboard_convention():
    cases = {
        ("params-wavlm.yaml", None): "wavlm-large-copt200h",
        ("params-xlsr.yaml", None): "xlsr300m-copt200h-norm",
        ("params-wavlm.yaml", "si-100h-p30"): "wavlm-large-copt100h-si-p30",
        ("params-xlsr.yaml", "ta-100h"): "xlsr300m-copt100h-ta",
        ("params-hubert.yaml", "cnn-unfreeze"): "hubert-large-copt200h-cnn-unfreeze",
        ("params-wav2vec2.yaml", None): "wav2vec2-large-lv60-copt200h",
        ("params-mhubert147.yaml", None): "mhubert147-copt200h",
    }
    for (config, experiment), expected in cases.items():
        assert load_raw_params(config, experiment)["model_name"] == expected
        assert Params.from_yaml(config, experiment).model_name == expected
    assert load_raw_params("params-wavlm.yaml", "si-100h-p30")["pretrain_hours"] == "100"


def test_bench_defaults_to_the_runs_selected_checkpoint_and_own_folder():
    plan = make_plan(load_raw_params("params-wavlm.yaml", "si-100h"), _args())
    assert plan.upstream == "models/selected-wavlm-large-si-100h"
    assert str(plan.out_dir) == "reports/bench/wavlm-large-si-100h"
    assert (plan.name, plan.kind, plan.family) == ("wavlm-large-copt100h-si", "adapted", "wavlm")
    assert plan.pretrain_hours == "100"


def test_bench_never_logs_live_and_logs_one_leaderboard_row():
    plan = make_plan(load_raw_params("params-wavlm.yaml"), _args(checkpoint="9000"))
    run = slsb_run_command(plan, ["slsb"])
    assert "--mlflow-uri" not in run  # no per-task/seed runs anywhere
    assert run[run.index("--tasks") + 1] == "asr,sid,er,sd,asv,ic"
    log = log_results_command(plan, "python", "log_results.py")
    assert log[2] == "reports/bench/wavlm-large-si-ta-200h"
    assert log[log.index("--name") + 1] == "wavlm-large-copt200h"
    assert log[log.index("--mlflow-uri") + 1].endswith("/SLSB-benchmark.mlflow")
    assert log[log.index("--checkpoint") + 1] == "9000"


def test_frozen_baseline_is_the_base_model_row():
    plan = make_plan(load_raw_params("params-wav2vec2.yaml"), _args(frozen=True))
    assert plan.upstream == "facebook/wav2vec2-large-lv60"
    assert (plan.name, plan.kind, plan.checkpoint, plan.pretrain_hours) == (
        "wav2vec2-large-lv60",
        "frozen",
        "",
        "",
    )


def test_an_adhoc_upstream_is_not_logged_without_a_name():
    raw = load_raw_params("params-wavlm.yaml")
    plan = make_plan(raw, _args(upstream="utter-project/mHuBERT-147"))
    assert plan.name is None and plan.kind == "frozen" and plan.family == "mhubert"


def test_bench_of_any_other_checkpoint_gets_its_own_folder():
    raw = load_raw_params("params-wavlm.yaml")
    plan = make_plan(raw, _args(upstream="models/wavlm-large-si-ta-200h/checkpoint-9000"))
    assert str(plan.out_dir) == "reports/bench/wavlm-large-si-ta-200h-checkpoint-9000"
    assert bench_name("microsoft/wavlm-large") == "microsoft-wavlm-large"


def test_a_configured_baseline_lands_where_report_reads_it():
    """report.py looks results up by the benchmark.upstreams KEY, so benchmarking a
    configured baseline by its path must use that key as the folder -- otherwise the
    baseline renders as "—" in the table even after it was run."""
    from pipeline.eval.benchmark import resolve_target

    raw = load_raw_params("params-wavlm.yaml")
    keys = raw["benchmark"]["upstreams"]
    for key, cfg in keys.items():
        if key == raw["run_name"]:
            continue  # the adapted run itself, covered above
        _, name = resolve_target(raw, cfg["path"])
        assert name == key, f"baseline {cfg['path']} -> {name}, report.py expects {key}"


def test_an_unconfigured_upstream_still_gets_a_derived_folder():
    from pipeline.eval.benchmark import resolve_target

    raw = load_raw_params("params-wavlm.yaml")
    _, name = resolve_target(raw, "models/wavlm-large-si-ta-200h/checkpoint-9000")
    assert name == "wavlm-large-si-ta-200h-checkpoint-9000"
    _, explicit = resolve_target(raw, "microsoft/wavlm-large", name="my-folder")
    assert explicit == "my-folder"  # --name / NAME= always wins
