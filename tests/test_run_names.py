"""One name per run, derived from its config (pipeline.common.naming): every folder, report,
log, DagsHub training run and SLSB leaderboard row of the run carries it."""

import argparse
import json

import pytest
import yaml

from pipeline.common.naming import passes_token
from pipeline.common.schema import DataParams, Params, load_params, load_raw_params
from pipeline.eval.benchmark import (
    frozen_name,
    log_results_command,
    make_plan,
    slsb_run_command,
)

V = "configs/variants/"


@pytest.mark.parametrize(
    ("config", "variants", "tag", "name"),
    [
        ("params-wavlm.yaml", [], None, "wavlm-large-copt-si-ta-200h"),
        ("params-hubert.yaml", [], None, "hubert-large-copt-si-ta-200h"),
        ("params-mhubert147.yaml", [], None, "mhubert147-copt-si-ta-200h"),
        ("params-xlsr.yaml", [], None, "xlsr300m-copt-si-ta-200h"),
        ("params-wav2vec2.yaml", [], None, "wav2vec2-large-lv60-copt-si-ta-200h"),
        ("params-hubert.yaml", ["cnn.yaml"], None, "hubert-large-copt-si-ta-200h-cnn"),
        ("params-xlsr.yaml", ["homog.yaml"], None, "xlsr300m-copt-si-ta-200h-homog"),
        (
            "params-wavlm.yaml",
            ["homog.yaml", "cnn.yaml"],
            None,
            "wavlm-large-copt-si-ta-200h-cnn-homog",
        ),
        ("params-hubert.yaml", ["lora.yaml"], None, "hubert-large-lora-si-ta-200h"),
        ("params-mhubert147.yaml", ["adapter.yaml"], None, "mhubert147-adapter-si-ta-200h"),
        ("params-xlsr.yaml", ["gclora.yaml"], "r16", "xlsr300m-gclora-si-ta-200h-r16"),
        ("params-wavlm.yaml", ["smoke-1h.yaml"], None, "wavlm-large-copt-si-ta-1h-smoke"),
        ("params-xlsr.yaml", ["smoke-1h.yaml"], None, "xlsr300m-copt-si-ta-1h-smoke"),
    ],
)
def test_the_name_is_derived_from_the_config(config, variants, tag, name):
    params = Params.from_yaml(config, [V + v for v in variants], tag)
    assert params.run == name
    assert params.pretrain.output_dir == f"models/{name}"
    assert str(params.reports_dir) == f"reports/{name}"
    assert str(params.selected_dir) == f"models/{name}/selected"
    assert str(params.logs_dir) == f"logs/{name}"


def test_passes_show_only_when_not_the_standard_30():
    assert passes_token(30.0) is None and passes_token(30.22) is None  # 68 updates over 1 h
    assert (
        passes_token(60.0) == "p60" and passes_token(10.4) == "p10" and passes_token(0.5) == "p0.5"
    )
    p = Params.from_yaml("params-wavlm.yaml")
    assert round(p.passes) == 30
    longer = p.model_copy(update={"pretrain": p.pretrain.model_copy(update={"max_updates": 27000})})
    assert longer.run == "wavlm-large-copt-si-ta-200h-p60"


def test_a_change_the_name_does_not_show_needs_a_tag(tmp_path):
    overlay = tmp_path / "lr.yaml"
    overlay.write_text(yaml.safe_dump({"pretrain": {"peak_lr": 1e-4}}))
    with pytest.raises(ValueError, match="does not show"):
        Params.from_yaml("params-wavlm.yaml", [str(overlay)])
    assert Params.from_yaml("params-wavlm.yaml", [str(overlay)], "lr1e-4").run.endswith("-lr1e-4")


def test_logging_cadence_and_scaled_warmup_need_no_tag(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {"pretrain": {"eval_every_updates": 100, "max_updates": 27000, "warmup_updates": 2000}}
        )
    )
    assert Params.from_yaml("params-wavlm.yaml", [str(overlay)]).run.endswith("-p60")


def test_a_tag_must_be_a_plain_lowercase_token():
    with pytest.raises(ValueError, match="tag"):
        Params.from_yaml("params-wavlm.yaml", tag="Bad Tag")


def test_data_is_named_by_its_data_id_and_a_changed_selection_gets_its_own():
    p = Params.from_yaml("params-wavlm.yaml")
    assert (p.data_id, p.shard.output_dir, p.select.manifest_dir) == (
        "si-ta-200h",
        "data/shars/si-ta-200h",
        "data/manifests/si-ta-200h",
    )
    smoke = DataParams.from_yaml("params-data.yaml", [V + "smoke-1h.yaml"])
    assert smoke.data_id == "si-ta-1h-smoke"
    assert str(smoke.reports_dir) == "reports/data/si-ta-1h-smoke"
    # a run that only changes training settings keeps the shared data
    assert Params.from_yaml("params-wavlm.yaml", [V + "cnn.yaml"]).data_id == "si-ta-200h"


def test_kmeans_and_labels_are_named_by_model_and_data():
    p = Params.from_yaml("params-wavlm.yaml")
    assert p.cluster.kmeans_output == "models/kmeans/wavlm-large-L18-k500-si-ta-200h.joblib"
    assert p.pretrain.labels_path == "data/labels/wavlm-large-L18-k500-si-ta-200h.jsonl.gz"
    smoke = Params.from_yaml("params-wavlm.yaml", [V + "smoke-1h.yaml"])
    assert smoke.cluster.kmeans_output == p.cluster.kmeans_output  # reuses the 200 h k-means
    assert smoke.pretrain.labels_path.endswith("-si-ta-200h-on-si-ta-1h-smoke.jsonl.gz")


def test_homogeneous_batching_needs_both_languages():
    p = Params.from_yaml("params-wavlm.yaml", [V + "homog.yaml"])
    lb = p.pretrain.language_batching
    assert lb.languages == ["sinhala", "tamil"] and lb.probabilities == {
        "sinhala": 0.5,
        "tamil": 0.5,
    }


def test_the_name_slots_are_the_dagshub_tags():
    parts = Params.from_yaml("params-xlsr.yaml", [V + "homog.yaml", V + "cnn.yaml"]).name_parts
    assert parts == {
        "model": "xlsr300m",
        "method": "copt",
        "languages": "si-ta",
        "hours": "200",
        "cnn": "trained",
        "passes": "30",
        "batching": "homogeneous",
        "data": "si-ta-200h",
    }


def test_frozen_baselines_are_named_model_frozen():
    assert frozen_name("wavlm") == "wavlm-large-frozen"
    assert frozen_name("wav2vec2") == "wav2vec2-large-lv60-frozen"


# ── run folder and data guards ──


def test_a_run_folder_belongs_to_one_config(tmp_path, monkeypatch):
    from pipeline.common.tracking import RUN_FILE, claim_run_folder

    monkeypatch.chdir(tmp_path)
    p = Params.from_yaml(str(_repo("params-xlsr.yaml")))
    claim_run_folder(p)
    record = json.loads((p.model_dir / RUN_FILE).read_text())
    assert record["name"] == p.run and record["parts"]["model"] == "xlsr300m"
    claim_run_folder(p)  # the same config: fine
    other = p.model_copy(deep=True)
    other.pretrain.peak_lr = 1e-3
    with pytest.raises(SystemExit, match="different config"):
        claim_run_folder(other)


def test_checkpoints_without_run_json_are_not_adopted(tmp_path, monkeypatch):
    from pipeline.common.tracking import claim_run_folder

    monkeypatch.chdir(tmp_path)
    p = Params.from_yaml(str(_repo("params-xlsr.yaml")))
    (p.model_dir / "checkpoint-1500.tmp").mkdir(parents=True)  # a half save: ignored
    claim_run_folder(p, write=False)
    (p.model_dir / "checkpoint-1500").mkdir()
    with pytest.raises(SystemExit, match="no run.json"):
        claim_run_folder(p)


def test_select_refuses_to_overwrite_a_different_selection(tmp_path, monkeypatch):
    from pipeline.data.select import claim_manifest_dir

    monkeypatch.chdir(tmp_path)
    d = DataParams.from_yaml(str(_repo("params-data.yaml")))
    claim_manifest_dir(d)
    claim_manifest_dir(d)  # the same settings: fine
    changed = d.model_copy(deep=True)
    changed.select.max_hours_per_channel = 3.0
    with pytest.raises(SystemExit, match="other settings"):
        claim_manifest_dir(changed)


def test_fit_kmeans_never_overwrites_an_existing_model(tmp_path, monkeypatch):
    import sys

    pytest.importorskip("lhotse")
    from pipeline.labels import fit_kmeans

    monkeypatch.chdir(tmp_path)
    p = Params.from_yaml(str(_repo("params-wavlm.yaml")))
    km = tmp_path / p.cluster.kmeans_output
    km.parent.mkdir(parents=True)
    km.write_bytes(b"existing")
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(_repo("params-wavlm.yaml"))])
    with pytest.raises(SystemExit, match="already exists"):
        fit_kmeans.main()
    assert km.read_bytes() == b"existing"


# ── the benchmark: one leaderboard row per run ──


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


def test_bench_benchmarks_the_runs_selected_checkpoint_into_its_own_folder():
    raw, params = load_params("params-wavlm.yaml", [V + "cnn.yaml"])
    plan = make_plan(raw, params, _args())
    name = "wavlm-large-copt-si-ta-200h-cnn"
    assert plan.upstream == f"models/{name}/selected"
    assert str(plan.out_dir) == f"reports/{name}/bench"
    assert (plan.name, plan.kind, plan.family, plan.pretrain_hours) == (
        name,
        "adapted",
        "wavlm",
        "200",
    )
    assert plan.tags["slsb.model"] == name and plan.tags["cnn"] == "trained"


def test_bench_never_logs_live_and_logs_one_leaderboard_row():
    raw, params = load_params("params-wavlm.yaml")
    plan = make_plan(raw, params, _args(checkpoint="9000"))
    run = slsb_run_command(plan, ["slsb"])
    assert "--mlflow-uri" not in run  # no per-task/seed runs anywhere
    assert run[run.index("--tasks") + 1] == "asr,sid,er,sd,asv,ic"
    log = log_results_command(plan, "python", "log_results.py")
    assert log[2] == "reports/wavlm-large-copt-si-ta-200h/bench"
    assert log[log.index("--name") + 1] == "wavlm-large-copt-si-ta-200h"
    assert log[log.index("--mlflow-uri") + 1].endswith("/SLSB-benchmark.mlflow")
    assert log[log.index("--checkpoint") + 1] == "9000"
    assert plan.partial_note == ""


def test_a_partial_benchmark_says_so_in_its_note():
    raw, params = load_params("params-wavlm.yaml", [V + "smoke-1h.yaml"])
    plan = make_plan(raw, params, _args())
    assert plan.tasks == "er" and "Partial benchmark" in plan.partial_note
    assert "asr" in plan.partial_note  # names what was left out


def test_frozen_baseline_row_and_folder():
    raw, params = load_params("params-wav2vec2.yaml")
    plan = make_plan(raw, params, _args(frozen=True))
    assert plan.upstream == "facebook/wav2vec2-large-lv60"
    assert (plan.name, plan.kind, plan.checkpoint, plan.pretrain_hours) == (
        "wav2vec2-large-lv60-frozen",
        "frozen",
        "",
        "",
    )
    assert str(plan.out_dir) == "reports/wav2vec2-large-lv60-frozen/bench" and plan.tags == {}


def test_an_adhoc_upstream_is_not_logged_without_a_name():
    raw, params = load_params("params-wavlm.yaml")
    plan = make_plan(raw, params, _args(upstream="utter-project/mHuBERT-147"))
    assert plan.name is None and plan.kind == "frozen" and plan.family == "mhubert"


def test_report_compares_the_run_with_its_frozen_baselines(tmp_path, monkeypatch):
    from pipeline.eval.report import build_report

    monkeypatch.chdir(tmp_path)
    raw, params = load_params(str(_repo("params-wavlm.yaml")))
    md, _ = build_report(raw, params)
    text = md.read_text()
    assert (tmp_path / md).resolve() == (tmp_path / "reports" / params.run / "results.md")
    for row in (params.run, "wavlm-large-frozen", "xlsr300m-frozen", "mhubert147-frozen"):
        assert f"| {row} |" in text


def _repo(name):
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / name


def test_load_raw_params_keeps_the_override_separate_from_the_derived_name():
    raw = load_raw_params("params-wavlm.yaml")
    assert raw["run"] == "wavlm-large-copt-si-ta-200h" and raw.get("run_name") is None
