"""The per-model DVC stages name paths that must equal the ones in configs/models/<model>.yaml."""

from pathlib import Path

import yaml

from pipeline.common.config import load_raw

REPO_ROOT = Path(__file__).resolve().parent.parent
DVC = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"]
LABELLED = ("wavlm", "hubert", "mhubert")
ALL_MODELS = ("wav2vec2", *LABELLED)


def _cfg(model):
    return load_raw(str(REPO_ROOT / "params.yaml"), model)


def _items(stage):
    return DVC[stage]["foreach"]


def test_every_model_has_a_stage_of_each_kind():
    assert set(_items("fit_kmeans")) == set(LABELLED)
    assert set(_items("assign_cluster_labels")) == set(LABELLED)
    for stage in ("copt", "ckpt", "bench"):
        assert set(_items(stage)) == set(ALL_MODELS), stage


def test_label_stage_paths_match_the_model_config():
    for model in LABELLED:
        cfg = _cfg(model)
        assert _items("fit_kmeans")[model]["kmeans"] == cfg["cluster"]["kmeans_output"]
        assert _items("fit_kmeans")[model]["reports"] == cfg["model"]["reports_dir"]
        item = _items("assign_cluster_labels")[model]
        assert item["kmeans"] == cfg["cluster"]["kmeans_output"]
        assert item["labels"] == cfg["cluster"]["labels_output"] == cfg["pretrain"]["labels_path"]


def test_training_and_checkpoint_stage_paths_match_the_model_config():
    for model in ALL_MODELS:
        cfg = _cfg(model)
        copt, ckpt = (_items(s)[model] for s in ("copt", "ckpt"))
        assert copt["out"] == ckpt["out"] == cfg["pretrain"]["output_dir"]
        assert copt["reports"] == ckpt["reports"] == cfg["model"]["reports_dir"]
        assert ckpt["selected"] == cfg["checkpoint_selection"]["output_dir"]


def test_benchmark_stages_name_the_models_leaderboard_run_and_checkpoint_dir():
    for model in ALL_MODELS:
        cfg = _cfg(model)
        item = _items("bench")[model]
        assert item["selected"] == cfg["checkpoint_selection"]["output_dir"]
        # the metrics path is where pipeline.benchmark writes: reports/bench/<leaderboard name>
        assert item["name"] == cfg["model"]["run_name"]
    assert DVC["benchmark"]["cmd"].startswith("python -m pipeline.benchmark")


def test_labelled_models_depend_on_their_labels_and_their_own_trainer_code():
    for model in LABELLED:
        copt = _items("copt")[model]
        assert copt["also"] == _cfg(model)["pretrain"]["labels_path"]
        assert copt["code"].rstrip("/").endswith(_cfg(model)["model"]["trainer"])
    assert _items("copt")["wav2vec2"]["code"].rstrip("/").endswith("xlsr")


def test_the_xlsr_stages_keep_their_names_and_outputs():
    pretrain = DVC["pretrain"]
    assert pretrain["cmd"] == "bash scripts/launch_pretrain.sh"
    assert pretrain["outs"] == [{"models/xlsr300m-si-ta-200h-norm/": {"persist": True}}]
    assert {"select_checkpoint", "benchmark", "report"} <= set(DVC)


def test_the_xlsr_benchmark_stage_writes_where_make_bench_and_the_report_look():
    """No --out: pipeline.benchmark's default is reports/bench/<run_name>/, which is also
    what `make bench` writes and what the report's upstream key for XLS-R is."""
    run_name = load_raw(str(REPO_ROOT / "params.yaml"))["model"]["run_name"]
    bench = DVC["benchmark"]
    assert "--out" not in bench["cmd"]
    assert bench["metrics"] == [{f"reports/bench/{run_name}/metrics.json": {"cache": False}}]
    assert run_name in load_raw(str(REPO_ROOT / "params.yaml"))["benchmark"]["upstreams"]
