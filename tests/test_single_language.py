"""Single-language ablation runs: language slicing, isolation from the joint run, resume
safety, and the comparison report. No GPU: training uses the tiny CPU model from the
resume tests and injected batches."""

import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from pipeline import experiment
from pipeline.report import ablation_table, load_upstream_metrics
from pipeline.schema import ExperimentsConfig, Params
from pipeline.select import Segment, keep_language, select, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def params() -> Params:
    return Params.from_yaml(str(REPO_ROOT / "params.yaml"))


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Relative paths (data/, models/, reports/) land in tmp_path, never in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _with_runs(params: Params, runs: dict, **top) -> dict:
    return {
        **params.model_dump(),
        "experiments": {"backbone_tag": "wavlm-large", "runs": runs},
        **top,
    }


def _segments() -> list[Segment]:
    segs = []
    for lang, genre in (("sinhala", "drama"), ("tamil", "news")):
        for ch in range(6):
            for i in range(5):
                segs.append(
                    Segment(
                        filename=f"{lang}{ch}_{i}.wav",
                        language=lang,
                        duration_seconds=60.0,
                        genre=genre,
                        speaking_style="spontaneous",
                        language_formality="spoken",
                        code_switching="no",
                        channel_id=f"{lang}_ch{ch}",
                        title="",
                        source_url="",
                        size_bytes=1000 + i,
                        row_index=0,
                        relpath=f"{lang.capitalize()}/{genre}/ch{ch}/{lang}{ch}_{i}.wav",
                    )
                )
    return segs


def _select_cfg():
    cfg = MagicMock()
    cfg.seed = 7
    cfg.target_hours = 0.2  # 0.1h per language = 6 min = 2 channels of 5 min
    cfg.language_mix = {"sinhala": 0.5, "tamil": 0.5}
    cfg.min_segment_seconds = 1.0
    cfg.max_segment_seconds = 9999.0
    cfg.max_hours_per_channel = 1.0
    cfg.holdout_hours = 0.05
    return cfg


# ── selection: one language, nothing else ───────────────────────────────


def test_sinhala_only_selection_contains_no_tamil():
    train, holdout = select(_segments(), _select_cfg())
    si_train, si_dev = keep_language(train, "sinhala"), keep_language(holdout, "sinhala")
    assert si_train and {s.language for s in si_train + si_dev} == {"sinhala"}


def test_tamil_only_selection_contains_no_sinhala():
    train, holdout = select(_segments(), _select_cfg())
    ta_train, ta_dev = keep_language(train, "tamil"), keep_language(holdout, "tamil")
    assert ta_train and {s.language for s in ta_train + ta_dev} == {"tamil"}


def test_the_two_slices_partition_the_joint_selection():
    train, holdout = select(_segments(), _select_cfg())
    si, ta = keep_language(train, "sinhala"), keep_language(train, "tamil")
    assert {s.relpath for s in si}.isdisjoint(s.relpath for s in ta)
    assert {s.relpath for s in si + ta} == {s.relpath for s in train}
    assert {s.channel_id for s in si}.isdisjoint(s.channel_id for s in holdout)


def test_selection_is_reproducible_so_each_run_gets_the_same_slice():
    first = keep_language(select(_segments(), _select_cfg())[0], "tamil")
    again = keep_language(select(_segments(), _select_cfg())[0], "tamil")
    assert [s.relpath for s in first] == [s.relpath for s in again]


def test_manifest_guard_rejects_the_other_language(tmp_path):
    train, _ = select(_segments(), _select_cfg())
    manifest = tmp_path / "train.jsonl"
    write_manifest(keep_language(train, "sinhala"), manifest)
    assert experiment.assert_manifest_language(manifest, "sinhala") > 0
    with pytest.raises(ValueError, match="tamil"):
        experiment.assert_manifest_language(manifest, "tamil")
    write_manifest(train, manifest)  # joint manifest, both languages
    with pytest.raises(ValueError, match="expected only 'sinhala'"):
        experiment.assert_manifest_language(manifest, "sinhala")


def _write_shar(shar_dir: Path, languages: list[str]):
    shar_dir.mkdir(parents=True)
    with gzip.open(shar_dir / "cuts.000000.jsonl.gz", "wt") as f:
        for i, lang in enumerate(languages):
            f.write(json.dumps({"id": str(i), "custom": {"language": lang}}) + "\n")


def test_shar_guard(tmp_path):
    _write_shar(tmp_path / "ok", ["sinhala"] * 3)
    assert experiment.assert_shar_language(tmp_path / "ok", "sinhala") == 3
    _write_shar(tmp_path / "mixed", ["sinhala", "tamil"])
    with pytest.raises(ValueError, match="tamil"):
        experiment.assert_shar_language(tmp_path / "mixed", "sinhala")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no cuts"):
        experiment.assert_shar_language(tmp_path / "empty", "sinhala")


def test_summary_reports_hours_clips_and_zero_foreign_clips(params):
    _, exp = experiment.resolve(params, "si-100h")
    train = keep_language(select(_segments(), _select_cfg())[0], "sinhala")
    text = experiment.summary_text(train, exp, params)
    assert "Language: Sinhala" in text and "Target hours: 100" in text
    assert "Clips of any other language: 0" in text and "drama:" in text
    assert f"Number of clips: {len(train)}" in text


def test_slicing_the_joint_manifests_gives_each_language_exactly_its_half(params, in_tmp):
    from pipeline.select import slice_joint_manifests

    rows = [
        {
            "id": f"{lg}{i}",
            "filename": f"{lg}/{i}.wav",
            "language": lg,
            "duration": 60.0,
            "genre": "drama",
            "channel_id": f"{lg}_c{i % 3 if i < 4 else 9}",
        }
        for lg in ("sinhala", "tamil")
        for i in range(6)
    ]
    Path("data/manifests").mkdir(parents=True)
    for split, subset in (("train", rows[:4] + rows[6:10]), ("dev", rows[4:6] + rows[10:])):
        Path(f"data/manifests/{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in subset)
        )
    ids = {}
    for name in ("si-100h", "ta-100h"):
        _, exp = experiment.resolve(params, name)
        slice_joint_manifests(exp, params)
        ids[name] = [json.loads(line)["id"] for line in exp.train_manifest.read_text().splitlines()]
        assert {i[:2] for i in ids[name]} == {name[:2]}
        assert exp.selection_stats.exists()
    assert sorted(ids["si-100h"] + ids["ta-100h"]) == sorted(
        json.loads(line)["id"]
        for line in Path("data/manifests/train.jsonl").read_text().splitlines()
    )


# ── configuration ───────────────────────────────────────────────────────


def test_repo_params_define_both_runs_and_still_validate_as_joint(params):
    assert set(params.experiments.runs) == {"si-100h", "ta-100h"}
    assert params.select.language_mix == {"sinhala": 0.5, "tamil": 0.5}
    assert params.pretrain.output_dir == "models/wavlm-large-si-ta-200h"  # joint untouched


def test_joint_config_without_experiments_still_validates(params):
    raw = params.model_dump(exclude={"experiments"})
    assert Params(**raw).experiments is None
    with pytest.raises(ValueError, match="unknown experiment"):
        experiment.resolve(Params(**raw), "si-100h")


def test_target_hours_must_match_the_language_share_of_the_mix(params):
    ok = {"si": {"language": "sinhala", "target_hours": 100.0}}
    assert Params(**_with_runs(params, ok)).experiments
    for bad in (50.0, 200.0):
        with pytest.raises(ValidationError, match="target_hours"):
            Params(**_with_runs(params, {"si": {"language": "sinhala", "target_hours": bad}}))


def test_language_must_be_in_the_mix(params):
    with pytest.raises(ValidationError, match="not in select.language_mix"):
        Params(**_with_runs(params, {"en": {"language": "english", "target_hours": 100.0}}))


def test_run_names_must_be_path_safe():
    with pytest.raises(ValidationError, match="must match"):
        ExperimentsConfig(
            backbone_tag="x", runs={"../si": {"language": "sinhala", "target_hours": 1}}
        )


# ── isolation from the joint run and from each other ────────────────────


def test_each_run_gets_its_own_directories(params):
    (_, si), (_, ta) = (experiment.resolve(params, n) for n in ("si-100h", "ta-100h"))
    for field in ("manifest_dir", "shar_dir", "model_dir", "selected_dir", "reports_dir"):
        assert getattr(si, field) != getattr(ta, field), field
    assert si.model_dir == Path("models/wavlm-large-si-100h")
    assert ta.selected_dir == Path("models/selected_single/wavlm-large-ta-100h")
    assert si.bench_dir == Path("reports/bench/wavlm-large-si-100h")
    assert si.mlflow_run_name == "wavlm-large-copt-si-100h"


def test_no_run_path_is_inside_a_joint_path(params):
    joint = [
        "data/shars",
        "models/selected",
        params.pretrain.output_dir,
        "reports/bench/wavlm_large_adapted",
    ]
    for name in params.experiments.runs:
        resolved, exp = experiment.resolve(params, name)
        mine = [exp.shar_dir, exp.model_dir, exp.selected_dir, exp.bench_dir]
        assert resolved.pretrain.output_dir == str(exp.model_dir)
        assert resolved.shard.output_dir == str(exp.shar_dir)
        for path in mine:
            assert not any(
                str(path) == j or str(path).startswith(j + "/") or j.startswith(str(path) + "/")
                for j in joint
            ), path


def test_resolving_changes_only_locations(params):
    resolved, _ = experiment.resolve(params, "ta-100h")
    locations = {"output_dir", "labels_path"}
    for section in ("pretrain", "shard"):
        before, after = (
            getattr(params, section).model_dump(),
            getattr(resolved, section).model_dump(),
        )
        assert {k for k in before if before[k] != after[k]} <= locations
    assert (
        resolved.select == params.select
        and resolved.cluster.kmeans_output == params.cluster.kmeans_output
    )


def test_a_run_that_would_land_on_the_joint_output_is_rejected(params):
    clash = params.model_copy(
        update={
            "pretrain": params.pretrain.model_copy(
                update={"output_dir": "models/wavlm-large-si-100h"}
            )
        }
    )
    with pytest.raises(ValueError, match="overlaps the joint run"):
        experiment.resolve(clash, "si-100h")


# ── no resuming across experiments ──────────────────────────────────────


def _manifest(exp, languages=("sinhala",)):
    exp.manifest_dir.mkdir(parents=True, exist_ok=True)
    exp.train_manifest.write_text(
        "".join(json.dumps({"id": str(i), "language": lg}) + "\n" for i, lg in enumerate(languages))
    )


def test_fresh_run_is_stamped_and_may_restart(params, in_tmp):
    resolved, exp = experiment.resolve(params, "si-100h")
    _manifest(exp)
    experiment.check_resume(exp, resolved, None)
    stamp = json.loads((exp.model_dir / experiment.PROVENANCE_FILE).read_text())
    assert stamp["experiment"] == "si-100h" and stamp["language"] == "sinhala"
    experiment.check_resume(exp, resolved, None)  # same run again: fine


def test_run_refuses_a_directory_stamped_by_another_run(params, in_tmp):
    resolved, exp = experiment.resolve(params, "si-100h")
    _manifest(exp)
    exp.model_dir.mkdir(parents=True)
    (exp.model_dir / experiment.PROVENANCE_FILE).write_text(
        json.dumps({"experiment": "ta-100h", "language": "tamil"})
    )
    with pytest.raises(ValueError, match="different run"):
        experiment.check_resume(exp, resolved, None)


def test_run_refuses_unstamped_checkpoints(params, in_tmp):
    resolved, exp = experiment.resolve(params, "ta-100h")
    _manifest(exp, ("tamil",))
    (exp.model_dir / "checkpoint-1500").mkdir(parents=True)  # e.g. copied from the joint run
    with pytest.raises(ValueError, match="cannot be proven"):
        experiment.check_resume(exp, resolved, None)


def test_changed_training_data_is_not_resumed(params, in_tmp):
    resolved, exp = experiment.resolve(params, "si-100h")
    _manifest(exp)
    experiment.check_resume(exp, resolved, None)
    _manifest(exp, ("sinhala", "sinhala"))
    with pytest.raises(ValueError, match="train_manifest_sha256"):
        experiment.check_resume(exp, resolved, None)


@pytest.mark.parametrize(
    "other",
    ["models/wavlm-large-ta-100h/checkpoint-3000", "models/wavlm-large-si-ta-200h/checkpoint-9000"],
)
def test_sinhala_run_cannot_resume_from_tamil_or_joint_checkpoint(params, in_tmp, other):
    resolved, exp = experiment.resolve(params, "si-100h")
    _manifest(exp)
    with pytest.raises(ValueError, match="only resume its own"):
        experiment.check_resume(exp, resolved, other)
    experiment.check_resume(exp, resolved, str(exp.model_dir / "checkpoint-3000"))  # its own: ok


def test_resolve_points_labels_and_shards_at_the_runs_own_files(params):
    resolved, exp = experiment.resolve(params, "ta-100h")
    assert exp.labels_path == Path("data/labels/ta-100h/wavlm_train_labels.jsonl.gz")
    assert resolved.pretrain.labels_path == str(exp.labels_path)
    assert resolved.cluster.labels_output == str(exp.labels_path)
    assert resolved.cluster.kmeans_output == params.cluster.kmeans_output  # joint k-means reused
    assert resolved.pretrain.labels_path != params.pretrain.labels_path


def test_train_refuses_shards_that_are_not_the_runs_language(params, in_tmp, monkeypatch):
    """Real (tiny, CPU) train(): shards carrying no/other language never get trained on."""
    pytest.importorskip("lhotse")
    import torch

    from pipeline.train import train
    from tests.test_train_e2e import _params as tiny_params
    from tests.test_train_e2e import _setup

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base, shars, labels = _setup(in_tmp)  # cuts without a language tag
    _, exp = experiment.resolve(params, "si-100h")
    _manifest(exp)
    tiny = tiny_params(in_tmp, base, shars, labels, max_updates=2)
    tiny = tiny.model_copy(
        update={"pretrain": tiny.pretrain.model_copy(update={"output_dir": str(exp.model_dir)})}
    )
    with pytest.raises(ValueError, match="expected only 'sinhala'"):
        train(tiny, exp)
    assert not list(exp.model_dir.glob("checkpoint-*"))


# ── report ──────────────────────────────────────────────────────────────


def _bench(name: str, scores: dict[str, dict[str, float]]):
    """scores: {"asr_sinhala": {"wer": .4}, ...} -> reports/bench/<name>/metrics.json"""
    results = [
        {
            "name": entry,
            "language": entry.split("_")[-1] if entry != "sid" else "multilingual",
            "status": "ok",
            "seed": 0,
            "metrics": metrics,
        }
        for entry, metrics in scores.items()
    ]
    path = Path("reports/bench") / name
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"results": results}))


def test_metrics_split_by_language_with_sid_shared(in_tmp):
    _bench("m", {"asr_sinhala": {"wer": 0.3}, "asr_tamil": {"wer": 0.6}, "sid": {"accuracy": 0.9}})
    assert load_upstream_metrics("m", "sinhala")["asr"]["mean"] == 0.3
    assert load_upstream_metrics("m", "tamil")["asr"]["mean"] == 0.6
    assert load_upstream_metrics("m", "tamil")["sid"]["mean"] == 0.9
    assert load_upstream_metrics("m")["asr"]["mean"] == pytest.approx(0.45)  # unchanged behaviour


def _cfg(cross: bool = True) -> dict:
    return {
        "experiments": {"backbone_tag": "wavlm-large"},
        "benchmark": {
            "tasks": "asr,sid,er",
            "cross_language": cross,
            "ablation_rows": [
                {"upstream": "frozen", "mode": "frozen", "train_language": "—"},
                {"upstream": "joint", "mode": "joint CoPT", "train_language": "sinhala+tamil"},
                {
                    "upstream": "si",
                    "mode": "single",
                    "train_language": "sinhala",
                    "experiment": "si-100h",
                },
            ],
        },
    }


def test_ablation_table_shows_dashes_for_missing_results_and_never_invents_any(in_tmp):
    _bench(
        "frozen",
        {"asr_sinhala": {"wer": 0.5}, "asr_tamil": {"wer": 0.7}, "er_tamil": {"accuracy": 0.4}},
    )
    header, rows = ablation_table(_cfg())
    assert header[:6] == [
        "Backbone",
        "Adaptation",
        "Train language",
        "Selected hours",
        "Checkpoint",
        "Benchmark language",
    ]
    by = {(r[1], r[5]): r for r in rows}
    assert by[("frozen", "sinhala")][6] == "0.500 ± 0.000" and by[("frozen", "sinhala")][8] == "—"
    assert by[("frozen", "tamil")][8] == "0.400 ± 0.000"
    assert all(c == "—" for c in by[("joint CoPT", "sinhala")][6:])  # not benchmarked yet
    assert all(c == "—" for c in by[("single", "sinhala (own)")][6:])
    assert by[("single", "sinhala (own)")][3] == "—"  # no selection stats yet


def test_ablation_cross_language_rows_are_optional(in_tmp):
    languages = lambda cfg: [r[5] for r in ablation_table(cfg)[1] if r[1] == "single"]
    assert languages(_cfg(cross=True)) == ["sinhala (own)", "tamil"]
    assert languages(_cfg(cross=False)) == ["sinhala (own)"]


def test_ablation_row_reports_hours_and_checkpoint_of_its_experiment(in_tmp):
    Path("reports/si-100h").mkdir(parents=True)
    Path("reports/si-100h/selection_stats.json").write_text(
        json.dumps({"hours_by_language": {"sinhala": 94.48}})
    )
    Path("reports/si-100h/checkpoint_selection.json").write_text(
        json.dumps({"best_checkpoint": "checkpoint-9000"})
    )
    row = next(r for r in ablation_table(_cfg())[1] if r[1] == "single")
    assert row[3] == "94.5" and row[4] == "checkpoint-9000"
