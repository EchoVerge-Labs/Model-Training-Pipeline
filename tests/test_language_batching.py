"""Language-homogeneous optimizer updates: schedule, streams, accumulation, DDP, resume, provenance.

CPU only, tiny random model, synthetic audio; every output goes to pytest's tmp_path.
"""

import json
import os
import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from pipeline.language_batching import (
    LanguageBatchSource,
    LanguageSchedule,
    LanguageTracker,
    MixedLanguageError,
    assert_ranks_agree,
    check_dataset,
)
from pipeline.schema import LanguageBatchingConfig, Params
from pipeline.train import train
from pipeline.variant import assert_isolated, deep_merge, load_raw
from tests.test_resume import REPO_ROOT, _make_base, _params

LANGS = ["sinhala", "tamil"]


def _cfg(**kw) -> LanguageBatchingConfig:
    return LanguageBatchingConfig(enabled=True, **kw)


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _batch(lang, n=2, seconds=1.0):
    length = int(seconds * 16000)
    return {
        "input_values": torch.randn(n, length),
        "attention_mask": torch.ones(n, length, dtype=torch.long),
        "languages": [lang] * n,
    }


class RecordingLoader:
    """Infinite single-language stream; logs each batch the trainer pulls, in order."""

    def __init__(self, lang, log, mixed=False):
        self.lang, self.log, self.mixed = lang, log, mixed

    def __iter__(self):
        while True:
            self.log.append(self.lang)
            b = _batch(self.lang)
            if self.mixed:
                b["languages"] = [self.lang, "other"]
            yield b


def _homogeneous_params(tmp_path, base, grad_accum=1, **pretrain):
    params = _params(
        tmp_path,
        base,
        target_batch_seconds=2.0 * grad_accum,  # per_device 2.0 -> grad_accum micro-batches
        eval_every_updates=2,
        save_every_updates=4,
        **pretrain,
    )
    lb = LanguageBatchingConfig(enabled=True, window=4, seed=7)
    return params.model_copy(
        update={"pretrain": params.pretrain.model_copy(update={"language_batching": lb})}
    )


def _run(params, tmp_path, log, **kw):
    loaders = {lang: RecordingLoader(lang, log) for lang in LANGS}
    return train(
        params, dataloader=loaders, reports_dir=tmp_path / "reports", device="cpu",
        use_mlflow=False, **kw,
    )  # fmt: skip


# ── schedule ──────────────────────────────────────────────────────────────


def test_both_languages_appear_and_split_is_50_50_over_whole_windows():
    sched = LanguageSchedule(_cfg(window=20, seed=1))
    seq = [sched.language_at(i) for i in range(2000)]
    assert set(seq) == set(LANGS)
    assert seq.count("sinhala") == seq.count("tamil") == 1000
    for block in range(100):  # exact within every window, not just overall
        assert seq[block * 20 : (block + 1) * 20].count("sinhala") == 10
    longest = max(
        len(run)
        for run in "".join("s" if x == "sinhala" else "t" for x in seq).replace("t", " ").split()
    )
    assert longest <= 20 and longest > 1  # not strict alternation, but bounded


def test_unbalanced_probabilities_are_respected_exactly_per_window():
    cfg = _cfg(window=10, probabilities={"sinhala": 0.7, "tamil": 0.3})
    seq = [LanguageSchedule(cfg).language_at(i) for i in range(10)]
    assert seq.count("sinhala") == 7 and seq.count("tamil") == 3


def test_alternate_mode_is_strict_round_robin():
    sched = LanguageSchedule(_cfg(mode="alternate"))
    assert [sched.language_at(i) for i in range(4)] == ["sinhala", "tamil", "sinhala", "tamil"]


def test_same_seed_same_schedule_different_seed_different_schedule():
    a = [LanguageSchedule(_cfg(seed=5)).language_at(i) for i in range(200)]
    b = [LanguageSchedule(_cfg(seed=5)).language_at(i) for i in range(200)]
    c = [LanguageSchedule(_cfg(seed=6)).language_at(i) for i in range(200)]
    assert a == b and a != c


def test_schedule_is_independent_of_how_it_is_queried_so_a_resume_continues_it():
    full = LanguageSchedule(_cfg(seed=3))
    uninterrupted = [full.language_at(i) for i in range(100)]
    resumed = LanguageSchedule(_cfg(seed=3))  # a fresh process that starts at update 37
    assert [resumed.language_at(i) for i in range(37, 100)] == uninterrupted[37:]
    assert full.counts_before(37) == {lang: uninterrupted[:37].count(lang) for lang in LANGS}


def test_simulated_ddp_ranks_get_the_same_language_every_update():
    ranks = [LanguageSchedule(_cfg(seed=11)) for _ in range(4)]
    for update in range(300):
        assert len({r.language_at(update) for r in ranks}) == 1


def _gloo_worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        assert_ranks_agree("sinhala", LANGS, torch.device("cpu"))
        try:
            # rank 1 is told a different language -> every rank must raise
            assert_ranks_agree("sinhala" if rank == 0 else "tamil", LANGS, torch.device("cpu"))
            results[rank] = "no error"
        except MixedLanguageError:
            results[rank] = "raised"
    finally:
        dist.destroy_process_group()


def test_a_rank_that_disagrees_stops_every_rank():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.spawn(_gloo_worker, args=(2, port, results), nprocs=2, join=True)
    assert dict(results) == {0: "raised", 1: "raised"}


# ── streams / micro-batches ───────────────────────────────────────────────


sf = pytest.importorskip("soundfile")
lhotse = pytest.importorskip("lhotse")


def _two_language_shars(tmp_path, n_each=24):
    from lhotse import CutSet, MonoCut, Recording

    from pipeline.shard import shard_cuts

    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    cuts = []
    for i in range(2 * n_each):
        lang = LANGS[i % 2]
        path = raw / f"{i}.wav"
        sf.write(path, (0.01 + rng.normal(0, 0.05, 16000 * (1 + i % 5))).astype("float32"), 16000)
        rec = Recording.from_file(path, recording_id=f"{lang}-{i}")
        cuts.append(
            MonoCut(
                id=f"{lang}-{i}",
                start=0,
                duration=rec.duration,
                channel=0,
                recording=rec,
                custom={"language": lang},
            )
        )
    out = tmp_path / "shars"
    shard_cuts(CutSet.from_cuts(cuts), str(out), shard_size=6, audio_format="flac")
    return out


def test_every_microbatch_is_one_language_and_has_the_same_budget_as_the_mixed_stream(tmp_path):
    from pipeline.datamodule import create_dataloader, create_language_dataloaders

    shars = _two_language_shars(tmp_path)
    budget = 12.0
    loaders = create_language_dataloaders(str(shars), LANGS, budget, num_workers=0, seed=1)
    total = {}
    for lang, loader in loaders.items():
        n_clips = 0
        for batch in loader:
            assert set(batch["languages"]) == {lang}
            assert len(batch["languages"]) == batch["input_values"].shape[0]
            padded = batch["input_values"].shape[1] / 16000
            assert (
                padded * batch["input_values"].shape[0] <= budget + 1e-6
                or batch["input_values"].shape[0] == 1
            )
            n_clips += batch["input_values"].shape[0]
        total[lang] = n_clips
    assert total == {"sinhala": 24, "tamil": 24}  # each stream is exactly its language, none lost

    mixed = create_dataloader(str(shars), budget, num_workers=0, seed=1)
    first = next(iter(mixed))
    assert "languages" not in first  # the original stream is untouched


def test_the_batch_source_restarts_an_exhausted_language_stream():
    class Finite:
        def __init__(self, lang):
            self.lang = lang

        def __iter__(self):
            return iter([_batch(self.lang)] * 2)

    src = LanguageBatchSource({"sinhala": Finite("sinhala"), "tamil": Finite("tamil")})
    assert all(src.next("sinhala")["languages"][0] == "sinhala" for _ in range(5))


# ── training loop ─────────────────────────────────────────────────────────


def test_gradient_accumulation_windows_never_mix_languages(tmp_path):
    base = _make_base(tmp_path)
    log: list[str] = []
    params = _homogeneous_params(tmp_path, base, grad_accum=3, max_updates=8)
    assert _run(params, tmp_path, log) == 8
    assert len(log) == 8 * 3
    sched = LanguageSchedule(params.pretrain.language_batching)
    for update in range(8):
        window = log[update * 3 : (update + 1) * 3]
        assert window == [sched.language_at(update)] * 3  # all three micro-batches, one language
    assert set(log) == set(LANGS)
    metrics = json.loads((tmp_path / "reports" / "pretrain_metrics.json").read_text())
    assert metrics["lang/mixed_microbatch_violations"] == 0
    assert metrics["lang/mixed_accumulation_violations"] == 0
    assert metrics["lang/sinhala/updates"] == metrics["lang/tamil/updates"] == 4
    header = (tmp_path / "reports" / "pretrain_language_curves.csv").read_text().splitlines()[0]
    assert "sinhala_audio_hours" in header and "tamil_loss" in header


def test_a_mixed_language_microbatch_stops_training(tmp_path):
    base = _make_base(tmp_path)
    params = _homogeneous_params(tmp_path, base, max_updates=4)
    loaders = {lang: RecordingLoader(lang, [], mixed=True) for lang in LANGS}
    with pytest.raises(MixedLanguageError, match="micro-batch"):
        train(
            params,
            dataloader=loaders,
            reports_dir=tmp_path / "reports",
            device="cpu",
            use_mlflow=False,
        )


def test_a_batch_without_language_tags_is_refused():
    tracker = LanguageTracker(LANGS)
    with pytest.raises(MixedLanguageError, match="languages"):
        tracker.check_microbatch({"input_values": torch.zeros(1, 4)}, "sinhala")


def test_a_window_that_switched_language_midway_is_refused():
    tracker = LanguageTracker(LANGS)
    tracker.begin_update()
    tracker._window_languages = {"sinhala", "tamil"}
    with pytest.raises(MixedLanguageError, match="accumulation"):
        tracker.end_update("sinhala")


def test_resume_at_step_n_continues_the_same_language_sequence(tmp_path):
    base = _make_base(tmp_path)
    straight: list[str] = []
    _run(
        _homogeneous_params(tmp_path, base, max_updates=8, output_dir=str(tmp_path / "a")),
        tmp_path,
        straight,
    )

    first: list[str] = []
    _run(
        _homogeneous_params(tmp_path, base, max_updates=4, output_dir=str(tmp_path / "b")),
        tmp_path,
        first,
    )
    second: list[str] = []
    _run(
        _homogeneous_params(tmp_path, base, max_updates=8, output_dir=str(tmp_path / "b")),
        tmp_path,
        second,
    )

    assert first + second == straight
    metrics = json.loads((tmp_path / "reports" / "pretrain_metrics.json").read_text())
    assert (
        metrics["lang/sinhala/updates"] == metrics["lang/tamil/updates"] == 4
    )  # ratio survives the resume


def test_disabled_keeps_the_original_mixed_path(tmp_path):
    base = _make_base(tmp_path)
    params = _params(tmp_path, base, max_updates=2)
    assert params.pretrain.language_batching.enabled is False
    plain = [
        {
            "input_values": torch.randn(2, 16000),
            "attention_mask": torch.ones(2, 16000, dtype=torch.long),
        }
    ] * 2  # untagged batches work
    assert (
        train(
            params,
            dataloader=plain,
            reports_dir=tmp_path / "reports",
            device="cpu",
            use_mlflow=False,
        )
        == 2
    )
    assert not (tmp_path / "reports" / "pretrain_language_curves.csv").exists()


def test_turning_off_synchronisation_or_accumulation_is_refused_when_enabled():
    with pytest.raises(ValueError, match="mix languages"):
        LanguageBatchingConfig(enabled=True, synchronize_across_ranks=False)
    with pytest.raises(ValueError, match="mix languages"):
        LanguageBatchingConfig(enabled=True, keep_language_across_gradient_accumulation=False)


# ── provenance + isolation ────────────────────────────────────────────────


def _manifest(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rows(n=4):
    return [
        {"id": f"{lang}-{i}", "language": lang, "duration": 10.0}
        for lang in LANGS
        for i in range(n)
    ]


def test_check_dataset_accepts_the_same_clips_and_flags_any_difference(tmp_path):
    ref, new = tmp_path / "ref.jsonl", tmp_path / "new.jsonl"
    _manifest(ref, _rows())
    _manifest(new, _rows())
    ok = check_dataset(str(new), str(ref), LANGS)
    assert ok["ok"] and ok["languages"]["tamil"]["matching_ids"] == 4

    changed = _rows()
    changed[0]["id"] = "sinhala-other"  # a different Sinhala clip
    _manifest(new, changed)
    bad = check_dataset(str(new), str(ref), LANGS)
    assert not bad["ok"] and bad["languages"]["sinhala"]["mismatch_count"] == 2


def test_training_refuses_a_dataset_that_differs_from_the_reference(tmp_path):
    base = _make_base(tmp_path)
    (tmp_path / "data/manifests").mkdir(parents=True)
    _manifest(tmp_path / "data/manifests/train.jsonl", _rows())
    _manifest(tmp_path / "ref.jsonl", _rows(5))
    params = _homogeneous_params(tmp_path, base, max_updates=2)
    lb = params.pretrain.language_batching.model_copy(
        update={"reference_manifest": str(tmp_path / "ref.jsonl")}
    )
    params = params.model_copy(
        update={"pretrain": params.pretrain.model_copy(update={"language_batching": lb})}
    )
    with pytest.raises(RuntimeError, match="differs"):
        _run(params, tmp_path, [])


def test_repo_manifest_matches_main_when_available():
    import subprocess

    if subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "main"], capture_output=True, check=False
    ).returncode:
        pytest.skip("no main branch")
    manifest = REPO_ROOT / "data/manifests/train.jsonl"
    if not manifest.exists():
        pytest.skip("no manifest")
    ref = "git:main:data/manifests/train.jsonl"
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # `git show` needs the repo
    try:
        assert check_dataset(str(manifest), ref, LANGS)["ok"]
    finally:
        os.chdir(cwd)


def test_homogeneous_overlay_changes_only_batching_and_output_locations():
    base = load_raw(str(REPO_ROOT / "params.yaml"))
    merged = load_raw(
        str(REPO_ROOT / "params.yaml"), str(REPO_ROOT / "configs/language_homogeneous.yaml")
    )
    assert_isolated(merged)

    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v

    a, b = dict(flat(base)), dict(flat(merged))
    changed = {k for k in a.keys() & b.keys() if a[k] != b[k]}
    assert changed == {
        "pretrain.output_dir",
        "checkpoint_selection.output_dir",
        "benchmark.mlflow_run_prefix",
        "benchmark.mlflow_run_tags",
    }
    # every other pre-training setting -- optimizer, LR, masking, batch seconds, max_updates -- is untouched
    pre_base = {k: v for k, v in a.items() if k.startswith(("pretrain.", "select.", "shard."))}
    assert all(b[k] == v for k, v in pre_base.items() if k != "pretrain.output_dir")
    params = Params(**merged)
    assert params.pretrain.language_batching.enabled and params.pretrain.max_updates == 13500


@pytest.mark.parametrize(
    "patch",
    [
        {"pretrain": {"output_dir": "models/xlsr300m-si-ta-200h-norm"}},
        {"checkpoint_selection": {"output_dir": "models/selected"}},
        {"experiment": {"reports_dir": "reports"}},
        {"checkpoint_selection": {"report_path": "reports/checkpoint_selection.json"}},
    ],
)
def test_outputs_that_would_overwrite_the_baseline_are_refused(patch):
    merged = load_raw(
        str(REPO_ROOT / "params.yaml"), str(REPO_ROOT / "configs/language_homogeneous.yaml")
    )
    with pytest.raises(ValueError, match="overwrite"):
        assert_isolated(deep_merge(merged, patch))


def test_overlay_file_is_valid_yaml_with_the_documented_defaults():
    cfg = yaml.safe_load((REPO_ROOT / "configs/language_homogeneous.yaml").read_text())
    lb = LanguageBatchingConfig(**cfg["pretrain"]["language_batching"])
    assert lb.probabilities == {"sinhala": 0.5, "tamil": 0.5} and lb.mode == "balanced_random"
