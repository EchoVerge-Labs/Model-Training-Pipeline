"""Language-homogeneous updates (pretrain.language_batching, runs named ...-homog), ported
from the experiment/*-language-homogeneous-batches branches: every optimizer update -- all its
gradient-accumulation micro-batches, on every rank -- comes from one language."""

import json

import numpy as np
import pytest
import torch

from pipeline.common.schema import LanguageBatchingConfig
from pipeline.common.tracking import RUN_FILE, run_record
from pipeline.training.contrastive import train
from pipeline.training.language_batching import (
    LanguageBatchSource,
    LanguageSchedule,
    LanguageTracker,
    MixedLanguageError,
)
from tests.test_resume import _make_base, _params

LANGS = ["sinhala", "tamil"]


def _cfg(**kw) -> LanguageBatchingConfig:
    return LanguageBatchingConfig(enabled=True, **kw).resolve(LANGS)


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
    lb = _cfg(window=4, seed=7)
    return params.model_copy(
        update={"pretrain": params.pretrain.model_copy(update={"language_batching": lb})}
    )


def _run(params, tmp_path, log, **kw):
    loaders = {lang: RecordingLoader(lang, log) for lang in LANGS}
    return train(
        params, dataloader=loaders, reports_dir=tmp_path / "reports", device="cuda",
        use_mlflow=False, **kw,
    )  # fmt: skip


# ── schedule ──


def test_both_languages_appear_and_split_is_50_50_over_whole_windows():
    sched = LanguageSchedule(_cfg(window=20, seed=1))
    seq = [sched.language_at(i) for i in range(2000)]
    assert seq.count("sinhala") == seq.count("tamil") == 1000
    for block in range(100):  # exact within every window, not just overall
        assert seq[block * 20 : (block + 1) * 20].count("sinhala") == 10
    longest = max(
        len(run)
        for run in "".join("s" if x == "sinhala" else "t" for x in seq).replace("t", " ").split()
    )
    assert 1 < longest <= 20  # not strict alternation, but bounded


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


def test_every_rank_computes_the_same_language_for_every_update():
    """No communication is needed: the language is a pure function of (seed, update)."""
    ranks = [LanguageSchedule(_cfg(seed=11)) for _ in range(4)]
    for update in range(300):
        assert len({r.language_at(update) for r in ranks}) == 1


def test_probabilities_must_cover_the_languages_and_sum_to_one():
    with pytest.raises(ValueError, match="match languages"):
        _cfg(probabilities={"sinhala": 1.0})
    with pytest.raises(ValueError, match="sum to 1"):
        _cfg(probabilities={"sinhala": 0.7, "tamil": 0.7})


# ── streams / micro-batches ──


def _two_language_shars(tmp_path, n_each=24):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("lhotse")
    from lhotse import CutSet, MonoCut, Recording

    from pipeline.data.shard import shard_cuts

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
    from pipeline.common.datamodule import create_dataloader, create_language_dataloaders

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
            n = batch["input_values"].shape[0]
            assert padded * n <= budget + 1e-6 or n == 1
            n_clips += n
        total[lang] = n_clips
    assert total == {"sinhala": 24, "tamil": 24}  # each stream is exactly its language

    first = next(iter(create_dataloader(str(shars), budget, num_workers=0, seed=1)))
    assert "languages" not in first  # the mixed stream is untouched


def test_the_shard_language_check_covers_every_cut(tmp_path):
    from pipeline.data.shard import check_shard_languages

    shars = _two_language_shars(tmp_path, n_each=6)
    assert check_shard_languages(str(shars), LANGS) == 12
    with pytest.raises(ValueError, match="expected only"):
        check_shard_languages(str(shars), ["sinhala"])


def test_the_batch_source_restarts_an_exhausted_language_stream():
    class Finite:
        def __init__(self, lang):
            self.lang = lang

        def __iter__(self):
            return iter([_batch(self.lang)] * 2)

    src = LanguageBatchSource({"sinhala": Finite("sinhala"), "tamil": Finite("tamil")})
    assert all(src.next("sinhala")["languages"][0] == "sinhala" for _ in range(5))


# ── training loop (GPU) ──


def test_gradient_accumulation_windows_never_mix_languages(tmp_path):
    base = _make_base(tmp_path)
    log: list[str] = []
    params = _homogeneous_params(tmp_path, base, grad_accum=3, max_updates=8)
    assert params.run.endswith("-homog")
    assert _run(params, tmp_path, log) == 8
    assert len(log) == 8 * 3
    sched = LanguageSchedule(params.pretrain.language_batching)
    for update in range(8):
        window = log[update * 3 : (update + 1) * 3]
        assert window == [sched.language_at(update)] * 3  # all three micro-batches, one language
    metrics = json.loads((tmp_path / "reports" / "pretrain_metrics.json").read_text())
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
            device="cuda",
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
    longer = _homogeneous_params(tmp_path, base, max_updates=8, output_dir=str(tmp_path / "b"))
    # a longer schedule is another config: hand the folder over deliberately (see test_resume)
    (tmp_path / "b" / RUN_FILE).write_text(json.dumps(run_record(longer)))
    second: list[str] = []
    _run(longer, tmp_path, second)

    assert first + second == straight
    metrics = json.loads((tmp_path / "reports" / "pretrain_metrics.json").read_text())
    assert metrics["lang/sinhala/updates"] == metrics["lang/tamil/updates"] == 4


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
    out = train(
        params, dataloader=plain, reports_dir=tmp_path / "reports", device="cuda", use_mlflow=False
    )
    assert out == 2
    assert not (tmp_path / "reports" / "pretrain_language_curves.csv").exists()
