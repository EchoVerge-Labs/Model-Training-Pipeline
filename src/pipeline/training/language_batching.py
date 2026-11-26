"""Language-homogeneous optimizer updates for joint Sinhala + Tamil pre-training.

One shared model and one optimizer still learn from both languages, but every
optimizer update is built from a single language: all gradient-accumulation
micro-batches of the update, on every DDP rank, come from that language.

Ported from the experiment/*-language-homogeneous-batches branches; both trainers use it
when pretrain.language_batching.enabled (runs named `...-homog`). Three pieces:
  - LanguageSchedule: which language update N uses. A pure function of
    (seed, N) -- no state, no communication -- so every rank computes the same
    answer, and a resumed run (which knows its global step) continues exactly
    where an uninterrupted one would be.
  - LanguageBatchSource: one batch stream per language (see
    datamodule.create_language_dataloaders), restarted when exhausted.
  - LanguageTracker: per-language accounting plus the "never mixed" assertions.
"""

import random
from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist

from pipeline.common.schema import LanguageBatchingConfig

SAMPLE_RATE = 16000


class MixedLanguageError(RuntimeError):
    """An optimizer update would not be single-language. Training stops."""


# ── Schedule ────────────────────────────────────────────────────────────


def _block_counts(languages: list[str], probabilities: dict[str, float], window: int) -> list[int]:
    """Updates per language in one block of `window`: probability * window, rounded
    by largest remainder so the counts sum to exactly `window`."""
    exact = [probabilities[lang] * window for lang in languages]
    counts = [int(x) for x in exact]
    by_remainder = sorted(range(len(exact)), key=lambda i: (exact[i] - counts[i], -i), reverse=True)
    for i in by_remainder[: window - sum(counts)]:
        counts[i] += 1
    return counts


class LanguageSchedule:
    """Which language optimizer update `update` (0-based) is built from.

    balanced_random: updates are grouped in blocks of `window`; each block holds
        every language exactly in proportion to its probability (so over any whole
        number of blocks the split is exact: 50/50 for two languages at 0.5/0.5),
        and the order within a block is a seeded shuffle. Random enough not to
        impose a period on the model, bounded enough that neither language can
        run away (a run of one language is at most `window` updates long at 50/50).
    alternate: strict round-robin over `languages`.
    """

    def __init__(self, cfg: LanguageBatchingConfig):
        self.mode = cfg.mode
        self.languages = list(cfg.languages)
        self.probabilities = dict(cfg.probabilities)
        self.window = cfg.window
        self.seed = cfg.seed
        self._counts = _block_counts(self.languages, self.probabilities, self.window)
        self._blocks: dict[int, list[str]] = {}

    def _block(self, index: int) -> list[str]:
        if index not in self._blocks:
            block = [lang for lang, n in zip(self.languages, self._counts) for _ in range(n)]
            # a string seed is hashed deterministically (unlike hash(tuple of str))
            random.Random(f"language-schedule:{self.seed}:{index}").shuffle(block)
            if len(self._blocks) > 4096:  # bound memory on very long runs
                self._blocks.clear()
            self._blocks[index] = block
        return self._blocks[index]

    def language_at(self, update: int) -> str:
        if update < 0:
            raise ValueError(f"update must be >= 0, got {update}")
        if self.mode == "alternate":
            return self.languages[update % len(self.languages)]
        return self._block(update // self.window)[update % self.window]

    def counts_before(self, update: int) -> dict[str, int]:
        """How many of updates [0, update) each language had -- lets a resumed run
        restore its per-language update counters without storing them."""
        counts = dict.fromkeys(self.languages, 0)
        for u in range(update):
            counts[self.language_at(u)] += 1
        return counts


# ── Batch source ────────────────────────────────────────────────────────


class LanguageBatchSource:
    """next(language) -> the next padded batch of that language's stream.

    Each stream is restarted when exhausted (a new epoch of that language), the
    same way the single-stream loop restarts its iterator.
    """

    def __init__(self, loaders: dict[str, Iterable]):
        self.loaders = loaders
        self._iters: dict[str, Iterator] = {}

    def next(self, language: str) -> dict:
        if language not in self.loaders:
            raise KeyError(
                f"no batch stream for language {language!r} (have {sorted(self.loaders)})"
            )
        if language not in self._iters:
            self._iters[language] = iter(self.loaders[language])
        try:
            return next(self._iters[language])
        except StopIteration:
            self._iters[language] = iter(self.loaders[language])
            return next(self._iters[language])


# ── Accounting + assertions ─────────────────────────────────────────────


def batch_languages(batch: dict) -> set[str]:
    """The set of languages in a micro-batch (from its per-item `languages`)."""
    if "languages" not in batch:
        raise MixedLanguageError(
            "batch carries no `languages` -- it did not come from a language-aware stream"
        )
    return {str(lang) for lang in batch["languages"]}


class LanguageTracker:
    """Per-language update counts, audio seconds and losses, plus the checks that
    make a mixed-language update impossible to miss."""

    def __init__(self, languages: list[str]):
        self.languages = list(languages)
        self.updates = dict.fromkeys(languages, 0)
        # cumulative audio seconds, summed over ranks; synced by reduce_seconds()
        self.seconds_total = dict.fromkeys(languages, 0.0)
        self._seconds_unreduced = dict.fromkeys(languages, 0.0)
        # running window (reset after each log)
        self._loss = dict.fromkeys(languages, 0.0)
        self._microbatches = dict.fromkeys(languages, 0)
        self.microbatch_violations = 0
        self.accumulation_violations = 0
        self.rank_violations = 0
        self._window_languages: set[str] = set()

    # -- per-update checks --

    def begin_update(self) -> None:
        self._window_languages = set()

    def check_microbatch(self, batch: dict, expected: str) -> None:
        langs = batch_languages(batch)
        if langs != {expected}:
            self.microbatch_violations += 1
            raise MixedLanguageError(
                f"micro-batch has languages {sorted(langs)}, but this update is {expected!r}"
            )
        self._window_languages |= langs

    def end_update(self, language: str) -> None:
        if self._window_languages != {language}:
            self.accumulation_violations += 1
            raise MixedLanguageError(
                f"gradient-accumulation window used languages {sorted(self._window_languages)}, "
                f"expected only {language!r}"
            )
        self.updates[language] += 1

    # -- accounting --

    def add_microbatch(self, language: str, seconds: float, loss: float) -> None:
        """loss: the micro-batch's loss in the unit the trainer logs as train/loss."""
        self._seconds_unreduced[language] += seconds
        self._loss[language] += loss
        self._microbatches[language] += 1

    def reduce_seconds(self, device) -> None:
        """Fold the audio seen since the last call, summed over all ranks, into
        seconds_total. A collective: every rank must call it at the same step."""
        local = torch.tensor(
            [self._seconds_unreduced[lang] for lang in self.languages],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        for lang, seconds in zip(self.languages, local.tolist()):
            self.seconds_total[lang] += seconds
            self._seconds_unreduced[lang] = 0.0

    def window_metrics(self) -> dict[str, float]:
        """Mean loss per micro-batch of each language since the last reset_window()."""
        out = {}
        for lang in self.languages:
            n = self._microbatches[lang]
            out[f"{lang}/loss"] = self._loss[lang] / n if n else float("nan")
        return out

    def reset_window(self) -> None:
        for lang in self.languages:
            self._loss[lang] = 0.0
            self._microbatches[lang] = 0

    def summary(self) -> dict[str, float]:
        total_updates = sum(self.updates.values())
        out: dict[str, float] = {}
        for lang in self.languages:
            out[f"lang/{lang}/updates"] = self.updates[lang]
            out[f"lang/{lang}/audio_hours"] = round(self.seconds_total[lang] / 3600, 4)
            out[f"lang/{lang}/update_share_pct"] = (
                round(100 * self.updates[lang] / total_updates, 3) if total_updates else 0.0
            )
        return out

    # -- checkpoint state --

    def state_dict(self) -> dict:
        # `updates` is not stored: the schedule recomputes it from the step.
        return {f"seconds_{lang}": self.seconds_total[lang] for lang in self.languages}

    def load_state(self, state: dict | None, updates: dict[str, int]) -> None:
        self.updates = dict(updates)
        for lang in self.languages:
            self.seconds_total[lang] = float((state or {}).get(f"seconds_{lang}", 0.0))


def assert_ranks_agree(language: str, languages: list[str], device, tracker=None) -> None:
    """Every rank must be building this update from the same language, or DDP's
    gradient all-reduce would average languages together. Raises on all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    idx = torch.tensor([languages.index(language)], device=device)
    lo, hi = idx.clone(), idx.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if lo.item() != hi.item():
        if tracker is not None:
            tracker.rank_violations += 1
        raise MixedLanguageError(
            f"ranks disagree on this update's language (this rank: {language!r}; "
            f"language indices across ranks span {lo.item()}..{hi.item()})"
        )


def language_curves_header(languages: list[str]) -> str:
    cols = ["step", "current_language"]
    for lang in languages:
        cols += [
            f"{lang}_updates",
            f"{lang}_update_share_pct",
            f"{lang}_audio_hours",
            f"{lang}_loss",
        ]
    return ",".join(cols)


def log_languages(tracker: LanguageTracker, language: str, step: int, curves_path) -> dict:
    """At a log step: per-language metrics for MLflow (lang/<language>/...), one row of the
    language curves CSV, and a console line. Resets the per-language loss window."""
    import math

    summary = tracker.summary()
    window = tracker.window_metrics()
    metrics = dict(summary)
    for key, value in window.items():
        if not math.isnan(value):  # a language with no micro-batch in this window
            metrics[f"lang/{key}"] = round(value, 4)
    shares = ", ".join(
        f"{lang}: {summary[f'lang/{lang}/updates']:.0f} upd "
        f"({summary[f'lang/{lang}/update_share_pct']:.1f}%) "
        f"{summary[f'lang/{lang}/audio_hours']:.2f}h"
        for lang in tracker.languages
    )
    print(f"  languages | {shares} | last update: {language}")
    row = [str(step), language]
    for lang in tracker.languages:
        row += [
            f"{summary[f'lang/{lang}/updates']:.0f}",
            f"{summary[f'lang/{lang}/update_share_pct']:.3f}",
            f"{summary[f'lang/{lang}/audio_hours']:.4f}",
            f"{window[f'{lang}/loss']:.6f}",
        ]
    with open(curves_path, "a") as f:
        f.write(",".join(row) + "\n")
    tracker.reset_window()
    return metrics
