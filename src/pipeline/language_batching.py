"""Language-homogeneous optimizer updates for joint Sinhala + Tamil pre-training.

One shared model and one optimizer still learn from both languages, but every
optimizer update is built from a single language: all gradient-accumulation
micro-batches of the update, on every DDP rank, come from that language.

Three pieces:
  - LanguageSchedule: which language update N uses. A pure function of
    (seed, N) -- no state, no communication -- so every rank computes the same
    answer, and a resumed run (which knows its global step) continues exactly
    where an uninterrupted one would be.
  - LanguageBatchSource: one batch stream per language (see
    datamodule.create_language_dataloaders), restarted when exhausted.
  - LanguageTracker: per-language accounting plus the "never mixed" assertions.
"""

import json
import random
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
import torch.distributed as dist

from pipeline.schema import LanguageBatchingConfig

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
        self._contrastive = dict.fromkeys(languages, 0.0)
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

    def add_microbatch(
        self, language: str, seconds: float, loss: float, contrastive_per_mask: float
    ) -> None:
        self._seconds_unreduced[language] += seconds
        self._loss[language] += loss
        self._contrastive[language] += contrastive_per_mask
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
            out[f"{lang}/contrastive_loss"] = self._contrastive[lang] / n if n else float("nan")
        return out

    def reset_window(self) -> None:
        for lang in self.languages:
            self._loss[lang] = self._contrastive[lang] = 0.0
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
        out["lang/mixed_microbatch_violations"] = self.microbatch_violations
        out["lang/mixed_accumulation_violations"] = self.accumulation_violations
        out["lang/rank_disagreements"] = self.rank_violations
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


# ── Dataset provenance ──────────────────────────────────────────────────


def _read_manifest_text(ref: str) -> str:
    if ref.startswith("git:"):
        _, rev, path = ref.split(":", 2)
        out = subprocess.run(
            ["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=False
        )
        if out.returncode != 0:
            raise FileNotFoundError(f"cannot read {ref}: {out.stderr.strip()}")
        return out.stdout
    return Path(ref).read_text(encoding="utf-8")


def _load_manifest(text: str) -> dict[str, tuple[str, float]]:
    clips = {}
    for line in text.splitlines():
        if line.strip():
            e = json.loads(line)
            clips[e["id"]] = (e["language"], float(e["duration"]))
    return clips


def check_dataset(
    manifest_path: str,
    reference: str,
    languages: list[str],
    shar_dir: str | None = None,
    shar_hours_tolerance: float = 0.02,
) -> dict:
    """Compare the manifest this run trains on with the original joint run's.

    The comparison is by clip id, per language. If shar_dir is given, the cut
    manifests there (cheap -- no audio is read) are also summed per language and
    must account for the manifest's hours to within shar_hours_tolerance (cuts
    shorter than min_cut_seconds are dropped when sharding, so slightly under).
    """
    new = _load_manifest(Path(manifest_path).read_text(encoding="utf-8"))
    ref = _load_manifest(_read_manifest_text(reference))
    report: dict = {
        "manifest": manifest_path,
        "reference": reference,
        "languages": {},
        "total_mismatches": 0,
    }
    for lang in languages:
        ref_ids = {i for i, (lg, _) in ref.items() if lg == lang}
        new_ids = {i for i, (lg, _) in new.items() if lg == lang}
        entry = {
            "original_clip_count": len(ref_ids),
            "new_clip_count": len(new_ids),
            "matching_ids": len(ref_ids & new_ids),
            "missing_from_new": len(ref_ids - new_ids),
            "extra_in_new": len(new_ids - ref_ids),
            "original_hours": round(sum(ref[i][1] for i in ref_ids) / 3600, 4),
            "new_hours": round(sum(new[i][1] for i in new_ids) / 3600, 4),
        }
        entry["mismatch_count"] = entry["missing_from_new"] + entry["extra_in_new"]
        report["total_mismatches"] += entry["mismatch_count"]
        report["languages"][lang] = entry

    if shar_dir is not None:
        import gzip

        hours = dict.fromkeys(languages, 0.0)
        count = dict.fromkeys(languages, 0)
        for f in sorted(Path(shar_dir).glob("cuts.*.jsonl.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    cut = json.loads(line)
                    lang = (cut.get("custom") or {}).get("language")
                    if lang in hours:
                        hours[lang] += cut["duration"] / 3600
                        count[lang] += 1
        for lang in languages:
            entry = report["languages"][lang]
            entry["shar_cuts"] = count[lang]
            entry["shar_hours"] = round(hours[lang], 4)
            have, want = hours[lang], entry["new_hours"]
            entry["shar_hours_ok"] = (
                want > 0 and have <= want * 1.001 and have >= want * (1 - shar_hours_tolerance)
            )
            if not entry["shar_hours_ok"]:
                report["total_mismatches"] += 1
    report["ok"] = report["total_mismatches"] == 0
    return report


def write_dataset_report(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    import argparse

    from pipeline.schema import Params

    parser = argparse.ArgumentParser(description="verify the run reuses the original 200h clips")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--overlay", default=None)
    parser.add_argument("--manifest", default="data/manifests/train.jsonl")
    parser.add_argument("--shar-dir", default="data/shars")
    args = parser.parse_args()

    lb = Params.from_yaml(args.config, args.overlay).pretrain.language_batching
    if not lb.reference_manifest:
        raise SystemExit("language_batching.reference_manifest is not set")
    shar_dir = args.shar_dir if Path(args.shar_dir).is_dir() else None
    report = check_dataset(args.manifest, lb.reference_manifest, lb.languages, shar_dir)
    write_dataset_report(report, lb.dataset_check_report)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(
            f"ERROR: dataset differs from the original run -- {report['total_mismatches']} mismatch(es)"
        )
    print(f"OK: same clips as {lb.reference_manifest}. Report: {lb.dataset_check_report}")


if __name__ == "__main__":
    main()
