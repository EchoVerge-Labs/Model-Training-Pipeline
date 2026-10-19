"""Data loading from Lhotse Shar tarballs for pre-training.

The dataset yields ready-made batches (use DataLoader(batch_size=None)) packed
by *seconds of padded audio*, not by clip count, so per-step memory is bounded
whatever the clip lengths. Every clip that is read ends up in a batch -- nothing
is dropped to fit a budget.
"""

import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

try:
    from lhotse import CutSet

    HAS_LHOTSE = True
except ImportError:
    HAS_LHOTSE = False

SAMPLE_RATE = 16000


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Zero-mean / unit-variance over one un-padded utterance -- what
    Wav2Vec2FeatureExtractor does when do_normalize is set (XLS-R expects it, and
    downstream eval applies it). Identical to the HuBERT-Large / WavLM-Large branches."""
    return (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)


def pad_batch(waveforms: list[torch.Tensor]) -> dict:
    """1-D waveforms -> {"input_values": (B, T) zero-padded, "attention_mask": (B, T)}."""
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.zeros(len(waveforms), max_len)
    attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w
        attention_mask[i, : w.shape[0]] = 1
    return {"input_values": padded, "attention_mask": attention_mask}


def pack_batches(
    waveforms: list[torch.Tensor], max_padded_seconds: float, rng: random.Random
) -> list[list[torch.Tensor]]:
    """Group waveforms into batches whose padded size (n_items * longest item) is
    at most max_padded_seconds, then shuffle the batches.

    Sorting by length first keeps neighbours similar in length, so padding
    waste is small. Every waveform lands in exactly one batch; an item longer
    than the budget on its own becomes a batch of one rather than being dropped.
    """
    budget_samples = max_padded_seconds * SAMPLE_RATE
    batches: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    for w in sorted(waveforms, key=lambda x: x.shape[0]):
        # ascending order: w is the longest in `current` once appended
        if current and (len(current) + 1) * w.shape[0] > budget_samples:
            batches.append(current)
            current = []
        current.append(w)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def _cut_language(cut) -> str | None:
    lang = (getattr(cut, "custom", None) or {}).get("language")
    return str(lang).lower() if lang is not None else None


class SharPretrainingDataset(IterableDataset):
    """Streams padded, seconds-budgeted batches from Shar tarballs.

    Shards are shuffled each epoch (seed + epoch) and split so every DDP rank
    and every DataLoader worker reads a disjoint subset -- without that, all
    workers and ranks would read identical data. Clips are shuffled across a
    `buffer_size` window and packed into batches within it.
    """

    def __init__(
        self,
        shar_dir: str,
        max_batch_seconds: float,
        shuffle: bool = True,
        seed: int = 0,
        buffer_size: int = 256,
        normalize: bool = False,
        language: str | None = None,
    ):
        super().__init__()
        if not HAS_LHOTSE:
            raise ImportError("lhotse is required: pip install lhotse")
        self.shar_dir = Path(shar_dir)
        self.max_batch_seconds = max_batch_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.normalize = normalize
        # With a language, only that language's cuts are read (matched on the cut's
        # `language` field, which shard.py takes from the manifest) and every batch
        # is tagged with it. None: the original mixed stream, batches untagged.
        self.language = language.lower() if language else None
        # Advances on every __iter__. Persistent DataLoader workers keep their
        # copy of the dataset alive, so each epoch reshuffles shards differently.
        self._epoch = 0

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        worker = torch.utils.data.get_worker_info()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        rng = random.Random(hash((self.seed, epoch, rank, worker.id if worker else 0)))

        cuts = CutSet.from_shar(
            in_dir=str(self.shar_dir),
            split_for_dataloading=True,
            shuffle_shards=self.shuffle,
            seed=self.seed + epoch,
            stateful_shuffle=False,
        )

        if self.language:
            cuts = cuts.filter(lambda c: _cut_language(c) == self.language)

        buffer: list[torch.Tensor] = []
        for cut in cuts:
            if self.language and _cut_language(cut) != self.language:
                raise ValueError(
                    f"{cut.id}: language {_cut_language(cut)!r} in the {self.language} stream"
                )
            if cut.sampling_rate != SAMPLE_RATE:
                raise ValueError(
                    f"{cut.id}: expected {SAMPLE_RATE} Hz audio, got {cut.sampling_rate}"
                )
            wave = torch.from_numpy(cut.load_audio()[0]).float()
            buffer.append(normalize_waveform(wave) if self.normalize else wave)
            if len(buffer) >= self.buffer_size:
                yield from self._flush(buffer, rng)
                buffer = []
        if buffer:
            yield from self._flush(buffer, rng)

    def _flush(self, buffer: list[torch.Tensor], rng: random.Random):
        for batch in pack_batches(buffer, self.max_batch_seconds, rng):
            padded = pad_batch(batch)
            if self.language:
                padded["languages"] = [self.language] * len(batch)
            yield padded


def _check_shard_count(shar_dir: Path, num_workers: int) -> None:
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    n_shards = len(list(shar_dir.glob("cuts.*.jsonl.gz")))
    needed = max(1, num_workers) * world_size
    if n_shards < needed:
        raise ValueError(
            f"{shar_dir} has {n_shards} shards but {world_size} rank(s) x {max(1, num_workers)} "
            f"worker(s) need at least {needed} (each gets its own subset) -- "
            f"lower num_workers or shard with a smaller shard_size"
        )


def create_dataloader(
    shar_dir: str,
    per_device_max_seconds: float,
    num_workers: int = 4,
    seed: int = 0,
    normalize: bool = False,
    language: str | None = None,
) -> DataLoader:
    """DataLoader over padded batches of at most per_device_max_seconds each.
    normalize applies per-utterance zero-mean/unit-variance (the base model's
    do_normalize flag -- the caller reads it). language restricts the stream to
    one language and tags its batches (see SharPretrainingDataset)."""
    shar_dir = Path(shar_dir)
    _check_shard_count(shar_dir, num_workers)
    dataset = SharPretrainingDataset(
        str(shar_dir),
        max_batch_seconds=per_device_max_seconds,
        seed=seed,
        normalize=normalize,
        language=language,
    )
    return DataLoader(
        dataset,
        batch_size=None,  # the dataset already yields whole batches
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def create_language_dataloaders(
    shar_dir: str,
    languages: list[str],
    per_device_max_seconds: float,
    num_workers: int = 4,
    seed: int = 0,
    normalize: bool = False,
) -> dict[str, DataLoader]:
    """One single-language stream per language over the SAME shar directory, each
    with the same per-device audio-seconds budget as the mixed stream -- so a
    homogeneous micro-batch is the same size as a mixed one, just one language.
    Only the stream a given update uses is read from; the others sit idle."""
    return {
        lang: create_dataloader(
            shar_dir, per_device_max_seconds, num_workers, seed, normalize, language=lang
        )
        for lang in languages
    }
