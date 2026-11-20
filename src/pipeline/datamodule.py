"""Data loading from Lhotse Shar tarballs for pre-training.

The dataset yields ready-made batches (use DataLoader(batch_size=None)) packed
by *seconds of padded audio*, not by clip count, so per-step memory is bounded
whatever the clip lengths. Every clip that is read ends up in a batch -- nothing
is dropped to fit a budget.
"""

import gzip
import json
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


def load_labels(labels_path: str) -> dict[str, list[int]]:
    """Reads the {"id": ..., "labels": [...]} gzip JSONL written by
    pipeline.assign_cluster_labels."""
    labels_by_id = {}
    with gzip.open(labels_path, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            labels_by_id[entry["id"]] = entry["labels"]
    return labels_by_id


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Zero-mean / unit-variance over one un-padded utterance -- what
    Wav2Vec2FeatureExtractor does when do_normalize is set (wavlm-large
    expects it). Also used when extracting the k-means layer features, so
    labels and training see identical inputs."""
    return (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)


def mix_utterances(
    waveforms: list[torch.Tensor],
    prob: float,
    rng: random.Random,
    max_overlap: float = 0.5,
    snr_db_range: tuple[float, float] = (-5.0, 5.0),
) -> list[torch.Tensor]:
    """WavLM's utterance mixing: with probability `prob`, overlap each utterance
    with a piece of *another* utterance from the batch.

    The piece is a random crop of at most `max_overlap` of the primary's length,
    scaled so its RMS is the primary's RMS times a gain drawn uniformly from
    `snr_db_range` (dB), and added at a random offset inside the primary. The
    secondary is always taken from the clean inputs (never from an already
    mixed utterance), lengths never change, and a batch of one is left as is.
    The frame-level cluster labels stay those of the clean primary.
    """
    if prob <= 0 or len(waveforms) < 2:
        return list(waveforms)
    mixed = []
    for i, primary in enumerate(waveforms):
        if rng.random() >= prob:
            mixed.append(primary)
            continue
        secondary = waveforms[rng.choice([k for k in range(len(waveforms)) if k != i])]
        longest = min(secondary.shape[0], int(max_overlap * primary.shape[0]))
        if longest < 1:
            mixed.append(primary)
            continue
        crop_len = rng.randint(1, longest)
        start = rng.randint(0, secondary.shape[0] - crop_len)
        crop = secondary[start : start + crop_len]
        crop_rms = crop.pow(2).mean().sqrt()
        primary_rms = primary.pow(2).mean().sqrt()
        if crop_rms <= 0 or primary_rms <= 0:
            mixed.append(primary)
            continue
        gain = primary_rms / crop_rms * 10 ** (rng.uniform(*snr_db_range) / 20)
        offset = rng.randint(0, primary.shape[0] - crop_len)
        out = primary.clone()
        out[offset : offset + crop_len] += crop * gain
        mixed.append(out)
    return mixed


def pad_batch(
    waveforms: list[torch.Tensor],
    label_seqs: list[list[int]] | None = None,
    normalize: bool = False,
    mix_prob: float = 0.0,
    rng: random.Random | None = None,
) -> dict:
    """1-D waveforms -> {"input_values": (B, T) zero-padded, "attention_mask": (B, T)}.

    If normalize is set, each waveform is normalised over its own samples
    *before* padding, so the zeros that pad it stay zero. If mix_prob > 0,
    utterances are then mixed (mix_utterances, needs rng) -- after
    normalisation and without re-normalising, as in fairseq's WavLM.

    If label_seqs is given, also returns "labels": (B, T_frames) padded with
    -100 (nn.CrossEntropyLoss's default ignore_index), one sequence per
    waveform in the same order.
    """
    if normalize:
        waveforms = [normalize_waveform(w) for w in waveforms]
    if mix_prob > 0:
        if rng is None:
            raise ValueError("mix_prob > 0 needs an rng")
        waveforms = mix_utterances(waveforms, mix_prob, rng)
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.zeros(len(waveforms), max_len)
    attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w
        attention_mask[i, : w.shape[0]] = 1
    batch = {"input_values": padded, "attention_mask": attention_mask}

    if label_seqs is not None:
        max_label_len = max(len(labels) for labels in label_seqs)
        label_tensor = torch.full((len(label_seqs), max_label_len), -100, dtype=torch.long)
        for i, labels in enumerate(label_seqs):
            label_tensor[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)
        batch["labels"] = label_tensor

    return batch


def pack_batches(
    items: list,
    max_padded_seconds: float,
    rng: random.Random,
    length_fn=lambda x: x.shape[0],
) -> list[list]:
    """Group items into batches whose padded size (n_items * longest item) is
    at most max_padded_seconds, then shuffle the batches.

    Sorting by length first keeps neighbours similar in length, so padding
    waste is small. Every item lands in exactly one batch; an item longer
    than the budget on its own becomes a batch of one rather than being
    dropped. `items` defaults to plain waveform tensors; pass `length_fn` to
    pack something else (e.g. (waveform, cut_id) pairs) by waveform length.
    """
    budget_samples = max_padded_seconds * SAMPLE_RATE
    batches: list[list] = []
    current: list = []
    for item in sorted(items, key=length_fn):
        # ascending order: item is the longest in `current` once appended
        if current and (len(current) + 1) * length_fn(item) > budget_samples:
            batches.append(current)
            current = []
        current.append(item)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


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
        labels_by_id: dict[str, list[int]] | None = None,
        normalize: bool = False,
        utterance_mix_prob: float = 0.0,
    ):
        super().__init__()
        if not HAS_LHOTSE:
            raise ImportError("lhotse is required: pip install lhotse")
        self.shar_dir = Path(shar_dir)
        self.max_batch_seconds = max_batch_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.labels_by_id = labels_by_id
        self.normalize = normalize
        self.utterance_mix_prob = utterance_mix_prob
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

        buffer: list[tuple[torch.Tensor, str]] = []
        for cut in cuts:
            if cut.sampling_rate != SAMPLE_RATE:
                raise ValueError(
                    f"{cut.id}: expected {SAMPLE_RATE} Hz audio, got {cut.sampling_rate}"
                )
            buffer.append((torch.from_numpy(cut.load_audio()[0]).float(), cut.id))
            if len(buffer) >= self.buffer_size:
                yield from self._flush(buffer, rng)
                buffer = []
        if buffer:
            yield from self._flush(buffer, rng)

    def _flush(self, buffer: list[tuple[torch.Tensor, str]], rng: random.Random):
        for batch in pack_batches(
            buffer, self.max_batch_seconds, rng, length_fn=lambda item: item[0].shape[0]
        ):
            waveforms = [w for w, _ in batch]
            if self.labels_by_id is None:
                yield pad_batch(waveforms, None, self.normalize, self.utterance_mix_prob, rng)
                continue
            label_seqs = []
            for _, cut_id in batch:
                if cut_id not in self.labels_by_id:
                    raise KeyError(
                        f"no cluster labels for cut '{cut_id}' in the labels file -- "
                        f"re-run `make labels` after the last `make shard`"
                    )
                label_seqs.append(self.labels_by_id[cut_id])
            yield pad_batch(waveforms, label_seqs, self.normalize, self.utterance_mix_prob, rng)


def create_dataloader(
    shar_dir: str,
    per_device_max_seconds: float,
    num_workers: int = 4,
    seed: int = 0,
    labels_path: str | None = None,
    normalize: bool = False,
    utterance_mix_prob: float = 0.0,
) -> DataLoader:
    """DataLoader over padded batches of at most per_device_max_seconds each.

    labels_path, if given, is loaded once here (not per worker) and attaches a
    -100-padded "labels" tensor to each batch for masked-prediction training.
    normalize applies per-utterance zero-mean/unit-variance (the base model's
    do_normalize flag -- the caller reads it). utterance_mix_prob > 0 mixes a
    second utterance from the same batch into each one (WavLM; mix_utterances).
    """
    shar_dir = Path(shar_dir)
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    n_shards = len(list(shar_dir.glob("cuts.*.jsonl.gz")))
    needed = max(1, num_workers) * world_size
    if n_shards < needed:
        raise ValueError(
            f"{shar_dir} has {n_shards} shards but {world_size} rank(s) x {max(1, num_workers)} "
            f"worker(s) need at least {needed} (each gets its own subset) -- "
            f"lower num_workers or shard with a smaller shard_size"
        )
    labels_by_id = load_labels(labels_path) if labels_path else None
    dataset = SharPretrainingDataset(
        str(shar_dir),
        max_batch_seconds=per_device_max_seconds,
        seed=seed,
        labels_by_id=labels_by_id,
        normalize=normalize,
        utterance_mix_prob=utterance_mix_prob,
    )
    return DataLoader(
        dataset,
        batch_size=None,  # the dataset already yields whole batches
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
