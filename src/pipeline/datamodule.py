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


def pad_batch(waveforms: list[torch.Tensor], label_seqs: list[list[int]] | None = None) -> dict:
    """1-D waveforms -> {"input_values": (B, T) zero-padded, "attention_mask": (B, T)}.

    If label_seqs is given, also returns "labels": (B, T_frames) padded with
    -100 (nn.CrossEntropyLoss's default ignore_index), one sequence per
    waveform in the same order.
    """
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
                yield pad_batch(waveforms)
                continue
            label_seqs = []
            for _, cut_id in batch:
                if cut_id not in self.labels_by_id:
                    raise KeyError(
                        f"no cluster labels for cut '{cut_id}' in the labels file -- "
                        f"re-run `make labels` after the last `make shard`"
                    )
                label_seqs.append(self.labels_by_id[cut_id])
            yield pad_batch(waveforms, label_seqs)


def create_dataloader(
    shar_dir: str,
    per_device_max_seconds: float,
    num_workers: int = 4,
    seed: int = 0,
    labels_path: str | None = None,
) -> DataLoader:
    """DataLoader over padded batches of at most per_device_max_seconds each.

    labels_path, if given, is loaded once here (not per worker) and attaches a
    -100-padded "labels" tensor to each batch for masked-prediction training.
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
    )
    return DataLoader(
        dataset,
        batch_size=None,  # the dataset already yields whole batches
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
