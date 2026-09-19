"""Data loading from Lhotse Shar tarballs for pre-training."""
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path

try:
    from lhotse import CutSet
    from lhotse.dataset import DynamicBucketingSampler
    from lhotse.shar import SharSource
    HAS_LHOTSE = True
except ImportError:
    HAS_LHOTSE = False


class SharPretrainingDataset(IterableDataset):
    """Streams audio from Shar tarballs for SSL pre-training.

    Each item is a dict with:
    - "input_values": 1D float tensor (raw waveform, 16 kHz)
    - "duration": float (seconds)
    """

    def __init__(self, shar_dir: str, shuffle: bool = True, seed: int = 0):
        super().__init__()
        self.shar_dir = Path(shar_dir)
        self.shuffle = shuffle
        self.seed = seed

        if not HAS_LHOTSE:
            raise ImportError("lhotse is required: pip install lhotse[webdataset]")

    def __iter__(self):
        cuts = CutSet.from_shar(
            fields={"recording": str(self.shar_dir / "recording.*.tar")},
            cuts=sorted(self.shar_dir.glob("cuts.*.jsonl.gz")),
            shuffle_shards=self.shuffle,
            seed=self.seed + _get_worker_seed(),
        )

        for cut in cuts:
            audio = cut.load_audio()  # numpy array
            yield {
                "input_values": torch.from_numpy(audio.squeeze()).float(),
                "duration": cut.duration,
            }


def _get_worker_seed() -> int:
    """Get a per-worker seed offset for multi-worker data loading."""
    worker_info = torch.utils.data.get_worker_info()
    return worker_info.id if worker_info else 0


def collate_pretrain(batch: list[dict], max_seconds: float = 200.0) -> dict:
    """Collate variable-length waveforms with dynamic padding.

    Also enforces per-batch max-seconds budget.
    """
    # Sort by length descending for efficient padding
    batch = sorted(batch, key=lambda x: x["input_values"].shape[0], reverse=True)

    # Budget enforcement: keep adding until we hit max_seconds
    total_seconds = 0.0
    kept = []
    for item in batch:
        total_seconds += item["duration"]
        if total_seconds > max_seconds and kept:
            break
        kept.append(item)

    # Pad to longest in this batch
    max_len = max(item["input_values"].shape[0] for item in kept)
    padded = torch.zeros(len(kept), max_len)
    attention_mask = torch.zeros(len(kept), max_len, dtype=torch.long)

    for i, item in enumerate(kept):
        length = item["input_values"].shape[0]
        padded[i, :length] = item["input_values"]
        attention_mask[i, :length] = 1

    return {
        "input_values": padded,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    shar_dir: str,
    per_device_max_seconds: float,
    num_workers: int = 4,
    seed: int = 0,
) -> DataLoader:
    """Create a DataLoader for pre-training."""
    dataset = SharPretrainingDataset(shar_dir, shuffle=True, seed=seed)
    return DataLoader(
        dataset,
        batch_size=64,          # max batch; collate_pretrain enforces seconds budget
        num_workers=num_workers,
        collate_fn=lambda b: collate_pretrain(b, max_seconds=per_device_max_seconds),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
