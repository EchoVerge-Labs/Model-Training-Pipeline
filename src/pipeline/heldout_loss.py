"""Held-out pre-training loss of saved checkpoints: an overfitting check that does not
touch the training job.

Training only logs the loss on the data it is fitting. This scores each checkpoint
with the SAME objective (contrastive + diversity, per masked position) on
  - dev:   the run's held-out clips (data/manifests/<run>/dev.jsonl; never trained on)
  - train: an equally sized random sample of the run's TRAINING clips
under identical conditions (eval mode, fixed masks), so the two are comparable.
A growing dev-minus-train gap across checkpoints (and a dev loss that stops falling or
rises) is the overfitting signature.

CPU by default, so a running GPU job is left alone:
  python -m pipeline.heldout_loss --experiment si-100h --include-base
"""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining

from pipeline import experiment
from pipeline.datamodule import SAMPLE_RATE, normalize_waveform, pad_batch
from pipeline.train import (
    TRAINING_STATE_FILE,
    compute_mask_and_negatives,
    get_gumbel_temperature,
)

WINDOW_SECONDS = 30.0  # as in shard.max_cut_seconds: the length training sees at most


def split_windows(wave: torch.Tensor, max_seconds: float = WINDOW_SECONDS) -> list[torch.Tensor]:
    """Equal windows of at most max_seconds (same rule as shard.split_long_cuts)."""
    n = math.ceil(wave.shape[0] / (max_seconds * SAMPLE_RATE))
    size = math.ceil(wave.shape[0] / n)
    return [
        wave[i * size : (i + 1) * size] for i in range(n) if wave[i * size : (i + 1) * size].numel()
    ]


def sample_manifest(path: str | Path, max_hours: float, seed: int) -> list[dict]:
    """A seeded random subset of the manifest rows, up to max_hours of audio."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    random.Random(seed).shuffle(rows)
    out, total = [], 0.0
    for row in rows:
        if total >= max_hours * 3600:
            break
        out.append(row)
        total += row["duration"]
    return out


def load_windows(rows: list[dict], raw_dir: str | Path, normalize: bool) -> list[torch.Tensor]:
    windows = []
    for row in rows:
        data, sr = sf.read(Path(raw_dir) / row["filename"], dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['filename']}: expected {SAMPLE_RATE} Hz, got {sr}")
        wave = torch.from_numpy(data[:, 0].copy())
        windows += split_windows(normalize_waveform(wave) if normalize else wave)
    return windows


def batches(windows: list[torch.Tensor], max_padded_seconds: float) -> list[dict]:
    """Length-sorted batches within a padded-seconds budget (deterministic order)."""
    out, cur = [], []
    for w in sorted(windows, key=lambda x: x.shape[0]):
        if cur and (len(cur) + 1) * w.shape[0] > max_padded_seconds * SAMPLE_RATE:
            out.append(pad_batch(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(pad_batch(cur))
    return out


@torch.no_grad()
def evaluate(model, data: list[dict], pretrain, step: int, device, seed: int = 0) -> dict:
    """Mean contrastive / diversity loss per masked position over `data`."""
    model.eval().to(device)
    model.set_gumbel_temperature(
        get_gumbel_temperature(
            step,
            pretrain.max_gumbel_temperature,
            pretrain.min_gumbel_temperature,
            pretrain.gumbel_temperature_decay,
        )
    )
    np.random.seed(seed)  # masks + negatives come from numpy: same for every checkpoint
    contrastive = diversity = masked = 0.0
    for batch in data:
        x, mask = batch["input_values"].to(device), batch["attention_mask"].to(device)
        mti, neg, _ = compute_mask_and_negatives(
            model, x, mask, pretrain.mask_time_prob, pretrain.mask_time_length, device
        )
        out = model(x, attention_mask=mask, mask_time_indices=mti, sampled_negative_indices=neg)
        n = mti.sum().item()
        masked += n
        contrastive += out.contrastive_loss.item()
        diversity += out.diversity_loss.item()
    return {
        "contrastive": contrastive / masked,
        "diversity": diversity / masked,
        "loss": (contrastive + pretrain.diversity_loss_weight * diversity) / masked,
    }


def checkpoint_dirs(model_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for p in model_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and (p / TRAINING_STATE_FILE).exists():
            found.append((int(m.group(1)), p))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="params.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument(
        "--hours", type=float, default=0.25, help="audio per split (dev and train sample)"
    )
    ap.add_argument(
        "--checkpoints", nargs="*", help="checkpoint dirs (default: all in the run's model dir)"
    )
    ap.add_argument(
        "--include-base", action="store_true", help="also score the base model (step 0)"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-batch-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    params, exp = experiment.load(args.config, args.experiment)
    pretrain = params.pretrain
    # as in train.py: whether utterances are normalised follows the base model
    normalize = bool(Wav2Vec2FeatureExtractor.from_pretrained(pretrain.base_model).do_normalize)
    device = torch.device(args.device)

    dev_rows = sample_manifest(exp.dev_manifest, args.hours, args.seed)
    train_rows = sample_manifest(
        exp.train_manifest, sum(r["duration"] for r in dev_rows) / 3600, args.seed
    )
    sets = {
        name: batches(load_windows(rows, args.raw_dir, normalize), args.max_batch_seconds)
        for name, rows in (("dev", dev_rows), ("train", train_rows))
    }
    print(f"{exp.name}: {len(dev_rows)} dev clips, {len(train_rows)} train-sample clips")

    targets = [
        (int(p.name.split("-")[1]), p) for p in map(Path, args.checkpoints or [])
    ] or checkpoint_dirs(exp.model_dir)
    if args.include_base:
        targets = [(0, Path(pretrain.base_model))] + targets
    if not targets:
        raise SystemExit(f"no checkpoints in {exp.model_dir} yet")

    rows = []
    for step, path in targets:
        model = Wav2Vec2ForPreTraining.from_pretrained(str(path))
        scores = {k: evaluate(model, v, pretrain, step, device, args.seed) for k, v in sets.items()}
        row = {
            "step": step,
            "train_loss": scores["train"]["loss"],
            "dev_loss": scores["dev"]["loss"],
            "gap": scores["dev"]["loss"] - scores["train"]["loss"],
            "dev_contrastive": scores["dev"]["contrastive"],
            "train_contrastive": scores["train"]["contrastive"],
        }
        rows.append(row)
        print(
            f"step {step:>6}  train-sample {row['train_loss']:.4f}  dev {row['dev_loss']:.4f}  "
            f"gap {row['gap']:+.4f}"
        )

    out = exp.reports_dir / "heldout_loss.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    print("Overfitting signature: dev loss stops falling or rises while the gap keeps widening.")


if __name__ == "__main__":
    main()
