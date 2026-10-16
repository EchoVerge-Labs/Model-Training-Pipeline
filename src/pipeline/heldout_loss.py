"""Held-out masked-prediction loss of saved checkpoints: an overfitting check that does
not touch the training job.

Training only logs the loss on the data it is fitting. This scores each checkpoint with
the SAME objective (cross-entropy over masked frames, plus masked accuracy) on
  - dev:   the run's held-out clips (data/manifests/<run>/dev.jsonl; never trained on)
  - train: an equally sized random sample of the run's TRAINING clips
under identical conditions (eval mode, fixed masks, no utterance mixing), so the two are
comparable. A growing dev-minus-train gap across checkpoints, with dev loss that stops
falling or rises, is the overfitting signature.

Targets are the joint run's k-means cluster ids (cluster.kmeans_output), assigned to
these clips exactly as assign_cluster_labels does for the training shards.

CPU by default, so a running GPU job is left alone:
  python -m pipeline.heldout_loss --experiment si-100h
"""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
from torch import nn

from pipeline import experiment
from pipeline.checkpoints import STATE_FILE
from pipeline.datamodule import SAMPLE_RATE, pad_batch
from pipeline.layer_features import base_normalizes, layer_features, load_feature_model
from pipeline.train import compute_mask
from pipeline.wavlm_model import WavLMForMaskedPrediction

WINDOW_SECONDS = 30.0  # as in shard.max_cut_seconds: the length training sees at most


def split_windows(wave: torch.Tensor, max_seconds: float = WINDOW_SECONDS) -> list[torch.Tensor]:
    """Equal windows of at most max_seconds (same rule as shard.split_long_cuts)."""
    n = math.ceil(wave.shape[0] / (max_seconds * SAMPLE_RATE))
    size = math.ceil(wave.shape[0] / n)
    parts = [wave[i * size : (i + 1) * size] for i in range(n)]
    return [p for p in parts if p.numel()]


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


def load_windows(rows: list[dict], raw_dir: str | Path) -> list[torch.Tensor]:
    """Raw (un-normalised) windows of the manifest clips."""
    windows = []
    for row in rows:
        data, sr = sf.read(Path(raw_dir) / row["filename"], dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['filename']}: expected {SAMPLE_RATE} Hz, got {sr}")
        windows += split_windows(torch.from_numpy(data[:, 0].copy()))
    return windows


def label_windows(windows, km, feature_model, normalize: bool, device) -> list[list[int]]:
    """Nearest-centroid cluster id per frame, as assign_cluster_labels does."""
    labels = []
    for w in windows:
        feats = layer_features(feature_model, w, normalize, device)
        labels.append(km.predict(feats).astype(int).tolist() if len(feats) else [])
    return labels


def make_batches(windows, labels, normalize: bool, max_padded_seconds: float) -> list[dict]:
    """Length-sorted batches within a padded-seconds budget (deterministic order)."""
    order = sorted(range(len(windows)), key=lambda i: windows[i].shape[0])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * windows[i].shape[0] > max_padded_seconds * SAMPLE_RATE:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return [
        pad_batch([windows[i] for i in idx], [labels[i] for i in idx], normalize=normalize)
        for idx in out
    ]


@torch.no_grad()
def evaluate(model, data: list[dict], pretrain, device, seed: int = 0) -> dict:
    """Masked-frame cross-entropy and accuracy over `data` (eval mode, fixed masks)."""
    model.eval().to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    np.random.seed(seed)  # masks come from numpy: same for every checkpoint
    total = correct = n = 0.0
    for batch in data:
        x, mask = batch["input_values"].to(device), batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        mti, _ = compute_mask(
            model, x, mask, pretrain.mask_time_prob, pretrain.mask_time_length, device
        )
        logits = model(x, attention_mask=mask, mask_time_indices=mti).float()
        common = min(logits.shape[1], labels.shape[1])
        logits, mti = logits[:, :common], mti[:, :common]
        target = labels[:, :common].clone()
        target[~mti] = -100
        valid = target != -100
        if not valid.any():
            continue
        total += loss_fn(logits.transpose(1, 2), target).item()
        correct += (logits.argmax(-1)[valid] == target[valid]).sum().item()
        n += valid.sum().item()
    return {"loss": total / n, "masked_accuracy": correct / n}


def checkpoint_dirs(model_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for p in model_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and (p / STATE_FILE).exists():
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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-batch-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    params, exp = experiment.load(args.config, args.experiment)
    pretrain, cluster = params.pretrain, params.cluster
    device = torch.device(args.device)
    normalize = base_normalizes(pretrain.base_model)

    targets = [(int(Path(p).name.split("-")[1]), Path(p)) for p in args.checkpoints or []]
    targets = targets or checkpoint_dirs(exp.model_dir)
    if not targets:
        raise SystemExit(f"no checkpoints in {exp.model_dir} yet")

    dev_rows = sample_manifest(exp.dev_manifest, args.hours, args.seed)
    train_rows = sample_manifest(
        exp.train_manifest, sum(r["duration"] for r in dev_rows) / 3600, args.seed
    )
    km = joblib.load(cluster.kmeans_output)
    feature_model = load_feature_model(pretrain.base_model, cluster.layer, device)
    sets = {}
    for name, rows in (("dev", dev_rows), ("train", train_rows)):
        windows = load_windows(rows, args.raw_dir)
        labels = label_windows(windows, km, feature_model, normalize, device)
        sets[name] = make_batches(windows, labels, normalize, args.max_batch_seconds)
    del feature_model
    print(f"{exp.name}: {len(dev_rows)} dev clips, {len(train_rows)} train-sample clips")

    rows = []
    for step, path in targets:
        model = WavLMForMaskedPrediction(pretrain.base_model, cluster.num_clusters)
        model.load_checkpoint(path)
        s = {k: evaluate(model, v, pretrain, device, args.seed) for k, v in sets.items()}
        row = {
            "step": step,
            "train_loss": s["train"]["loss"],
            "dev_loss": s["dev"]["loss"],
            "gap": s["dev"]["loss"] - s["train"]["loss"],
            "train_masked_acc": s["train"]["masked_accuracy"],
            "dev_masked_acc": s["dev"]["masked_accuracy"],
        }
        rows.append(row)
        print(
            f"step {step:>6}  train-sample {row['train_loss']:.4f}  dev {row['dev_loss']:.4f}  "
            f"gap {row['gap']:+.4f}  dev-acc {row['dev_masked_acc']:.3f}"
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
