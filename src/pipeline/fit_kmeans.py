"""Fit k-means on MFCC frames sampled from the training set -- HuBERT
iteration-1 pseudo-label targets.

Usage:
  python -m pipeline.fit_kmeans --config params.yaml
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from pipeline.mfcc_features import (
    SAMPLE_RATE,
    cut_target_length,
    default_hubert_config,
    extract_mfcc_aligned,
    load_final_train_cuts,
    load_waveform,
)
from pipeline.schema import Params


def sample_cuts(cuts, sample_hours: float, seed: int) -> list:
    """Seeded shuffle, then take cuts up to sample_hours of audio."""
    cuts = list(cuts)
    random.Random(seed).shuffle(cuts)
    budget = sample_hours * 3600
    total = 0.0
    sampled = []
    for cut in cuts:
        if total >= budget:
            break
        sampled.append(cut)
        total += cut.duration
    return sampled


def collect_frames(cuts, config, n_mfcc: int) -> np.ndarray:
    frames = []
    for cut in cuts:
        target_length = cut_target_length(config, cut)
        if target_length <= 0:
            continue
        waveform = load_waveform(cut)
        frames.append(extract_mfcc_aligned(waveform, SAMPLE_RATE, target_length, n_mfcc))
    if not frames:
        raise RuntimeError("no MFCC frames collected -- check the sampled cuts")
    return np.concatenate(frames, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    cfg = params.cluster

    cuts = load_final_train_cuts(params)
    sampled = sample_cuts(cuts, cfg.sample_hours, cfg.seed)
    sampled_hours = sum(c.duration for c in sampled) / 3600
    print(
        f"Sampled {len(sampled)} cuts ({sampled_hours:.1f}h) of {len(cuts)} total for k-means fit"
    )

    config = default_hubert_config(params.pretrain.base_model)
    features = collect_frames(sampled, config, cfg.n_mfcc)
    print(f"Collected {features.shape[0]} MFCC frames, dim={features.shape[1]}")

    km = MiniBatchKMeans(
        n_clusters=cfg.num_clusters,
        random_state=cfg.seed,
        batch_size=max(1024, cfg.num_clusters * 10),
        n_init="auto",
    )
    km.fit(features)

    out_path = Path(cfg.kmeans_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out_path)
    print(f"Saved k-means model to {out_path}")

    _, counts = np.unique(km.labels_, return_counts=True)
    stats = {
        "num_clusters": cfg.num_clusters,
        "frames_used": int(features.shape[0]),
        "sampled_cuts": len(sampled),
        "sampled_hours": round(sampled_hours, 2),
        "inertia": float(km.inertia_),
        "cluster_usage_min": int(counts.min()) if len(counts) else 0,
        "cluster_usage_max": int(counts.max()) if len(counts) else 0,
        "clusters_used": len(counts),
    }
    reports_path = Path("reports/kmeans_stats.json")
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reports_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {reports_path}: {stats}")


if __name__ == "__main__":
    main()
