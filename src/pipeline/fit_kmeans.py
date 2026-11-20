"""Fit k-means on layer features of the pretrained model, sampled from the
training set -- the cluster centroids that define WavLM's masked-prediction
targets (see pipeline.layer_features).

Usage:
  python -m pipeline.fit_kmeans --config params.yaml
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from pipeline.layer_features import (
    base_normalizes,
    layer_features,
    load_feature_model,
    load_shar_cuts,
    load_waveform,
)
from pipeline.schema import Params

FRAMES_PER_SECOND = 50  # WavLMModel's CNN stride: 320 samples at 16 kHz


def collect_frames(
    cuts, extract, sample_hours: float, max_frames: int, seed: int
) -> tuple[np.ndarray, int, float]:
    """Streams cuts until sample_hours of audio, features of each via
    extract(waveform) -> (T', D), randomly thinned so about max_frames survive
    (fairseq HuBERT clusters a sample of frames, not all of them). Returns
    (features, n_cuts, hours). `cuts` should already be in random order."""
    rng = np.random.default_rng(seed)
    budget = sample_hours * 3600
    keep_fraction = min(1.0, max_frames / (budget * FRAMES_PER_SECOND))
    frames, n_cuts, seconds = [], 0, 0.0
    for cut in cuts:
        if seconds >= budget:
            break
        feats = extract(load_waveform(cut))
        n_keep = min(len(feats), max(1, round(len(feats) * keep_fraction)))
        frames.append(feats[rng.choice(len(feats), size=n_keep, replace=False)])
        n_cuts += 1
        seconds += cut.duration
    if not frames:
        raise RuntimeError("no frames collected -- is the Shar directory empty?")
    features = np.concatenate(frames, axis=0)
    if len(features) > max_frames:
        features = features[rng.choice(len(features), size=max_frames, replace=False)]
    return features, n_cuts, seconds / 3600


def fit_kmeans(features: np.ndarray, num_clusters: int, seed: int) -> MiniBatchKMeans:
    """Same settings as fairseq's HuBERT learn_kmeans.py."""
    km = MiniBatchKMeans(
        n_clusters=num_clusters,
        init="k-means++",
        batch_size=10000,
        n_init=20,
        max_no_improvement=100,
        reassignment_ratio=0.0,
        random_state=seed,
    )
    return km.fit(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    cfg = params.cluster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cuts = load_shar_cuts(params.shard.output_dir, shuffle=True, seed=cfg.seed)
    model = load_feature_model(params.pretrain.base_model, cfg.layer, device)
    normalize = base_normalizes(params.pretrain.base_model)
    features, n_cuts, sampled_hours = collect_frames(
        cuts,
        lambda w: layer_features(model, w, normalize, device),
        cfg.sample_hours,
        cfg.max_frames,
        cfg.seed,
    )
    model = None  # free the feature model (GPU memory) before k-means
    print(
        f"Read {n_cuts} cuts ({sampled_hours:.1f}h) -> {features.shape[0]} frames of "
        f"layer {cfg.layer}, dim={features.shape[1]}"
    )

    km = fit_kmeans(features, cfg.num_clusters, cfg.seed)

    out_path = Path(cfg.kmeans_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out_path)
    print(f"Saved k-means model to {out_path}")

    _, counts = np.unique(km.labels_, return_counts=True)
    stats = {
        "layer": cfg.layer,
        "num_clusters": cfg.num_clusters,
        "frames_used": int(features.shape[0]),
        "sampled_cuts": n_cuts,
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
