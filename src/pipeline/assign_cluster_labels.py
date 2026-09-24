"""Assign k-means cluster ids to every training cut's MFCC frames -- the
masked-prediction targets pipeline.train reads via pretrain.labels_path.

Usage:
  python -m pipeline.assign_cluster_labels --config params.yaml
"""

import argparse
import gzip
import json
from pathlib import Path

import joblib

from pipeline.mfcc_features import (
    SAMPLE_RATE,
    cut_target_length,
    default_hubert_config,
    extract_mfcc_aligned,
    load_final_train_cuts,
    load_waveform,
)
from pipeline.schema import Params


def assign_cut(cut, km, config, n_mfcc: int) -> list[int]:
    target_length = cut_target_length(config, cut)
    if target_length <= 0:
        return []
    waveform = load_waveform(cut)
    feats = extract_mfcc_aligned(waveform, SAMPLE_RATE, target_length, n_mfcc)
    return km.predict(feats).astype(int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    cfg = params.cluster

    km = joblib.load(cfg.kmeans_output)
    config = default_hubert_config(params.pretrain.base_model)
    cuts = load_final_train_cuts(params)
    print(f"Assigning cluster labels to {len(cuts)} cuts")

    out_path = Path(cfg.labels_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for i, cut in enumerate(cuts):
            labels = assign_cut(cut, km, config, cfg.n_mfcc)
            if not labels:
                n_empty += 1
                continue
            f.write(json.dumps({"id": cut.id, "labels": labels}) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(cuts)}")

    if n_empty:
        print(f"WARNING: {n_empty} cuts produced no frames and were skipped")
    print(f"Wrote labels for {len(cuts) - n_empty} cuts to {out_path}")


if __name__ == "__main__":
    main()
