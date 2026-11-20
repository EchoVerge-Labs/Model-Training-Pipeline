"""Assign k-means cluster ids to every training cut's layer features -- the
masked-prediction targets pipeline.train reads via pretrain.labels_path.

Usage:
  python -m pipeline.assign_cluster_labels --config params.yaml
"""

import argparse
import gzip
import json
from pathlib import Path

import joblib
import torch

from pipeline.layer_features import (
    base_normalizes,
    layer_features,
    load_feature_model,
    load_shar_cuts,
    load_waveform,
)
from pipeline.schema import Params


def assign_cut(cut, km, extract) -> list[int]:
    """Nearest-centroid cluster id per frame, from extract(waveform) -> (T', D)."""
    feats = extract(load_waveform(cut))
    if len(feats) == 0:
        return []
    return km.predict(feats).astype(int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    cfg = params.cluster

    km = joblib.load(cfg.kmeans_output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_feature_model(params.pretrain.base_model, cfg.layer, device)
    normalize = base_normalizes(params.pretrain.base_model)

    def extract(waveform):
        return layer_features(model, waveform, normalize, device)

    cuts = load_shar_cuts(params.shard.output_dir)
    print(f"Assigning cluster labels to every cut in {params.shard.output_dir}")

    out_path = Path(cfg.labels_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")  # never leave a half-written file
    n_written = n_empty = 0
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        for cut in cuts:
            labels = assign_cut(cut, km, extract)
            if not labels:
                n_empty += 1
                continue
            f.write(json.dumps({"id": cut.id, "labels": labels}) + "\n")
            n_written += 1
            if n_written % 1000 == 0:
                print(f"  {n_written} cuts labelled")
    tmp_path.replace(out_path)

    if n_empty:
        print(f"WARNING: {n_empty} cuts produced no frames and were skipped")
    print(f"Wrote labels for {n_written} cuts to {out_path}")


if __name__ == "__main__":
    main()
