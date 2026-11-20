#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around `slsb run` for the dvc.yaml `benchmark` stage.
#
# slsb writes its results to <out>/results_<upstream-with-slashes-as-__>.json
# (see SLSB-benchmark's src/slsb/cli.py) -- a filename that depends on the
# --upstream string. dvc.yaml needs a fixed, predictable `metrics:` path, so
# this copies whatever slsb actually produced to <out>/metrics.json.

UPSTREAM="${1:?usage: run_benchmark.sh <upstream> <tasks> <seeds> <out_dir> [mlflow_uri] [data_dir]}"
TASKS="${2:?}"
SEEDS="${3:?}"
OUT_DIR="${4:?}"
MLFLOW_URI="${5:-}"
# slsb defaults --data-dir to ./data, which in this repo is the pre-training
# data, not the benchmark's -- so fall back to the sibling SLSB-benchmark checkout.
DATA_DIR="${6:-../SLSB-benchmark/data}"
# Likewise slsb reads its probe hyperparams from ./params.yaml, which here is the
# pipeline's params file -- use the one that sits next to the benchmark data.
SLSB_PARAMS="$(dirname "$DATA_DIR")/params.yaml"

ARGS=(run --upstream "$UPSTREAM" --tasks "$TASKS" --seeds "$SEEDS" --out "$OUT_DIR"
      --data-dir "$DATA_DIR" --params "$SLSB_PARAMS")
if [ -n "$MLFLOW_URI" ]; then
    ARGS+=(--mlflow-uri "$MLFLOW_URI")
fi

slsb "${ARGS[@]}"

SAFE_UPSTREAM="${UPSTREAM//\//__}"
RESULTS_JSON="$OUT_DIR/results_${SAFE_UPSTREAM}.json"

if [ ! -f "$RESULTS_JSON" ]; then
    echo "ERROR: expected slsb results file not found: $RESULTS_JSON" >&2
    echo "       (looked for it by replacing '/' with '__' in --upstream=$UPSTREAM)" >&2
    exit 1
fi

cp "$RESULTS_JSON" "$OUT_DIR/metrics.json"
echo "Copied $RESULTS_JSON -> $OUT_DIR/metrics.json"
