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
# Optional. slsb defaults --data-dir to ./data (here: the pre-training data) and reads
# its probe hyperparams from ./params.yaml (here: the pipeline's file, which has no
# `asr` section). With a data dir, use the benchmark's own data and params next to it.
DATA_DIR="${6:-}"

ARGS=(run --upstream "$UPSTREAM" --tasks "$TASKS" --seeds "$SEEDS" --out "$OUT_DIR")
if [ -n "$DATA_DIR" ]; then
    ARGS+=(--data-dir "$DATA_DIR" --params "$(dirname "$DATA_DIR")/params.yaml")
fi
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
