#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around `slsb run` for the dvc.yaml `benchmark` stage.
#
# slsb writes its results to <out>/results_<upstream-with-slashes-as-__>.json
# (see SLSB-benchmark's src/slsb/cli.py) -- a filename that depends on the
# --upstream string. dvc.yaml needs a fixed, predictable `metrics:` path, so
# this copies whatever slsb actually produced to <out>/metrics.json.

UPSTREAM="${1:?usage: run_benchmark.sh <upstream> <tasks> <seeds> <out_dir> [mlflow_uri] [data_dir] [run_prefix] [run_tags]}"
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
# slsb names its MLflow runs "<upstream>_<task>_<language>" -- for a local
# upstream that is "models/selected_asr_tamil", the same for every seed and every
# model. With a run prefix, the runs are renamed <prefix>_<task>_<language>_s<seed>
# and tagged (key=value,key=value) once slsb is done; see label_benchmark_runs.py.
RUN_PREFIX="${7:-}"
RUN_TAGS="${8:-}"

ARGS=(run --upstream "$UPSTREAM" --tasks "$TASKS" --seeds "$SEEDS" --out "$OUT_DIR"
      --data-dir "$DATA_DIR" --params "$SLSB_PARAMS")
if [ -n "$MLFLOW_URI" ]; then
    ARGS+=(--mlflow-uri "$MLFLOW_URI")
fi

START_MS="$(date +%s%3N)"
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

if [ -n "$MLFLOW_URI" ] && [ -n "$RUN_PREFIX" ] && [ -n "${DAGSHUB_TOKEN:-}" ]; then
    LABEL_ARGS=(--mlflow-uri "$MLFLOW_URI" --upstream "$UPSTREAM" --since-ms "$START_MS"
                --prefix "$RUN_PREFIX" --tags "$RUN_TAGS")
    # models/selected is a copy of whichever checkpoint select_checkpoint picked.
    if [ "$UPSTREAM" = "models/selected" ]; then
        LABEL_ARGS+=(--selection-report reports/checkpoint_selection.json)
    fi
    # The scores are already on disk and in MLflow; a failed rename is not worth
    # failing a multi-hour stage over, and the command can simply be run again.
    python scripts/label_benchmark_runs.py "${LABEL_ARGS[@]}" \
        || echo "WARNING: could not label the MLflow runs; re-run: python scripts/label_benchmark_runs.py ${LABEL_ARGS[*]}" >&2
fi
