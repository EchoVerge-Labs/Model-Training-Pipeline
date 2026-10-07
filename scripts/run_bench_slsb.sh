#!/usr/bin/env bash
# SLSB v0.3 (all six task families, seeds 0,1,2), one upstream after the other.
#   scripts/run_bench_slsb.sh [<model dir> <out name>]...
# With no arguments: the proxy-selected checkpoints of the HuBERT-Large and
# mHuBERT-147 continued pre-training runs.
set -uo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PATH=~/SLSB-benchmark/.venv/bin:$PATH
set -a && source .env && set +a

TASKS=asr,sid,er,sd,asv,ic
SEEDS=0,1,2
MLFLOW_URI=https://dagshub.com/EchoVerge-LABS/Model-Training-Pipeline.mlflow
DATA_DIR=../SLSB-benchmark/data

run() {  # <checkpoint dir> <out name>
    echo "=== $(date '+%F %T') start $2 ($1)"
    bash scripts/run_benchmark.sh "$1" "$TASKS" "$SEEDS" "reports/bench/$2" "$MLFLOW_URI" "$DATA_DIR"
    echo "=== $(date '+%F %T') end $2 (exit $?)"
}

if [ $# -eq 0 ]; then
    set -- models/hubert-large-si-ta-200h/checkpoint-4500 hubert_large_adapted_v0.3 \
           models/mhubert147-si-ta-200h/checkpoint-9000 mhubert147_adapted_v0.3
fi
while [ $# -ge 2 ]; do
    run "$1" "$2"
    shift 2
done
