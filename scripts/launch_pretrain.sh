#!/usr/bin/env bash
set -euo pipefail

# ─── Config ───────────────────────────────────────────
# MODEL picks the model to train (xlsr, wav2vec2, wavlm, hubert, mhubert): its
# configs/models/<MODEL>.yaml is merged over params.yaml. Unset = params.yaml as it is
# (the XLS-R run).
SCRIPT="src/pipeline/train.py"
CONFIG="params.yaml"
MODEL="${MODEL:-}"

# Source NCCL env for GB10
source configs/nccl-env.sh

# DagsHub credentials (DAGSHUB_USER / DAGSHUB_TOKEN) for MLflow logging, if present
if [ -f .env ]; then set -a; source .env; set +a; fi

# ─── Detect number of nodes ──────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching pre-training: model=${MODEL:-xlsr (params.yaml)}, nnodes=$NNODES, node_rank=$NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# ─── Launch ──────────────────────────────────────────
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=1 \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $SCRIPT \
    --config $CONFIG \
    ${MODEL:+--model "$MODEL"} \
    "$@"

echo "Pre-training complete."
