#!/usr/bin/env bash
set -euo pipefail

# ─── Config ───────────────────────────────────────────
# Runs src/pipeline/train.py, which picks the trainer from the model's objective family.
# Choose the model with `--model <key>` (mhubert147|hubert|wavlm|xlsr|wav2vec2) or
# `--config <params file>`; with neither it trains mHuBERT-147 from params.yaml.
# Any other argument goes to the trainer, e.g. --resume-from CKPT_DIR / --no-resume.
SCRIPT="${SCRIPT:-src/pipeline/train.py}"

# Source NCCL env for GB10
source configs/nccl-env.sh

# DagsHub credentials (DAGSHUB_USER / DAGSHUB_TOKEN) for MLflow logging, if present
if [ -f .env ]; then set -a; source .env; set +a; fi

# ─── Detect number of nodes ──────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching pre-training: nnodes=$NNODES, node_rank=$NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# ─── Launch ──────────────────────────────────────────
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=1 \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $SCRIPT \
    "$@"

echo "Pre-training complete."
