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

# DagsHub credentials (DAGSHUB_USER / DAGSHUB_TOKEN) for MLflow logging, if present.
# Values already set in the shell win over .env, so `NNODES=2 NODE_RANK=1 make copt`
# is honoured (the Makefile promises this): remember them, then restore after sourcing.
_SHELL_NNODES=${NNODES-}; _SHELL_NODE_RANK=${NODE_RANK-}
_SHELL_MASTER_ADDR=${MASTER_ADDR-}; _SHELL_MASTER_PORT=${MASTER_PORT-}
if [ -f .env ]; then set -a; source .env; set +a; fi
NNODES=${_SHELL_NNODES:-${NNODES-}}; NODE_RANK=${_SHELL_NODE_RANK:-${NODE_RANK-}}
MASTER_ADDR=${_SHELL_MASTER_ADDR:-${MASTER_ADDR-}}; MASTER_PORT=${_SHELL_MASTER_PORT:-${MASTER_PORT-}}

# ─── Detect number of nodes ──────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching FULL continued pre-training (CoPT; not parameter-efficient: no adapters/LoRA/GC-LoRA): model=${MODEL:-xlsr (params.yaml)}, nnodes=$NNODES, node_rank=$NODE_RANK"
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

echo "Full continued pre-training complete."
