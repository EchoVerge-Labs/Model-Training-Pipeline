#!/usr/bin/env bash
# NCCL environment for 2x DGX Spark (GB10 Grace-Blackwell)
# Verified values from community cluster guides

export NCCL_SOCKET_IFNAME=enp1s0f0np0    # ConnectX-7 interface — VERIFY on your setup
export NCCL_IB_HCA=mlx5_0                # InfiniBand HCA
export NCCL_IB_GID_INDEX=3               # Required for RoCEv2 on GB10
export NCCL_NET_GDR_LEVEL=0              # GPU Direct RDMA OFF on GB10
export NCCL_DMABUF_ENABLE=0              # DMA-BUF OFF on GB10
export NCCL_ALGO=Ring                    # Optimal for 2 nodes
export NCCL_DEBUG=WARN                   # Change to INFO for troubleshooting
export NCCL_TIMEOUT=1800                 # 30 min timeout

echo "NCCL environment loaded for GB10 2-node setup"
echo "  Interface: $NCCL_SOCKET_IFNAME"
echo "  HCA: $NCCL_IB_HCA"
echo "  Algorithm: $NCCL_ALGO"
