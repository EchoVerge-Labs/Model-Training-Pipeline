#!/usr/bin/env bash
set -euo pipefail
# Test NCCL bus bandwidth between two DGX Spark nodes.
# Run BEFORE any multi-node training.
#
# Usage:
#   Node 0: MASTER_ADDR=10.8.100.40 NODE_RANK=0 bash scripts/nccl_bandwidth_test.sh
#   Node 1: MASTER_ADDR=10.8.100.40 NODE_RANK=1 bash scripts/nccl_bandwidth_test.sh

source configs/nccl-env.sh

MASTER_ADDR=${MASTER_ADDR:-10.8.100.40}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-2}

echo "NCCL bandwidth test: node $NODE_RANK of $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Use PyTorch's built-in NCCL test
python3 -c "
import os, torch, torch.distributed as dist, time

os.environ['MASTER_ADDR'] = '$MASTER_ADDR'
os.environ['MASTER_PORT'] = '$MASTER_PORT'
os.environ['WORLD_SIZE'] = '$NNODES'
os.environ['RANK'] = '$NODE_RANK'

dist.init_process_group('nccl')
rank = dist.get_rank()
device = torch.device('cuda:0')
print(f'Rank {rank}: connected, device={torch.cuda.get_device_name()}')

# Warmup
for _ in range(5):
    t = torch.randn(1024, 1024, device=device)
    dist.all_reduce(t)

# Benchmark
sizes_mb = [1, 10, 100, 500]
for size_mb in sizes_mb:
    n_floats = size_mb * 1024 * 1024 // 4
    tensor = torch.randn(n_floats, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Bus bandwidth = data * 2 * (n-1)/n / time  (for all_reduce)
    data_bytes = n_floats * 4
    bus_bw = data_bytes * 2 * (2-1)/2 * iters / elapsed / 1e9
    if rank == 0:
        print(f'  {size_mb:4d} MB: {bus_bw:.1f} GB/s  ({elapsed/iters*1000:.1f} ms/iter)')

dist.destroy_process_group()
if rank == 0:
    print()
    print('Expected for 200 GbE ConnectX-7: ~20-24 GB/s at large sizes')
    print('If you see <10 GB/s, check NCCL_SOCKET_IFNAME and cabling')
"
