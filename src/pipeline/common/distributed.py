"""Distributed-training helpers shared by every trainer (raw DDP, torchrun)."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    local_rank: int
    world_size: int
    rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    """Read torchrun's environment and, for more than one process, start NCCL."""
    info = DistInfo(
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
    )
    if info.world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(info.local_rank)
    return info


def pick_device(info: DistInfo, device: str | None, max_gpu_memory_fraction: float) -> torch.device:
    """`device` is a seam for tests; a real run uses this rank's CUDA device if there is one."""
    dev = torch.device(
        device or (f"cuda:{info.local_rank}" if torch.cuda.is_available() else "cpu")
    )
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(max_gpu_memory_fraction, dev)
    return dev


def assert_ranks_agree_on_resume(local_step: int, device) -> None:
    """A checkpoint only exists on the node that saved it. If ranks resumed from
    different steps their optimizer states would silently diverge, so fail every
    rank together (a one-sided error would leave the others hanging)."""
    reference = torch.tensor([local_step], device=device)
    dist.broadcast(reference, src=0)
    disagree = torch.tensor([int(local_step != int(reference.item()))], device=device)
    dist.all_reduce(disagree, op=dist.ReduceOp.MAX)
    if disagree.item():
        raise RuntimeError(
            f"ranks disagree on the resume checkpoint (this rank: step {local_step}, rank 0: "
            f"step {int(reference.item())}). Copy the checkpoint directory to every node, or "
            f"pass --no-resume to start fresh."
        )
