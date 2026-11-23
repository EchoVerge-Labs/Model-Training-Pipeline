"""The compute device. This pipeline runs on a CUDA GPU only; there is no CPU path."""

import torch


def get_device(local_rank: int | None = None, device: str | torch.device | None = None):
    """The indexed CUDA device to run on: `device` if given (e.g. "cuda"), else
    cuda:<local_rank> for distributed runs, else the current GPU.

    Raises instead of falling back to the CPU: a silent CPU run would take weeks
    and produce nothing useful, so a missing GPU should stop the job immediately.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available; this pipeline only runs on a GPU.")
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type != "cuda":
        raise ValueError(f"This pipeline only runs on a CUDA GPU, got device={resolved}")
    if resolved.index is None:
        index = local_rank if local_rank is not None else torch.cuda.current_device()
        resolved = torch.device("cuda", index)
    return resolved
