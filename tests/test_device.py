"""pipeline.common.device: GPU only, always an indexed CUDA device, never a CPU fallback."""

import pytest
import torch

from pipeline.common.device import get_device


def test_default_is_an_indexed_cuda_device():
    device = get_device()
    assert device.type == "cuda" and device.index is not None


def test_a_plain_cuda_string_is_given_an_index():
    assert get_device(device="cuda").index is not None


def test_local_rank_picks_that_gpu():
    assert get_device(local_rank=0) == torch.device("cuda", 0)


def test_cpu_is_rejected():
    with pytest.raises(ValueError, match="only runs on a CUDA GPU"):
        get_device(device="cpu")


def test_no_gpu_is_an_error_not_a_silent_cpu_run(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        get_device()
