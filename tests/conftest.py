"""Skip the GPU-only tests when there is no GPU, instead of failing them.

This pipeline has no CPU path (see pipeline.common.device.get_device), so most of
the training tests load a tiny model onto CUDA and cannot run anywhere else. On a
machine with a GPU nothing here applies: every test runs, and every failure is a
real failure. On a machine without one -- a CI runner -- the tests that need a GPU
are reported as skipped, so the suite still checks everything that is pure logic
(config schema, run naming, report rendering, Drive index, selection, ...).

The skip is keyed on the specific errors a missing GPU raises, not on a marker, so
a GPU test added later is handled without anyone having to remember to tag it.
"""

import pytest
import torch

# Only these: our own refusal to run on the CPU, and torch's two ways of saying
# the same thing. Anything else is a genuine failure and is left alone.
_MISSING_GPU = (
    "No CUDA GPU is available",  # pipeline.common.device.get_device
    "Found no NVIDIA driver",  # torch.cuda init, no driver at all (CI runners)
    "No CUDA GPUs are available",  # torch.cuda init, driver but no visible device
    "Torch not compiled with CUDA",  # CPU-only torch build
    "Distributed package doesn't have NCCL",  # DDP tests, no GPU backend
)

_REASON = "needs a CUDA GPU; this pipeline has no CPU path"


def _is_missing_gpu(exc: BaseException) -> bool:
    """True if `exc`, or anything it was raised from, is a missing-GPU error."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(marker in str(exc) for marker in _MISSING_GPU):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except BaseException as exc:
        if not torch.cuda.is_available() and _is_missing_gpu(exc):
            pytest.skip(_REASON)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except BaseException as exc:
        if not torch.cuda.is_available() and _is_missing_gpu(exc):
            pytest.skip(_REASON)
        raise
