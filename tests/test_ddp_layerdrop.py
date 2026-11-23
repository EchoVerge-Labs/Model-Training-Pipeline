"""Contrastive trainer under DDP with LayerDrop: unused-parameter detection must be on.

LayerDrop skips whole transformer layers, so their parameters get no gradient that step; DDP
only tolerates that with find_unused_parameters=True, otherwise the second step fails. XLS-R's
checkpoint ships layerdrop=0.1, so a multi-node run would have crashed with the old hardcoded
False. A one-rank NCCL group still builds DDP's reducer, so this exercises the real train()
DDP path on a single GPU.
"""

import socket

import pytest
import torch.distributed as dist
from transformers import Wav2Vec2ForPreTraining

from pipeline import train_contrastive
from pipeline.train_contrastive import skips_layers
from tests.test_resume import _make_base, _params, _run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def one_rank_group(monkeypatch):
    """train() believes it is rank 0 of 2 and calls dist.init_process_group("nccl"); give it a
    real single-rank NCCL group so DDP's reducer is real, and always tear it down."""
    port = _free_port()
    real_init = dist.init_process_group
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda backend: real_init(
            backend, init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        ),
    )
    for name, value in (("WORLD_SIZE", "2"), ("RANK", "0"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(name, value)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _base_with_layerdrop(tmp_path, layerdrop):
    base = _make_base(tmp_path)
    model = Wav2Vec2ForPreTraining.from_pretrained(base)
    model.config.layerdrop = layerdrop
    model.save_pretrained(base)
    return base


def test_skips_layers_follows_the_models_layerdrop(tmp_path):
    for layerdrop, expected in ((0.0, False), (0.1, True)):
        model = Wav2Vec2ForPreTraining.from_pretrained(_base_with_layerdrop(tmp_path, layerdrop))
        assert skips_layers(model) is expected


def test_train_runs_under_ddp_when_layerdrop_skips_layers(tmp_path, one_rank_group):
    base = _base_with_layerdrop(tmp_path, 0.99)  # nearly every layer skipped every step
    params = _params(tmp_path, base, max_updates=3)
    assert _run(params, tmp_path) == 3


def test_without_unused_parameter_detection_the_same_run_crashes(
    tmp_path, one_rank_group, monkeypatch
):
    """Negative control: this is what the old hardcoded find_unused_parameters=False did."""
    monkeypatch.setattr(train_contrastive, "skips_layers", lambda model: False)
    base = _base_with_layerdrop(tmp_path, 0.99)
    params = _params(tmp_path, base, max_updates=3)
    with pytest.raises(RuntimeError, match="finished reduction in the prior iteration"):
        _run(params, tmp_path)
