"""Validate that params.yaml matches the Pydantic schema."""
from pipeline.schema import Params


def test_params_loads():
    p = Params.from_yaml("params.yaml")
    assert p.pretrain.base_model == "facebook/wav2vec2-xls-r-300m"
    assert p.pretrain.precision == "bf16"


def test_language_mix_sums_to_one():
    p = Params.from_yaml("params.yaml")
    total = sum(p.select.language_mix.values())
    assert 0.99 <= total <= 1.01, f"language_mix sums to {total}"


def test_milestone_checkpoints_within_max_updates():
    p = Params.from_yaml("params.yaml")
    for m in p.pretrain.milestone_checkpoints:
        assert m <= p.pretrain.max_updates, \
            f"Milestone {m} > max_updates {p.pretrain.max_updates}"


def test_warmup_before_max():
    p = Params.from_yaml("params.yaml")
    assert p.pretrain.warmup_updates < p.pretrain.max_updates
