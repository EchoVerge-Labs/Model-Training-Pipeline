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


def _shard_cfg(**overrides):
    from pipeline.schema import ShardConfig
    base = dict(shard_size=1000, min_cut_seconds=5.0, target_cut_seconds=15.0,
                max_cut_seconds=30.0, output_dir="data/shars")
    base.update(overrides)
    return ShardConfig(**base)


def test_shard_cut_limits_in_params_are_consistent():
    p = Params.from_yaml("params.yaml")
    assert p.shard.max_cut_seconds == 30.0
    assert p.shard.max_cut_seconds >= p.shard.target_cut_seconds + p.shard.min_cut_seconds


def test_max_cut_seconds_must_keep_split_windows_above_the_minimum():
    import pytest
    with pytest.raises(ValueError, match="2 \\* min_cut_seconds"):
        _shard_cfg(min_cut_seconds=20.0, target_cut_seconds=25.0, max_cut_seconds=30.0)


def test_max_cut_seconds_must_leave_room_for_merged_short_clips():
    import pytest
    with pytest.raises(ValueError, match="target_cut_seconds \\+"):
        _shard_cfg(max_cut_seconds=18.0)
