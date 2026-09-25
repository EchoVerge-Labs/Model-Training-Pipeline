"""Validate that params.yaml matches the Pydantic schema."""

from pipeline.schema import Params


def test_params_loads():
    p = Params.from_yaml("params.yaml")
    assert p.pretrain.base_model == "microsoft/wavlm-large"
    assert p.pretrain.precision == "bf16"


def test_cluster_config_loads_and_matches_labels_path():
    p = Params.from_yaml("params.yaml")
    assert p.cluster.num_clusters > 0
    assert p.pretrain.labels_path == p.cluster.labels_output


def test_language_mix_sums_to_one():
    p = Params.from_yaml("params.yaml")
    total = sum(p.select.language_mix.values())
    assert 0.99 <= total <= 1.01, f"language_mix sums to {total}"


def test_milestone_checkpoints_within_max_updates():
    p = Params.from_yaml("params.yaml")
    for m in p.pretrain.milestone_checkpoints:
        assert m <= p.pretrain.max_updates, f"Milestone {m} > max_updates {p.pretrain.max_updates}"


def test_warmup_before_max():
    p = Params.from_yaml("params.yaml")
    assert p.pretrain.warmup_updates < p.pretrain.max_updates


def _shard_cfg(**overrides):
    from pipeline.schema import ShardConfig

    base = {
        "shard_size": 1000,
        "min_cut_seconds": 5.0,
        "target_cut_seconds": 15.0,
        "max_cut_seconds": 30.0,
        "output_dir": "data/shars",
    }
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


# Settings shared with the wav2vec2 XLS-R run on `main`; a fair comparison needs
# them identical. Values copied from main's params.yaml.
MAIN_SHARED = {
    "precision": "bf16",
    "freeze_feature_encoder": True,
    "max_updates": 13500,
    "warmup_updates": 1000,
    "peak_lr": 5.0e-5,
    "lr_schedule": "tri_stage",
    "hold_ratio": 0.4,
    "optimizer": "adamw",
    "adam_betas": [0.9, 0.98],
    "adam_eps": 1.0e-6,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "target_batch_seconds": 1600,
    "per_device_max_seconds": 200,
    "num_workers": 4,
    "mask_time_prob": 0.65,
    "mask_time_length": 10,
    "save_every_updates": 1500,
    "eval_every_updates": 500,
    "keep_last_n_checkpoints": 5,
    "milestone_checkpoints": [4500, 9000, 13500],
    "head_lr_mult": 1.0,
    "layerdrop": 0.1,  # XLS-R stock config, which main trains with unchanged
}


def test_training_settings_match_the_wav2vec2_run_on_main():
    p = Params.from_yaml("params.yaml").pretrain
    for name, expected in MAIN_SHARED.items():
        assert getattr(p, name) == expected, f"{name} differs from main"


def test_wavlm_branch_specifics():
    p = Params.from_yaml("params.yaml")
    assert p.pretrain.base_model == "microsoft/wavlm-large"
    assert p.pretrain.utterance_mix_prob == 0.2  # WavLM's mixing; method, not tuning
    assert "wavlm" in p.pretrain.output_dir
    assert "wavlm" in p.cluster.kmeans_output and "wavlm" in p.cluster.labels_output
