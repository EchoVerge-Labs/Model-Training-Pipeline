"""Validate that params-mhubert147.yaml matches the Pydantic schema."""

import pytest

from pipeline.common.schema import Params


def test_params_loads():
    p = Params.from_yaml("params-mhubert147.yaml")
    assert p.pretrain.base_model == "utter-project/mHuBERT-147"
    assert p.pretrain.precision == "bf16"


def test_cluster_config_loads_and_matches_labels_path():
    p = Params.from_yaml("params-mhubert147.yaml")
    assert p.cluster.num_clusters > 0
    assert p.pretrain.labels_path == p.cluster.labels_output


def test_language_mix_sums_to_one():
    p = Params.from_yaml("params-mhubert147.yaml")
    total = sum(p.select.language_mix.values())
    assert 0.99 <= total <= 1.01, f"language_mix sums to {total}"


def test_milestone_checkpoints_within_max_updates():
    p = Params.from_yaml("params-mhubert147.yaml")
    for m in p.pretrain.milestone_checkpoints:
        assert m <= p.pretrain.max_updates, f"Milestone {m} > max_updates {p.pretrain.max_updates}"


def test_warmup_before_max():
    p = Params.from_yaml("params-mhubert147.yaml")
    assert p.pretrain.warmup_updates < p.pretrain.max_updates


def _shard_cfg(**overrides):
    from pipeline.common.schema import ShardConfig

    base = {
        "shard_size": 1000,
        "min_cut_seconds": 5.0,
        "target_cut_seconds": 15.0,
        "max_cut_seconds": 30.0,
        "output_dir": "data/shars/si-ta-200h",
    }
    base.update(overrides)
    return ShardConfig(**base)


def test_shard_cut_limits_in_params_are_consistent():
    p = Params.from_yaml("params-mhubert147.yaml")
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


def test_gpu_memory_cap_is_bounded():
    import pytest
    from pydantic import ValidationError

    from pipeline.common.schema import PretrainConfig

    base = Params.from_yaml("params-mhubert147.yaml").pretrain.model_dump()
    for bad in (0.0, 1.5):
        with pytest.raises(ValidationError):
            PretrainConfig(**{**base, "max_gpu_memory_fraction": bad})


def test_mhubert_targets_use_its_own_layer_count():
    p = Params.from_yaml("params-mhubert147.yaml")
    assert p.cluster.layer == 9  # mHuBERT-147 has 12 layers, not HuBERT-Large's 24
    assert p.cluster.num_clusters == 500
    assert "mhubert" in p.pretrain.output_dir
    assert "mhubert" in p.cluster.kmeans_output and "mhubert" in p.cluster.labels_output


ALL_PARAMS = {
    "params-mhubert147.yaml": ("mhubert147", "masked_prediction"),
    "params-hubert.yaml": ("hubert", "masked_prediction"),
    "params-wavlm.yaml": ("wavlm", "masked_prediction"),
    "params-xlsr.yaml": ("xlsr", "contrastive"),
    "params-wav2vec2.yaml": ("wav2vec2", "contrastive"),
}


def test_every_params_file_loads_with_its_model_family():
    for path, (key, family) in ALL_PARAMS.items():
        p = Params.from_yaml(path)
        assert (p.model_key, p.model_family) == (key, family), path


def test_every_params_file_matches_its_registry_spec():
    """base_model, precision and (masked) label paths per model -- the unique bits the
    now-removed per-model schema tests checked, against the registry as source of truth."""
    from pipeline.models.registry import get_model_spec

    for path, (key, _family) in ALL_PARAMS.items():
        p = Params.from_yaml(path)
        assert p.pretrain.base_model == get_model_spec(key).base_model, path
        assert p.pretrain.precision == "bf16", path
        if p.model_family == "masked_prediction":
            assert p.cluster.num_clusters > 0 and p.pretrain.labels_path == p.cluster.labels_output


def _raw(path="params-mhubert147.yaml"):
    from pipeline.common.schema import load_raw_params

    return load_raw_params(path)


def test_masked_prediction_requires_cluster_and_derives_its_label_paths():
    import pytest
    from pydantic import ValidationError

    raw = _raw()
    del raw["cluster"]
    with pytest.raises(ValidationError, match="cluster"):
        Params(**raw)
    p = Params(**_raw())  # no paths in the file: derived from the model and the data id
    assert p.cluster.kmeans_output == "models/kmeans/mhubert147-L9-k500-si-ta-200h.joblib"
    assert p.pretrain.labels_path == "data/labels/mhubert147-L9-k500-si-ta-200h.jsonl.gz"


def test_contrastive_requires_gumbel_fields():
    import pytest
    from pydantic import ValidationError

    raw = _raw("params-xlsr.yaml")
    del raw["pretrain"]["max_gumbel_temperature"]
    with pytest.raises(ValidationError, match="max_gumbel_temperature"):
        Params(**raw)


def test_unknown_model_key_is_rejected():
    import pytest
    from pydantic import ValidationError

    raw = _raw()
    raw["model_key"] = "bert"
    with pytest.raises(ValidationError):
        Params(**raw)


def _schedule(**over):
    from pipeline.common.schema import PretrainConfig

    base = Params.from_yaml("params-xlsr.yaml").pretrain.model_dump()
    return PretrainConfig(**{**base, **over})


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"milestone_checkpoints": [4500, 20000]}, "beyond max_updates"),
        ({"milestone_checkpoints": [4000]}, "multiple of save_every_updates"),
        ({"warmup_updates": 20000}, "below max_updates"),
        ({"min_gumbel_temperature": 2.0, "max_gumbel_temperature": 1.0}, "must not exceed"),
    ],
)
def test_inconsistent_checkpoint_schedule_is_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        _schedule(**bad)


def test_a_consistent_schedule_is_accepted():
    _schedule(milestone_checkpoints=[1500, 3000], max_updates=3000)


def test_data_config_is_separate_and_merged_into_every_model_file():
    from pipeline.common.schema import DataParams

    data = DataParams.from_yaml("params-data.yaml")
    for path in ALL_PARAMS:
        p = Params.from_yaml(path)
        assert p.select == data.select and p.shard == data.shard


def test_model_files_hold_no_data_sections():
    for path in ALL_PARAMS:
        with open(path) as f:
            text = f.read()
        assert "\nselect:" not in text and "\nshard:" not in text, path
