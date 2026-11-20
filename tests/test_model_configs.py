"""Every model's config merges over params.yaml into a valid Params, and says what it should."""

from pathlib import Path

import pytest

from pipeline.common.config import deep_merge, load_params, load_raw
from pipeline.models.registry import MODELS, TRAINERS, trainer_module

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {  # model -> (trainer, base_model, cluster layer or None)
    "xlsr": ("xlsr", "facebook/wav2vec2-xls-r-300m", None),
    "wav2vec2": ("xlsr", "facebook/wav2vec2-large-lv60", None),
    "wavlm": ("wavlm", "microsoft/wavlm-large", 18),
    "hubert": ("hubert", "facebook/hubert-large-ll60k", 18),
    "mhubert": ("hubert", "utter-project/mHuBERT-147", 9),
}


def _load(model):
    return load_params(str(REPO_ROOT / "params.yaml"), model=model)


def test_every_registered_model_has_a_config_and_a_known_trainer():
    assert set(MODELS) == set(EXPECTED)
    for name in MODELS:
        assert (REPO_ROOT / "configs" / "models" / f"{name}.yaml").exists(), name
        assert _load(name).model.trainer in TRAINERS


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_model_config_selects_the_right_trainer_base_model_and_layer(model):
    trainer, base, layer = EXPECTED[model]
    p = _load(model)
    assert (p.model.trainer, p.pretrain.base_model) == (trainer, base)
    assert (p.cluster.layer if p.cluster else None) == layer
    assert MODELS[model].endswith(f"{trainer}.trainer")
    assert trainer_module(trainer).train  # the trainer module imports and exposes train()


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_models_share_the_training_recipe_of_params_yaml(model):
    """A fair comparison needs the same batch, schedule, masking and LR for every model."""
    base = load_params(str(REPO_ROOT / "params.yaml")).pretrain
    p = _load(model).pretrain
    for name in (
        "precision", "max_updates", "warmup_updates", "peak_lr", "lr_schedule", "hold_ratio",
        "adam_betas", "adam_eps", "weight_decay", "max_grad_norm", "target_batch_seconds",
        "per_device_max_seconds", "mask_time_prob", "mask_time_length", "save_every_updates",
        "eval_every_updates", "keep_last_n_checkpoints", "milestone_checkpoints",
    ):  # fmt: skip
        assert getattr(p, name) == getattr(base, name), name


def test_outputs_of_different_models_never_collide():
    seen = {}
    for model in EXPECTED:
        p = _load(model)
        raw = load_raw(str(REPO_ROOT / "params.yaml"), model)
        paths = {
            "model_dir": p.pretrain.output_dir,
            "reports": p.model.reports_dir,
            "selected": raw["checkpoint_selection"].get("output_dir"),
        }
        if p.cluster:
            paths |= {"kmeans": p.cluster.kmeans_output, "labels": p.cluster.labels_output}
        for kind, path in paths.items():
            key = (kind, path)
            assert key not in seen, f"{model} and {seen[key]} both write {kind} -> {path}"
            seen[key] = model


def test_xlsr_config_changes_nothing():
    """params.yaml is the XLS-R run: its overlay must not alter any value."""
    assert load_raw(str(REPO_ROOT / "params.yaml"), "xlsr") == deep_merge(
        load_raw(str(REPO_ROOT / "params.yaml")), {"model": {"trainer": "xlsr", "tag": "xlsr300m"}}
    )


def test_train_cli_parses_every_registered_model_and_rejects_unknown_ones():
    import pytest

    from pipeline.train import main  # the CLI

    with pytest.raises(SystemExit):
        main(["--model", "not-a-model"])
    with pytest.raises(SystemExit) as exc:  # --help parses the flags without training anything
        main(["--help"])
    assert exc.value.code == 0


# ── ports of the checks the model branches kept in tests/test_schema.py ──


def test_model_specific_training_settings():
    """What each branch's schema tests pinned, now per model: WavLM mixes utterances (its
    method, not tuning); the masked-prediction models run layerdrop 0.1 and a head at the
    encoder's LR; the wav2vec2 trainers keep the checkpoint's own layerdrop."""
    for model, mix in (
        ("wavlm", 0.2),
        ("hubert", 0.0),
        ("mhubert", 0.0),
        ("xlsr", 0.0),
        ("wav2vec2", 0.0),
    ):
        assert _load(model).pretrain.utterance_mix_prob == mix, model
    for model in ("wavlm", "hubert", "mhubert"):
        p = _load(model)
        assert (p.pretrain.layerdrop, p.pretrain.head_lr_mult) == (0.1, 1.0), model
        assert p.pretrain.labels_path == p.cluster.labels_output, model
        assert p.cluster.num_clusters == 500, model
    # the wav2vec2-family models state their checkpoint's own value (XLS-R 0.1, lv60 0.0), so
    # every model's layerdrop is visible in its config
    assert _load("xlsr").pretrain.layerdrop == 0.1
    assert _load("wav2vec2").pretrain.layerdrop == 0.0


def test_gpu_memory_cap_is_the_shared_one_and_is_bounded():
    import pytest
    from pydantic import ValidationError

    from pipeline.schema import PretrainConfig

    for model in EXPECTED:
        assert _load(model).pretrain.max_gpu_memory_fraction == 0.8, model
    base = _load("xlsr").pretrain.model_dump()
    for bad in (0.0, 1.5):
        with pytest.raises(ValidationError):
            PretrainConfig(**{**base, "max_gpu_memory_fraction": bad})


def test_a_trainer_without_the_settings_of_its_objective_is_refused():
    """The Gumbel settings are optional in the schema, but the xlsr trainer needs them; the
    masked-prediction trainers need a cluster section and a labels path instead."""
    import pytest
    from pydantic import ValidationError

    from pipeline.schema import Params

    raw = load_raw(str(REPO_ROOT / "params.yaml"))
    for name in ("max_gumbel_temperature", "diversity_loss_weight"):
        broken = deep_merge(raw, {"pretrain": {name: None}})
        with pytest.raises(ValidationError, match="xlsr trainer needs"):
            Params(**broken)
    with pytest.raises(ValidationError, match="needs a `cluster` section"):
        Params(**deep_merge(raw, {"model": {"trainer": "wavlm"}}))
    wavlm = load_raw(str(REPO_ROOT / "params.yaml"), "wavlm")
    with pytest.raises(ValidationError, match="labels_path"):
        Params(**deep_merge(wavlm, {"pretrain": {"labels_path": None}}))


def test_every_model_names_its_leaderboard_run_and_family():
    """model.run_name is both the MLflow run name and the model's name in the SLSB leaderboard."""
    names = {m: (_load(m).model.run_name, _load(m).model.family) for m in EXPECTED}
    assert names == {
        "xlsr": ("xlsr300m-copt200h-norm", "xlsr"),
        "wav2vec2": ("wav2vec2-large-lv60-copt200h", "wav2vec2"),
        "wavlm": ("wavlm-large-copt200h", "wavlm"),
        "hubert": ("hubert-large-copt200h", "hubert"),
        "mhubert": ("mhubert147-copt200h", "mhubert"),
    }
