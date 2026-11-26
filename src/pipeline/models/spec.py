"""The ModelSpec type and the two builders the per-model files use.

A `ModelSpec` holds everything that differs between models, so the shared code looks facts
up here instead of branching on strings: the objective family, the transformers classes,
where the encoder lives inside the wrapper, whether cluster labels are needed, and the
encoder width (PE-CoPT adapter dimensions depend on it).

This module imports nothing heavy: transformers classes are named by string and only
imported when `config_class` / `model_class` is read, so data-prep code that just needs the
family (via pipeline.common.schema) doesn't pull in torch.
"""

import importlib
from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal["masked_prediction", "contrastive"]


@dataclass(frozen=True)
class ModelSpec:
    key: str  # MODEL=<key>; also names params-<key>.yaml and reports/phase0-<key>.json
    family: ModelFamily
    base_model: str  # the checkpoint continued pre-training starts from
    hf_model_type: str  # the checkpoint's config.model_type
    config_class_name: str  # transformers class names, resolved lazily
    model_class_name: str  # the bare encoder (what select_checkpoint / SLSB load)
    backbone_attr: str  # where the encoder sits inside the training wrapper
    hidden_size: int
    params_file: str
    trainer_module: str  # the family trainer this model runs through
    mask_function: str  # that module's masking function
    # The model's name: the first slot of every run name (wavlm-large-copt-si-ta-200h, see
    # pipeline.common.naming) and of its frozen baseline (wavlm-large-frozen), and its family on
    # the SLSB leaderboard (SLSB-benchmark.mlflow).
    slsb_name: str
    slsb_family: str

    @property
    def needs_labels(self) -> bool:
        """Masked prediction trains against offline k-means cluster ids."""
        return self.family == "masked_prediction"

    @property
    def config_class(self) -> type:
        return getattr(importlib.import_module("transformers"), self.config_class_name)

    @property
    def model_class(self) -> type:
        return getattr(importlib.import_module("transformers"), self.model_class_name)


def masked_spec(
    key, base_model, hf_type, config_cls, model_cls, hidden_size, params, name, slsb_family
):
    return ModelSpec(
        key=key,
        family="masked_prediction",
        base_model=base_model,
        hf_model_type=hf_type,
        config_class_name=config_cls,
        model_class_name=model_cls,
        backbone_attr="backbone",  # MaskedPredictionModel.backbone
        hidden_size=hidden_size,
        params_file=params,
        trainer_module="pipeline.training.masked_prediction",
        mask_function="compute_mask",
        slsb_name=name,
        slsb_family=slsb_family,
    )


def contrastive_spec(key, base_model, params, name, slsb_family):
    return ModelSpec(
        key=key,
        family="contrastive",
        base_model=base_model,
        hf_model_type="wav2vec2",
        config_class_name="Wav2Vec2Config",
        model_class_name="Wav2Vec2Model",
        backbone_attr="wav2vec2",  # Wav2Vec2ForPreTraining.wav2vec2
        hidden_size=1024,
        params_file=params,
        trainer_module="pipeline.training.contrastive",
        mask_function="compute_mask_and_negatives",
        slsb_name=name,
        slsb_family=slsb_family,
    )


def feature_encoder_status(model, spec: ModelSpec) -> tuple[int, int]:
    """(CNN parameters, CNN parameters that train), read from the model itself -- so it
    reports what will really happen, not just what the config asked for."""
    cnn = list(getattr(model, spec.backbone_attr).feature_extractor.parameters())
    return sum(p.numel() for p in cnn), sum(p.numel() for p in cnn if p.requires_grad)


def describe_feature_encoder(model, spec: ModelSpec) -> str:
    total, trainable = feature_encoder_status(model, spec)
    if trainable == total:
        state = "TRAINABLE"
    elif trainable == 0:
        state = "frozen"
    else:
        state = f"PARTLY trainable ({trainable / 1e6:.1f}M)"
    return f"CNN feature encoder ({total / 1e6:.1f}M params): {state}"
