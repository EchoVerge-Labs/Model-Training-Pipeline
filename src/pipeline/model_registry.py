"""Registry of the five models this pipeline continues pre-training.

One `ModelSpec` per model key holds everything that differs between models, so the
shared code looks facts up here instead of branching on strings: the objective family,
the transformers classes, where the encoder lives inside the wrapper, whether cluster
labels are needed, and the encoder width (PE-CoPT adapter dimensions depend on it).

This module imports nothing heavy: transformers classes are named by string and only
imported when `config_class` / `model_class` is read, so data-prep code that just needs
the family (via pipeline.schema) doesn't pull in torch.
"""

import importlib
from dataclasses import dataclass
from typing import Literal

ModelFamily = Literal["masked_prediction", "contrastive"]


@dataclass(frozen=True)
class ModelSpec:
    key: str  # also the suffix of the per-model files: params-<key>.yaml, reports/*-<key>.json
    family: ModelFamily
    base_model: str  # the checkpoint continued pre-training starts from
    hf_model_type: str  # the checkpoint's config.model_type
    config_class_name: str  # transformers class names, resolved lazily
    model_class_name: str  # the bare encoder (what select_checkpoint / SLSB load)
    backbone_attr: str  # where the encoder sits inside the training wrapper
    hidden_size: int
    params_file: str
    trainer_module: str  # where this model's mask helpers live, until the trainers merge
    mask_function: str  # that module's masking function
    run_name_prefix: str  # MLflow run names are "<prefix>-si-ta-<batch seconds>s"

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


def _masked(key, base_model, hf_type, config_cls, model_cls, hidden_size, params, run_prefix):
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
        trainer_module="pipeline.train_masked_prediction",
        mask_function="compute_mask",
        run_name_prefix=run_prefix,
    )


def _contrastive(key, base_model, params, trainer, run_prefix):
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
        trainer_module=trainer,
        mask_function="compute_mask_and_negatives",
        run_name_prefix=run_prefix,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        _masked(
            "mhubert147",
            "utter-project/mHuBERT-147",
            "hubert",
            "HubertConfig",
            "HubertModel",
            768,
            "params.yaml",
            "mhubert147",
        ),
        _masked(
            "hubert",
            "facebook/hubert-large-ll60k",
            "hubert",
            "HubertConfig",
            "HubertModel",
            1024,
            "params-hubert.yaml",
            "hubert-large",
        ),
        _masked(
            "wavlm",
            "microsoft/wavlm-large",
            "wavlm",
            "WavLMConfig",
            "WavLMModel",
            1024,
            "params-wavlm.yaml",
            "wavlm-large",
        ),
        _contrastive(
            "xlsr",
            "facebook/wav2vec2-xls-r-300m",
            "params-xlsr.yaml",
            "pipeline.train_contrastive",
            "xlsr300m",
        ),
        _contrastive(
            "wav2vec2",
            "facebook/wav2vec2-large-lv60",
            "params-wav2vec2.yaml",
            "pipeline.train_contrastive",
            "wav2vec2-mono",
        ),
    )
}


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


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {key!r}; expected one of {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
