"""The model registry: MODEL_REGISTRY and get_model_spec, assembled from the per-model
specs (one file per model in this package). ModelSpec and the feature-encoder helpers are
re-exported from pipeline.models.spec so `from pipeline.models.registry import ...` keeps
working for every name the pipeline used before the split.
"""

from pipeline.models import hubert, mhubert147, wav2vec2, wavlm, xlsr
from pipeline.models.spec import (
    ModelFamily,
    ModelSpec,
    describe_feature_encoder,
    feature_encoder_status,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelFamily",
    "ModelSpec",
    "describe_feature_encoder",
    "feature_encoder_status",
    "get_model_spec",
]

MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.key: spec for spec in (mhubert147.SPEC, hubert.SPEC, wavlm.SPEC, xlsr.SPEC, wav2vec2.SPEC)
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {key!r}; expected one of {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
