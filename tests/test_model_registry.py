"""model_registry: one spec per model, consistent with the params files and the code."""

import importlib
import json
from pathlib import Path

import pytest

from pipeline.common.schema import Params
from pipeline.models.registry import MODEL_REGISTRY, get_model_spec
from pipeline.training.masked_model import BACKBONES

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_KEYS = {"mhubert147", "hubert", "wavlm", "xlsr", "wav2vec2"}


def test_registry_has_the_five_models_keyed_by_their_own_key():
    assert set(MODEL_REGISTRY) == EXPECTED_KEYS
    assert all(key == spec.key for key, spec in MODEL_REGISTRY.items())


def test_unknown_model_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown model 'bert'"):
        get_model_spec("bert")


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_spec_matches_its_params_file(key):
    spec = MODEL_REGISTRY[key]
    params = Params.from_yaml(str(ROOT / spec.params_file))
    assert params.model_key == key
    assert params.model_family == spec.family
    assert params.pretrain.base_model == spec.base_model
    assert spec.needs_labels == (params.cluster is not None)


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_trainer_module_defines_the_mask_function(key):
    spec = MODEL_REGISTRY[key]
    assert callable(getattr(importlib.import_module(spec.trainer_module), spec.mask_function))


@pytest.mark.parametrize("key", ["mhubert147", "hubert", "wavlm"])
def test_masked_prediction_specs_agree_with_the_model_wrapper(key):
    spec = MODEL_REGISTRY[key]
    config_cls, model_cls = BACKBONES[spec.hf_model_type]
    assert (spec.config_class, spec.model_class) == (config_cls, model_cls)
    assert spec.backbone_attr == "backbone"


def test_contrastive_specs_point_at_the_wav2vec2_encoder():
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    for key in ("xlsr", "wav2vec2"):
        spec = MODEL_REGISTRY[key]
        assert (spec.config_class, spec.model_class) == (Wav2Vec2Config, Wav2Vec2Model)
        assert not spec.needs_labels


def test_hidden_size_matches_the_local_base_model_config_when_present():
    config = ROOT / "models/mhubert147-base/config.json"
    if not config.exists():
        pytest.skip("no local mHuBERT-147 checkpoint")
    assert json.loads(config.read_text())["hidden_size"] == MODEL_REGISTRY["mhubert147"].hidden_size


def test_backbone_attr_exists_on_the_training_wrappers():
    from transformers import HubertConfig, Wav2Vec2Config, Wav2Vec2ForPreTraining

    from pipeline.training.masked_model import MaskedPredictionModel

    tiny = {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "conv_dim": (16,),
        "conv_stride": (5,),
        "conv_kernel": (10,),
        "num_feat_extract_layers": 1,
        "num_conv_pos_embeddings": 16,
        "num_conv_pos_embedding_groups": 2,
    }
    masked = MaskedPredictionModel.from_config(HubertConfig(**tiny), num_clusters=4)
    contrastive = Wav2Vec2ForPreTraining(
        Wav2Vec2Config(**tiny, codevector_dim=16, proj_codevector_dim=16)
    )
    assert hasattr(masked, MODEL_REGISTRY["hubert"].backbone_attr)
    assert hasattr(contrastive, MODEL_REGISTRY["xlsr"].backbone_attr)
    assert (
        masked.config.hidden_size == tiny["hidden_size"]
    )  # the spec's hidden_size is the encoder's
