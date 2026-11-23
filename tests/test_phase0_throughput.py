"""scripts/phase0_throughput.py: one train step per objective family, on tiny models on the GPU."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import HubertConfig, Wav2Vec2Config, Wav2Vec2ForPreTraining

from pipeline.masked_prediction_model import MaskedPredictionModel
from pipeline.model_registry import MODEL_REGISTRY
from pipeline.schema import Params

ROOT = Path(__file__).resolve().parent.parent
TINY = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "conv_dim": (16, 16),
    "conv_stride": (5, 2),
    "conv_kernel": (10, 3),
    "num_feat_extract_layers": 2,
    "num_conv_pos_embeddings": 16,
    "num_conv_pos_embedding_groups": 2,
}


@pytest.fixture(scope="module")
def phase0():
    spec = importlib.util.spec_from_file_location("phase0", ROOT / "scripts/phase0_throughput.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase0"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key", ["mhubert147", "wavlm", "xlsr", "wav2vec2"])
def test_train_step_runs_for_both_families(phase0, key):
    import importlib

    spec = MODEL_REGISTRY[key]
    params = Params.from_yaml(str(ROOT / spec.params_file))
    trainer = importlib.import_module(spec.trainer_module)
    device = torch.device("cuda")
    if params.model_family == "masked_prediction":
        from transformers import WavLMConfig

        config = (WavLMConfig if key == "wavlm" else HubertConfig)(**TINY)
        model = MaskedPredictionModel.from_config(config, params.cluster.num_clusters)
    else:
        model = Wav2Vec2ForPreTraining(
            Wav2Vec2Config(
                **TINY,
                do_stable_layer_norm=True,
                feat_extract_norm="layer",
                num_codevectors_per_group=8,
                num_codevector_groups=2,
                codevector_dim=16,
                proj_codevector_dim=16,
                mask_time_prob=0.5,
            )
        )
    model.freeze_feature_encoder()
    model.to(device)
    step = phase0.make_train_step(key, model, params, trainer, 2, 8000, device, torch.bfloat16)
    model.train()
    assert torch.isfinite(step())
