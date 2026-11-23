"""CNN feature encoder freeze/unfreeze: one switch (pretrain.freeze_feature_encoder), all models.

Full CoPT trains the CNN too; true freezes it. The status line is read from the parameters
themselves, and the end-to-end tests check the CNN weights really do (not) change.
"""

import pytest
import torch
from transformers import HubertModel, Wav2Vec2ForPreTraining

from pipeline.models.registry import (
    MODEL_REGISTRY,
    describe_feature_encoder,
    feature_encoder_status,
)
from pipeline.training.masked_prediction import train as train_masked
from tests.test_resume import _make_base, _run
from tests.test_resume import _params as contrastive_params
from tests.test_smoke_train import _tiny_model as tiny_masked
from tests.test_smoke_train_xlsr import _tiny_model as tiny_contrastive
from tests.test_train_e2e_hubert import _params as masked_params
from tests.test_train_e2e_hubert import _setup as masked_setup


@pytest.mark.parametrize(("key", "build"), [("wavlm", tiny_masked), ("xlsr", tiny_contrastive)])
def test_status_line_reports_the_real_state_for_both_families(key, build):
    spec, model = MODEL_REGISTRY[key], build()
    total, trainable = feature_encoder_status(model, spec)
    assert trainable == total > 0
    assert "TRAINABLE" in describe_feature_encoder(model, spec)

    model.freeze_feature_encoder()
    assert feature_encoder_status(model, spec) == (total, 0)
    assert describe_feature_encoder(model, spec).endswith("frozen")

    next(getattr(model, spec.backbone_attr).feature_extractor.parameters()).requires_grad_(True)
    assert "PARTLY trainable" in describe_feature_encoder(model, spec)


def _cnn_weights(feature_extractor):
    return [p.detach().cpu().clone() for p in feature_extractor.parameters()]


def _changed(before, after):
    return any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


@pytest.mark.parametrize("freeze", [True, False])
def test_contrastive_trainer_trains_the_cnn_only_when_unfrozen(tmp_path, capsys, freeze):
    base = _make_base(tmp_path)
    before = _cnn_weights(Wav2Vec2ForPreTraining.from_pretrained(base).wav2vec2.feature_extractor)
    params = contrastive_params(tmp_path, base, max_updates=3, freeze_feature_encoder=freeze)
    _run(params, tmp_path)
    after = _cnn_weights(
        Wav2Vec2ForPreTraining.from_pretrained(tmp_path / "out").wav2vec2.feature_extractor
    )
    assert _changed(before, after) is (not freeze)
    assert ("frozen" if freeze else "TRAINABLE") in capsys.readouterr().out


@pytest.mark.parametrize("freeze", [True, False])
def test_masked_trainer_trains_the_cnn_only_when_unfrozen(tmp_path, monkeypatch, capsys, freeze):
    monkeypatch.chdir(tmp_path)
    base, shars, labels = masked_setup(tmp_path)
    before = _cnn_weights(HubertModel.from_pretrained(base).feature_extractor)
    params = masked_params(tmp_path, base, shars, labels, max_updates=4)
    params.pretrain.freeze_feature_encoder = freeze
    train_masked(params)
    after = _cnn_weights(HubertModel.from_pretrained(tmp_path / "out").feature_extractor)
    assert _changed(before, after) is (not freeze)
    assert ("frozen" if freeze else "TRAINABLE") in capsys.readouterr().out
