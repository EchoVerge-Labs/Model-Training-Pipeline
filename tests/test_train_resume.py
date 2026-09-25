"""Checkpoint save/resume, layerdrop, and the monitoring helpers, on a tiny random WavLM."""

import pytest
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

from pipeline.callbacks import MaskedAccuracyStallDetector
from pipeline.checkpoints import STATE_FILE, latest_checkpoint, list_checkpoints
from pipeline.train import prediction_perplexity, save_checkpoint
from pipeline.wavlm_model import WavLMForMaskedPrediction

CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "conv_dim": (32, 32),
    "conv_kernel": (10, 3),
    "conv_stride": (5, 2),
}


def _model(clusters=8):
    return WavLMForMaskedPrediction.from_config(WavLMConfig(**CONFIG), num_clusters=clusters)


def test_a_saved_checkpoint_resumes_backbone_head_optimizer_and_step(tmp_path):
    torch.manual_seed(0)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 8000)
    model(x).sum().backward()
    optimizer.step()  # gives Adam real state to round-trip

    ckpt = tmp_path / "checkpoint-7"
    save_checkpoint(model, Wav2Vec2FeatureExtractor(), optimizer, 7, ckpt)

    assert not (tmp_path / "checkpoint-7.tmp").exists()
    assert latest_checkpoint(tmp_path) == (7, ckpt)

    fresh = _model()
    fresh.load_checkpoint(ckpt)
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True):
        assert torch.equal(a, b)  # backbone AND head restored

    state = torch.load(ckpt / STATE_FILE, map_location="cpu")
    assert state["global_step"] == 7
    opt2 = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    opt2.load_state_dict(state["optimizer"])
    assert len(opt2.state) == len(optimizer.state) > 0


def test_the_saved_backbone_still_loads_as_a_plain_wavlm_model(tmp_path):
    """select_checkpoint / slsb load these directories as upstreams."""
    model = _model()
    ckpt = tmp_path / "checkpoint-1"
    save_checkpoint(
        model, Wav2Vec2FeatureExtractor(), torch.optim.AdamW(model.parameters()), 1, ckpt
    )
    WavLMModel.from_pretrained(ckpt)


def test_incomplete_and_tmp_checkpoints_are_never_resumed(tmp_path):
    model = _model()
    opt = torch.optim.AdamW(model.parameters())
    save_checkpoint(model, Wav2Vec2FeatureExtractor(), opt, 5, tmp_path / "checkpoint-5")

    (tmp_path / "checkpoint-9.tmp").mkdir()  # crashed mid-save
    half = tmp_path / "checkpoint-8"  # backbone but no head / optimizer state
    half.mkdir()
    model.wavlm.save_pretrained(half)

    assert [step for step, _ in list_checkpoints(tmp_path)] == [5]
    assert latest_checkpoint(tmp_path)[0] == 5
    assert latest_checkpoint(tmp_path / "missing") is None


def test_layerdrop_is_overridden_to_zero(tmp_path):
    """The wavlm-large checkpoint ships layerdrop=0.1, which leaves parameters
    unused in a step and breaks DDP(find_unused_parameters=False)."""
    WavLMModel(WavLMConfig(layerdrop=0.1, **CONFIG)).save_pretrained(tmp_path)
    assert WavLMModel.from_pretrained(tmp_path).config.layerdrop == 0.1
    model = WavLMForMaskedPrediction(str(tmp_path), num_clusters=4, layerdrop=0.0)
    assert model.config.layerdrop == 0.0


def test_a_model_that_cannot_apply_masks_is_rejected():
    with pytest.raises(ValueError, match="masked_spec_embed"):
        WavLMForMaskedPrediction.from_config(
            WavLMConfig(apply_spec_augment=False, **CONFIG), num_clusters=4
        )


def test_prediction_perplexity_spans_collapse_to_uniform():
    k = 10
    collapsed = torch.zeros(k)
    collapsed[3] = 100.0
    assert prediction_perplexity(collapsed, 100) == pytest.approx(1.0, abs=1e-3)
    assert prediction_perplexity(torch.full((k,), 10.0), 100) == pytest.approx(k, rel=1e-3)


def test_stall_detector_kills_after_n_low_checks_and_recovers_when_learning():
    d = MaskedAccuracyStallDetector(num_clusters=500, max_consecutive_before_kill=3)
    with pytest.warns(UserWarning):
        d.check(0.001, 500)
    with pytest.warns(UserWarning):
        d.check(0.001, 1000)
    assert d.consecutive_low == 2
    assert d.check(0.30, 1500, perplexity=40.0)  # learning resets the count
    assert d.consecutive_low == 0

    with pytest.warns(UserWarning):
        d.check(0.30, 2000, perplexity=1.2)  # accuracy fine, but collapsed
    with pytest.warns(UserWarning):
        d.check(0.30, 2500, perplexity=1.2)
    assert not d.check(0.30, 3000, perplexity=1.2)
