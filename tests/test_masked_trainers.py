"""The masked-prediction trainers (wavlm, hubert), on tiny random models, CPU.

Each test runs once per trainer. They are ports of the tests that lived on the
WavLM-Large / HuBERT-Large / mHuBERT-147 branches (smoke, end-to-end train + resume,
checkpoints, layerdrop, monitoring), adapted to the shared checkpoint code in
pipeline.common.checkpoints.
"""

import gzip
import importlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("lhotse")
import soundfile as sf
import torch
from lhotse import CutSet, MonoCut, Recording
from torch import nn
from transformers import (
    HubertConfig,
    HubertModel,
    Wav2Vec2FeatureExtractor,
    WavLMConfig,
    WavLMModel,
)

from pipeline.callbacks import MaskedAccuracyStallDetector
from pipeline.common.checkpoints import TRAINING_STATE_FILE, list_checkpoints
from pipeline.common.masked_prediction import HEAD_FILE, prediction_perplexity
from pipeline.schema import (
    ClusterConfig,
    ModelConfig,
    Params,
    PretrainConfig,
    SelectConfig,
    ShardConfig,
)
from pipeline.shard import shard_cuts

K = 6
CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "conv_dim": (32, 32),
    "conv_kernel": (10, 3),
    "conv_stride": (5, 2),
}


@dataclass(frozen=True)
class Family:
    name: str
    config_cls: type
    encoder_cls: type
    wrapper: str  # class in pipeline.models.<name>.model
    attr: str  # the encoder's attribute on the wrapper
    mix_prob: float

    @property
    def trainer(self):
        return importlib.import_module(f"pipeline.models.{self.name}.trainer")

    @property
    def wrapper_cls(self):
        return getattr(importlib.import_module(f"pipeline.models.{self.name}.model"), self.wrapper)

    def tiny(self, clusters=8):
        return self.wrapper_cls.from_config(self.config_cls(**CONFIG), num_clusters=clusters)


FAMILIES = [
    Family("wavlm", WavLMConfig, WavLMModel, "WavLMForMaskedPrediction", "wavlm", 0.5),
    Family("hubert", HubertConfig, HubertModel, "HubertForMaskedPrediction", "hubert", 0.0),
]


@pytest.fixture(params=FAMILIES, ids=lambda f: f.name)
def fam(request):
    return request.param


# ── smoke: the objective itself ──


def test_compute_mask_returns_a_bool_mask_over_the_conv_output_length(fam):
    model = fam.tiny()
    x = torch.randn(4, 16000)
    attn = torch.ones(4, 16000, dtype=torch.long)
    mask, sub = fam.trainer.compute_mask(model, x, attn, 0.65, 10, torch.device("cpu"))
    seq = int(model._get_feat_extract_output_lengths(x.shape[-1]))
    assert mask.shape == (4, seq) and mask.dtype == torch.bool  # no internal dtype cast
    assert mask.any() and not mask.all()
    assert sub.shape == (4, seq)


def test_three_masked_prediction_steps_run_and_backprop(fam):
    model = fam.tiny(clusters=8)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(3):
        x = torch.randn(2, 16000)
        attn = torch.ones(2, 16000, dtype=torch.long)
        mask, _ = fam.trainer.compute_mask(model, x, attn, 0.65, 10, torch.device("cpu"))
        logits = model(x, attention_mask=attn, mask_time_indices=mask)
        assert logits.shape == (*mask.shape, 8)
        target = torch.randint(0, 8, mask.shape)
        target[~mask] = -100
        loss = loss_fn(logits.transpose(1, 2), target)
        assert loss.requires_grad
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# ── checkpoints ──


def test_a_saved_checkpoint_resumes_backbone_head_optimizer_and_step(fam, tmp_path):
    torch.manual_seed(0)
    model = fam.tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8000)).sum().backward()
    optimizer.step()  # gives Adam real state to round-trip

    ckpt = fam.trainer.save_checkpoint(model, Wav2Vec2FeatureExtractor(), optimizer, 7, tmp_path)

    assert ckpt == tmp_path / "checkpoint-7"
    assert not list(tmp_path.glob(".tmp-*")) and not list(tmp_path.glob("*.tmp"))
    assert list_checkpoints(tmp_path, extra_files=(HEAD_FILE,)) == [ckpt]

    fresh = fam.tiny()
    fresh.load_checkpoint(ckpt)
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True):
        assert torch.equal(a, b)  # backbone AND head restored

    state = torch.load(ckpt / TRAINING_STATE_FILE, map_location="cpu")
    assert state["global_step"] == 7
    opt2 = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    opt2.load_state_dict(state["optimizer"])
    assert len(opt2.state) == len(optimizer.state) > 0


def test_the_saved_backbone_still_loads_as_the_plain_encoder(fam, tmp_path):
    """select_checkpoint / slsb load these directories as upstreams."""
    model = fam.tiny()
    ckpt = fam.trainer.save_checkpoint(
        model, Wav2Vec2FeatureExtractor(), torch.optim.AdamW(model.parameters()), 1, tmp_path
    )
    assert type(fam.encoder_cls.from_pretrained(ckpt)) is fam.encoder_cls


def test_incomplete_and_tmp_checkpoints_are_never_resumed(fam, tmp_path):
    model = fam.tiny()
    opt = torch.optim.AdamW(model.parameters())
    fam.trainer.save_checkpoint(model, Wav2Vec2FeatureExtractor(), opt, 5, tmp_path)

    (tmp_path / "checkpoint-9.tmp").mkdir()  # crashed mid-save (the branches' layout)
    (tmp_path / ".tmp-checkpoint-10").mkdir()  # crashed mid-save (this layout)
    half = tmp_path / "checkpoint-8"  # backbone but no head / optimizer state
    half.mkdir()
    getattr(model, fam.attr).save_pretrained(half)

    found = list_checkpoints(tmp_path, extra_files=(HEAD_FILE,))
    assert [p.name for p in found] == ["checkpoint-5"]
    assert list_checkpoints(tmp_path / "missing", extra_files=(HEAD_FILE,)) == []


def test_a_checkpoint_in_the_old_branch_layout_is_complete(fam, tmp_path):
    """Checkpoints written by the model branches before the unification must still resume:
    same file names (backbone weights, head, training_state with optimizer + global_step)."""
    model = fam.tiny()
    opt = torch.optim.AdamW(model.parameters())
    ckpt = tmp_path / "checkpoint-3"
    ckpt.mkdir()
    model.save_pretrained(ckpt)
    torch.save({"optimizer": opt.state_dict(), "global_step": 3}, ckpt / TRAINING_STATE_FILE)
    assert list_checkpoints(tmp_path, extra_files=(HEAD_FILE,)) == [ckpt]


def test_layerdrop_is_applied_from_the_config(fam, tmp_path):
    """The large checkpoints ship layerdrop=0.1, which leaves parameters unused in a step."""
    fam.encoder_cls(fam.config_cls(layerdrop=0.1, **CONFIG)).save_pretrained(tmp_path)
    assert fam.encoder_cls.from_pretrained(tmp_path).config.layerdrop == 0.1
    model = fam.wrapper_cls(str(tmp_path), num_clusters=4, layerdrop=0.0)
    assert model.config.layerdrop == 0.0


def test_a_model_that_cannot_apply_masks_is_rejected(fam):
    with pytest.raises(ValueError, match="masked_spec_embed"):
        fam.wrapper_cls.from_config(
            fam.config_cls(apply_spec_augment=False, **CONFIG), num_clusters=4
        )


# ── monitoring ──


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


# ── end to end: train() on real Shar shards + a label file ──


def _setup(fam, tmp_path, return_attention_mask=False):
    base = tmp_path / "base"
    torch.manual_seed(0)
    fam.encoder_cls(fam.config_cls(layerdrop=0.1, **CONFIG)).save_pretrained(base)
    Wav2Vec2FeatureExtractor(
        do_normalize=True, sampling_rate=16000, return_attention_mask=return_attention_mask
    ).save_pretrained(base)
    model = fam.encoder_cls.from_pretrained(base)

    raw = tmp_path / "raw"
    raw.mkdir()
    cuts, labels = [], {}
    rng = np.random.default_rng(0)
    for i in range(8):
        n = 12000 + 500 * i
        path = raw / f"{i}.wav"
        sf.write(path, (rng.standard_normal(n) * 0.1).astype("float32"), 16000)
        rec = Recording.from_file(path, recording_id=f"c{i}")
        cuts.append(MonoCut(id=f"c{i}", start=0, duration=rec.duration, channel=0, recording=rec))
        labels[f"c{i}"] = rng.integers(
            0, K, int(model._get_feat_extract_output_lengths(n))
        ).tolist()
    shars = tmp_path / "shars"
    shard_cuts(CutSet.from_cuts(cuts), str(shars), shard_size=4, audio_format="flac")
    labels_path = tmp_path / "labels.jsonl.gz"
    with gzip.open(labels_path, "wt", encoding="utf-8") as f:
        for cid, lab in labels.items():
            f.write(json.dumps({"id": cid, "labels": lab}) + "\n")
    return base, shars, labels_path


def _params(fam, tmp_path, base, shars, labels_path, max_updates):
    return Params(
        select=SelectConfig(
            seed=1,
            target_hours=1.0,
            language_mix={"sinhala": 1.0},
            min_segment_seconds=1.0,
            max_segment_seconds=30.0,
            max_hours_per_channel=10.0,
            holdout_hours=0.1,
            catalog_path="unused",
            exclude_channels_file="unused",
            drive_root="unused",
            drive_index_path="unused",
        ),
        shard=ShardConfig(
            shard_size=4,
            min_cut_seconds=5.0,
            target_cut_seconds=15.0,
            max_cut_seconds=30.0,
            output_dir=str(shars),
        ),
        model=ModelConfig(trainer=fam.name, tag=f"{fam.name}-tiny"),
        cluster=ClusterConfig(
            layer=1,
            num_clusters=K,
            seed=1,
            sample_hours=1.0,
            max_frames=1000,
            kmeans_output="unused",
            labels_output=str(labels_path),
        ),
        pretrain=PretrainConfig(
            base_model=str(base),
            precision="fp32",
            max_updates=max_updates,
            warmup_updates=1,
            peak_lr=1e-3,
            hold_ratio=0.4,
            adam_betas=[0.9, 0.98],
            adam_eps=1e-6,
            weight_decay=0.01,
            max_grad_norm=1.0,
            utterance_mix_prob=fam.mix_prob,
            layerdrop=0.0,  # overrides the checkpoint's 0.1 (see the assert below)
            head_lr_mult=10.0,
            target_batch_seconds=2,
            per_device_max_seconds=1,
            num_workers=0,
            mask_time_prob=0.65,
            mask_time_length=10,
            labels_path=str(labels_path),
            save_every_updates=2,
            eval_every_updates=2,
            keep_last_n_checkpoints=5,
            milestone_checkpoints=[],
            output_dir=str(tmp_path / "out"),
        ),
    )


def test_train_runs_checkpoints_atomically_and_resumes(fam, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    base, shars, labels_path = _setup(fam, tmp_path)
    run = lambda updates: fam.trainer.train(
        _params(fam, tmp_path, base, shars, labels_path, updates), use_mlflow=False
    )

    assert run(4) == 4
    out = tmp_path / "out"
    assert list_checkpoints(out, (HEAD_FILE,))[0].name == "checkpoint-4"
    assert not list(out.glob("*.tmp")) and not list(out.glob(".tmp-*"))
    ckpt = out / "checkpoint-4"
    assert (ckpt / HEAD_FILE).exists() and (ckpt / TRAINING_STATE_FILE).exists()
    assert (ckpt / "preprocessor_config.json").exists()
    state = torch.load(ckpt / TRAINING_STATE_FILE, map_location="cpu")
    lrs = sorted({g["lr_mult"] for g in state["optimizer"]["param_groups"]})
    assert lrs == [1.0, 10.0]  # head trains at head_lr_mult x the encoder LR
    assert (
        fam.encoder_cls.from_pretrained(out).config.layerdrop == 0.0
    )  # pretrain.layerdrop applied

    curves = (tmp_path / "reports" / "pretrain_curves.csv").read_text().splitlines()
    assert curves[0].startswith("step,loss,masked_accuracy,unmasked_accuracy,pred_perplexity")
    assert [int(r.split(",")[0]) for r in curves[1:]] == [2, 4]

    capsys.readouterr()
    assert run(6) == 6
    printed = capsys.readouterr().out
    assert "Resuming" in printed and "step 4" in printed
    assert list_checkpoints(out, (HEAD_FILE,))[0].name == "checkpoint-6"
    curves = (tmp_path / "reports" / "pretrain_curves.csv").read_text().splitlines()
    assert [int(r.split(",")[0]) for r in curves[1:]] == [2, 4, 6]  # appended, not restarted
    assert "Waveform normalisation (do_normalize): True" in printed


def test_explicit_resume_and_no_resume(fam, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    base, shars, labels_path = _setup(fam, tmp_path)
    params = lambda updates: _params(fam, tmp_path, base, shars, labels_path, updates)
    fam.trainer.train(params(4), use_mlflow=False)
    capsys.readouterr()

    fam.trainer.train(params(6), use_mlflow=False, resume_from=tmp_path / "out" / "checkpoint-2")
    assert "at step 2" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="not a complete checkpoint"):
        fam.trainer.train(params(6), use_mlflow=False, resume_from=tmp_path / "nowhere")
    fam.trainer.train(params(2), use_mlflow=False, auto_resume=False)
    assert "Starting from the base model" in capsys.readouterr().out


def _masks_seen_by_the_model(fam, tmp_path, monkeypatch, return_attention_mask):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    base, shars, labels_path = _setup(fam, tmp_path, return_attention_mask)
    seen = []
    original = fam.wrapper_cls.forward

    def spy(self, input_values, attention_mask=None, mask_time_indices=None):
        seen.append(attention_mask)
        return original(self, input_values, attention_mask, mask_time_indices)

    monkeypatch.setattr(fam.wrapper_cls, "forward", spy)
    fam.trainer.train(_params(fam, tmp_path, base, shars, labels_path, 2), use_mlflow=False)
    return seen


@pytest.mark.parametrize("return_attention_mask", [False, True])
def test_the_attention_mask_is_always_passed(fam, tmp_path, monkeypatch, return_attention_mask):
    """Every model is trained with the padding mask, whatever its feature extractor's
    return_attention_mask says (mHuBERT-147's says False). That is how the per-model branches
    trained, and how SLSB evaluates: it requests and passes the mask for every upstream."""
    seen = _masks_seen_by_the_model(fam, tmp_path, monkeypatch, return_attention_mask)
    assert seen and all(m is not None for m in seen)
