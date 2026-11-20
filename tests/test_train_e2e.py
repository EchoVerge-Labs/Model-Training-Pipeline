"""End-to-end train(): tiny local WavLM, real Shar shards + label file, CPU.
Covers the actual loop (normalisation, head LR groups, metrics, atomic
checkpoints) and that relaunching resumes instead of restarting."""

import gzip
import json

import numpy as np
import pytest

pytest.importorskip("lhotse")
import soundfile as sf
import torch
from lhotse import CutSet, MonoCut, Recording
from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

from pipeline.checkpoints import STATE_FILE, latest_checkpoint
from pipeline.schema import (
    ClusterConfig,
    Params,
    PretrainConfig,
    SelectConfig,
    ShardConfig,
)
from pipeline.shard import shard_cuts
from pipeline.train import train

K = 6


def _setup(tmp_path):
    base = tmp_path / "base"
    torch.manual_seed(0)
    WavLMModel(
        WavLMConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            conv_dim=(32, 32),
            conv_kernel=(10, 3),
            conv_stride=(5, 2),
            layerdrop=0.1,  # like wavlm-large; pretrain.layerdrop overrides it
        )
    ).save_pretrained(base)
    Wav2Vec2FeatureExtractor(do_normalize=True, sampling_rate=16000).save_pretrained(base)
    model = WavLMModel.from_pretrained(base)

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
        frames = int(model._get_feat_extract_output_lengths(n))
        labels[f"c{i}"] = rng.integers(0, K, frames).tolist()
    shars = tmp_path / "shars"
    shard_cuts(CutSet.from_cuts(cuts), str(shars), shard_size=4, audio_format="flac")
    labels_path = tmp_path / "labels.jsonl.gz"
    with gzip.open(labels_path, "wt", encoding="utf-8") as f:
        for cid, lab in labels.items():
            f.write(json.dumps({"id": cid, "labels": lab}) + "\n")
    return base, shars, labels_path


def _params(tmp_path, base, shars, labels_path, max_updates):
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
            utterance_mix_prob=0.5,
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


def test_train_runs_checkpoints_atomically_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    base, shars, labels_path = _setup(tmp_path)

    train(_params(tmp_path, base, shars, labels_path, max_updates=4))
    out = tmp_path / "out"
    assert latest_checkpoint(out)[0] == 4
    assert not list(out.glob("*.tmp"))
    ckpt = out / "checkpoint-4"
    assert (ckpt / "masked_prediction_head.pt").exists() and (ckpt / STATE_FILE).exists()
    assert (ckpt / "preprocessor_config.json").exists()
    state = torch.load(ckpt / STATE_FILE, map_location="cpu")
    lrs = sorted({g["lr_mult"] for g in state["optimizer"]["param_groups"]})
    assert lrs == [1.0, 10.0]  # head trains at head_lr_mult x the encoder LR
    assert WavLMModel.from_pretrained(out).config.layerdrop == 0.0  # pretrain.layerdrop applied

    curves = (tmp_path / "reports" / "pretrain_curves.csv").read_text().splitlines()
    assert curves[0].startswith("step,loss,masked_accuracy,unmasked_accuracy,pred_perplexity")
    assert [int(r.split(",")[0]) for r in curves[1:]] == [2, 4]

    capsys.readouterr()
    train(_params(tmp_path, base, shars, labels_path, max_updates=6))
    printed = capsys.readouterr().out
    assert "Resuming from" in printed and "step 4" in printed
    assert latest_checkpoint(out)[0] == 6
    curves = (tmp_path / "reports" / "pretrain_curves.csv").read_text().splitlines()
    assert [int(r.split(",")[0]) for r in curves[1:]] == [2, 4, 6]  # appended, not restarted
    assert "Waveform normalisation (do_normalize): True" in printed
