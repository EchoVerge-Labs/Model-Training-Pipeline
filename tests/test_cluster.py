"""fit_kmeans / assign_cluster_labels: layer features of a (tiny, random)
WavLM -> k-means -> per-cut pseudo-labels, on synthetic audio (needs lhotse)."""

import json

import numpy as np
import pytest

pytest.importorskip("lhotse")
import joblib
import soundfile as sf
import torch
from transformers import WavLMConfig, WavLMModel

from pipeline.assign_cluster_labels import assign_cut
from pipeline.fit_kmeans import collect_frames, fit_kmeans, sample_cuts
from pipeline.layer_features import layer_features, load_final_train_cuts, truncate_to_layer
from pipeline.schema import ClusterConfig, Params, PretrainConfig, SelectConfig, ShardConfig

TINY_CONFIG = WavLMConfig(
    hidden_size=16,
    num_hidden_layers=3,
    num_attention_heads=2,
    intermediate_size=32,
    conv_dim=(16, 16),
    conv_kernel=(10, 3),
    conv_stride=(5, 2),
    do_stable_layer_norm=True,
)
CPU = torch.device("cpu")


def _make(tmp_path, clips):
    """clips: [(id, relpath, seconds, channel_id)] -> writes wavs + manifest, returns manifest path."""
    raw = tmp_path / "raw"
    manifest = tmp_path / "train.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for cid, rel, seconds, channel in clips:
            (raw / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                raw / rel, (np.random.randn(int(16000 * seconds)) * 0.1).astype("float32"), 16000
            )
            f.write(
                json.dumps(
                    {
                        "id": cid,
                        "filename": rel,
                        "language": "sinhala",
                        "genre": "drama",
                        "channel_id": channel,
                    }
                )
                + "\n"
            )
    return manifest, raw


def _params():
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
            shard_size=10,
            min_cut_seconds=5.0,
            target_cut_seconds=15.0,
            max_cut_seconds=30.0,
            output_dir="unused",
        ),
        cluster=ClusterConfig(
            layer=2,
            num_clusters=4,
            seed=1,
            sample_hours=1.0,
            max_frames=10_000,
            kmeans_output="unused",
            labels_output="unused",
        ),
        pretrain=PretrainConfig(
            base_model="unused",
            max_updates=10,
            warmup_updates=1,
            peak_lr=5e-5,
            hold_ratio=0.4,
            adam_betas=[0.9, 0.98],
            adam_eps=1e-6,
            weight_decay=0.01,
            max_grad_norm=1.0,
            target_batch_seconds=10,
            per_device_max_seconds=10,
            num_workers=0,
            mask_time_prob=0.65,
            mask_time_length=10,
            labels_path="unused",
            save_every_updates=5,
            eval_every_updates=5,
            keep_last_n_checkpoints=1,
            milestone_checkpoints=[10],
            output_dir="unused",
        ),
    )


def test_load_final_train_cuts_matches_shard_ids(tmp_path):
    manifest, raw = _make(
        tmp_path,
        [
            ("long", "Sinhala/drama/vidA/long.wav", 20.0, "s1"),
            *[(f"short-{i}", f"Sinhala/drama/vidB/s{i}.wav", 1.0, "s2") for i in range(6)],
        ],
    )
    cuts = load_final_train_cuts(_params(), manifest_path=str(manifest), raw_dir=str(raw))
    ids = sorted(c.id for c in cuts)
    assert "long" in ids
    assert any(cid.startswith("concat-") for cid in ids)  # the 6 short clips got merged


def _feature_model():
    torch.manual_seed(0)
    model = WavLMModel(TINY_CONFIG)
    return truncate_to_layer(model, 2).eval()


def test_truncate_to_layer_gives_the_raw_output_of_that_layer():
    """For the pre-LN (large) encoder the full model's hidden_states[N] is the
    raw output of layer N; the truncated model must return exactly that, not
    the LayerNorm'd one."""
    torch.manual_seed(0)
    full = WavLMModel(TINY_CONFIG).eval()
    x = torch.randn(1, 8000)
    with torch.no_grad():
        expected = full(x, output_hidden_states=True).hidden_states[2]
        truncated = truncate_to_layer(full, 2)
        got = truncated(x).last_hidden_state
    assert len(truncated.encoder.layers) == 2
    torch.testing.assert_close(got, expected)


def test_layer_features_frame_count_matches_the_training_side_arithmetic():
    model = _feature_model()
    for n_samples in (8000, 16001, 24123):
        feats = layer_features(model, torch.randn(n_samples), True, CPU)
        assert feats.shape == (int(model._get_feat_extract_output_lengths(n_samples)), 16)


def test_fit_and_assign_labels_round_trip(tmp_path):
    manifest, raw = _make(
        tmp_path,
        [(f"c{i}", f"Sinhala/drama/vidA/c{i}.wav", 2.0, "s1") for i in range(8)],
    )
    params = _params()
    cuts = list(load_final_train_cuts(params, manifest_path=str(manifest), raw_dir=str(raw)))
    # all 8 clips are short (2s < min_cut_seconds=5.0) and share a channel, so
    # shard.py's concatenate_short_cuts merges them into one or more longer cuts
    assert len(cuts) >= 1
    assert sum(c.duration for c in cuts) == pytest.approx(16.0, abs=1e-3)

    sampled = sample_cuts(cuts, sample_hours=params.cluster.sample_hours, seed=params.cluster.seed)
    assert 0 < len(sampled) <= len(cuts)

    model = _feature_model()

    def extract(waveform):
        return layer_features(model, waveform, True, CPU)

    features = collect_frames(sampled, extract, max_frames=params.cluster.max_frames, seed=1)
    assert features.shape[1] == TINY_CONFIG.hidden_size

    km = fit_kmeans(features, num_clusters=4, seed=0)
    km_path = tmp_path / "km.joblib"
    joblib.dump(km, km_path)
    km = joblib.load(km_path)

    for cut in cuts:
        labels = assign_cut(cut, km, extract)
        # one label per CNN frame of the audio the dataloader will feed the model
        n_samples = len(cut.load_audio()[0])
        assert len(labels) == int(model._get_feat_extract_output_lengths(n_samples))
        assert all(0 <= label < 4 for label in labels)


def test_collect_frames_caps_the_frame_count():
    model = _feature_model()
    waves = [torch.randn(16000) for _ in range(5)]

    class Cut:
        def __init__(self, w):
            self.w, self.duration = w, 1.0

        def load_audio(self):
            return self.w.numpy()[None]

    feats = collect_frames(
        [Cut(w) for w in waves],
        lambda w: layer_features(model, w, True, CPU),
        max_frames=100,
        seed=0,
    )
    assert 0 < len(feats) <= 100
