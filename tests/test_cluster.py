"""fit_kmeans / assign_cluster_labels: layer features of a (tiny, random)
WavLM -> k-means -> per-cut pseudo-labels, on synthetic audio (needs lhotse)."""

import numpy as np
import pytest

pytest.importorskip("lhotse")
import joblib
import soundfile as sf
import torch
from lhotse import CutSet, MonoCut, Recording
from transformers import WavLMConfig, WavLMModel

from pipeline.assign_cluster_labels import assign_cut
from pipeline.fit_kmeans import collect_frames, fit_kmeans
from pipeline.layer_features import layer_features, load_shar_cuts, truncate_to_layer
from pipeline.shard import shard_cuts

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


def _make_shars(tmp_path, seconds_per_clip):
    """One cut per clip of synthetic audio, sharded like pipeline.shard does."""
    raw = tmp_path / "raw"
    raw.mkdir()
    cuts = []
    for i, seconds in enumerate(seconds_per_clip):
        path = raw / f"c{i}.wav"
        sf.write(path, (np.random.randn(int(16000 * seconds)) * 0.1).astype("float32"), 16000)
        rec = Recording.from_file(path, recording_id=f"c{i}")
        cuts.append(MonoCut(id=f"c{i}", start=0, duration=rec.duration, channel=0, recording=rec))
    out = tmp_path / "shars"
    shard_cuts(CutSet.from_cuts(cuts), str(out), shard_size=4, audio_format="flac")
    return str(out)


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


def test_load_shar_cuts_yields_exactly_the_sharded_ids_with_audio(tmp_path):
    shars = _make_shars(tmp_path, [2.0] * 9)
    cuts = list(load_shar_cuts(shars))
    assert sorted(c.id for c in cuts) == sorted(f"c{i}" for i in range(9))
    assert all(len(c.load_audio()[0]) == 32000 for c in cuts)  # audio comes from the shards
    # shuffled shard order still covers everything once
    shuffled = list(load_shar_cuts(shars, shuffle=True, seed=1))
    assert sorted(c.id for c in shuffled) == sorted(c.id for c in cuts)


def test_fit_and_assign_labels_round_trip(tmp_path):
    shars = _make_shars(tmp_path, [2.0 + 0.1 * i for i in range(9)])
    model = _feature_model()

    def extract(waveform):
        return layer_features(model, waveform, True, CPU)

    features, n_cuts, hours = collect_frames(
        load_shar_cuts(shars, shuffle=True, seed=1),
        extract,
        sample_hours=1.0,
        max_frames=10_000,
        seed=1,
    )
    assert n_cuts == 9 and hours == pytest.approx(sum(2.0 + 0.1 * i for i in range(9)) / 3600)
    assert features.shape[1] == TINY_CONFIG.hidden_size

    km = fit_kmeans(features, num_clusters=4, seed=0)
    km_path = tmp_path / "km.joblib"
    joblib.dump(km, km_path)
    km = joblib.load(km_path)

    for cut in load_shar_cuts(shars):
        labels = assign_cut(cut, km, extract)
        # one label per CNN frame of the audio the dataloader will feed the model
        n_samples = len(cut.load_audio()[0])
        assert len(labels) == int(model._get_feat_extract_output_lengths(n_samples))
        assert all(0 <= label < 4 for label in labels)


def test_collect_frames_stops_at_the_hour_budget_and_caps_the_frames(tmp_path):
    shars = _make_shars(tmp_path, [2.0] * 8)
    model = _feature_model()
    feats, n_cuts, hours = collect_frames(
        load_shar_cuts(shars),
        lambda w: layer_features(model, w, True, CPU),
        sample_hours=5.0 / 3600,  # 5 s: the first 3 two-second cuts cross it
        max_frames=100,
        seed=0,
    )
    assert n_cuts == 3 and hours == pytest.approx(6.0 / 3600)
    assert 0 < len(feats) <= 100
