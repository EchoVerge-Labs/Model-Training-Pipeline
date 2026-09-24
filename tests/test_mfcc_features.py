"""mfcc_features: output-length formula and frame alignment for k-means pseudo-labels."""

import torch
from transformers import HubertConfig, HubertModel

from pipeline.mfcc_features import extract_mfcc_aligned, get_feat_extract_output_lengths

CONFIG = HubertConfig(
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
    conv_dim=(32, 32),
    conv_kernel=(10, 3),
    conv_stride=(5, 2),
)


def test_output_length_matches_a_real_hubert_model():
    model = HubertModel(CONFIG)
    for num_samples in (16000, 32000, 4000, 50001):
        expected = int(model._get_feat_extract_output_lengths(num_samples))
        assert get_feat_extract_output_lengths(CONFIG, num_samples) == expected


def test_extract_mfcc_aligned_always_returns_target_length():
    for seconds in (0.5, 1.0, 3.3, 7.0):
        num_samples = int(seconds * 16000)
        target_length = get_feat_extract_output_lengths(CONFIG, num_samples)
        waveform = torch.randn(num_samples)
        feats = extract_mfcc_aligned(waveform, sample_rate=16000, target_length=target_length)
        assert feats.shape == (target_length, 39)
