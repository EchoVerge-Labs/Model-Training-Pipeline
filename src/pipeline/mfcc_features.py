"""MFCC feature extraction for HuBERT iteration-1 k-means pseudo-labels.

HuBERT's first pretraining iteration clusters 39-dim MFCC frames (13
coefficients + delta + delta-delta) with k-means; the resulting cluster ids
become the masked-prediction targets in train.py. This module produces those
frames aligned 1:1 with HubertModel's own CNN-downsampled output length, so
labels line up with the sequence length train.py computes at training time.
"""

import numpy as np
import torch
import torchaudio
from transformers import HubertConfig

from pipeline.schema import Params
from pipeline.shard import (
    concatenate_short_cuts,
    load_manifest_as_cuts,
    shuffle_cuts,
    split_long_cuts,
)

SAMPLE_RATE = 16000


def get_feat_extract_output_lengths(config: HubertConfig, input_length: int) -> int:
    """Port of HubertModel._get_feat_extract_output_lengths -- same conv-stride
    arithmetic, without needing a model instance."""
    length = input_length
    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
        length = (length - kernel_size) // stride + 1
    return length


def extract_mfcc_aligned(
    waveform: torch.Tensor,
    sample_rate: int,
    target_length: int,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Returns (target_length, 3*n_mfcc) MFCC + delta + delta-delta features.

    Hop size is chosen to land close to target_length, then the result is
    truncated or last-frame-padded to exactly target_length -- matching the
    CNN's downsample rate exactly via hop-size math alone is fragile across
    input lengths, so we align by construction instead.
    """
    if waveform.numel() == 0 or target_length <= 0:
        return np.zeros((max(target_length, 0), 3 * n_mfcc), dtype=np.float32)

    hop_length = max(1, waveform.shape[-1] // target_length)
    win_length = min(waveform.shape[-1], hop_length * 4)

    mfcc_tf = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": max(win_length, 1),
            "hop_length": hop_length,
            "win_length": win_length,
            "n_mels": max(n_mfcc * 2, 23),
        },
    )
    mfcc = mfcc_tf(waveform.unsqueeze(0)).squeeze(0)  # (n_mfcc, T)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    feats = torch.cat([mfcc, delta, delta2], dim=0).transpose(0, 1)  # (T, 3*n_mfcc)

    feats = feats.numpy().astype(np.float32)
    t = feats.shape[0]
    if t == target_length:
        return feats
    if t > target_length:
        return feats[:target_length]
    # pad with the last frame (rare: only off-by-a-few at boundaries)
    pad = (
        np.repeat(feats[-1:], target_length - t, axis=0)
        if t > 0
        else np.zeros((target_length - t, 3 * n_mfcc), dtype=np.float32)
    )
    return np.concatenate([feats, pad], axis=0)


def load_final_train_cuts(
    params: Params,
    manifest_path: str = "data/manifests/train.jsonl",
    raw_dir: str = "data/raw",
):
    """Rebuilds the exact final cut set pipeline.shard writes to Shar (same
    split/concat/shuffle, same seed), so cluster labels are keyed by the same
    cut ids (including -pN / concat-... ids from split/merge) the dataloader
    will look up at training time."""
    cfg = params.shard
    cuts = load_manifest_as_cuts(manifest_path, raw_dir)
    cuts = split_long_cuts(cuts, cfg.max_cut_seconds)
    cuts = concatenate_short_cuts(cuts, cfg.min_cut_seconds, cfg.target_cut_seconds)
    cuts = shuffle_cuts(cuts, params.select.seed)
    return cuts


def load_waveform(cut) -> torch.Tensor:
    audio = cut.load_audio()[0]
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def cut_target_length(config: HubertConfig, cut) -> int:
    num_samples = round(cut.duration * SAMPLE_RATE)
    return get_feat_extract_output_lengths(config, num_samples)


def default_hubert_config(base_model: str) -> HubertConfig:
    return HubertConfig.from_pretrained(base_model)
