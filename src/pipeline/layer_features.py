"""Layer features of the ORIGINAL pretrained model, for WavLM k-means targets.

WavLM (like HuBERT) learns by predicting a cluster id for every masked frame. The ids are
made once, offline, before training: run the pretrained checkpoint over the
training audio, cluster the output of one of its transformer layers with
k-means (pipeline.fit_kmeans), and assign every frame to its nearest centroid
(pipeline.assign_cluster_labels). Training then continues the same model with
the usual masked-prediction cross-entropy -- no second model runs during
training.

The frame rate is the model's own CNN downsampling, so labels line up 1:1 with
the sequence length train.py computes. Waveforms are normalised exactly as in
training (datamodule.normalize_waveform), so the features seen here are the
ones the checkpoint was pretrained on.
"""

import numpy as np
import torch
from lhotse import CutSet
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from pipeline.datamodule import SAMPLE_RATE, normalize_waveform


def truncate_to_layer(model: WavLMModel, layer: int) -> WavLMModel:
    """Keeps only the first `layer` transformer layers, so last_hidden_state is
    the raw output of layer `layer` (what fairseq's extract_features(output_layer=)
    returns) and the upper layers cost nothing. For the pre-LN encoder used by
    the large models, the final encoder LayerNorm is dropped: it would
    otherwise be applied to a layer it doesn't belong to."""
    n_layers = len(model.encoder.layers)
    if not 1 <= layer <= n_layers:
        raise ValueError(f"layer must be in [1, {n_layers}], got {layer}")
    model.encoder.layers = model.encoder.layers[:layer]
    if model.config.do_stable_layer_norm:
        model.encoder.layer_norm = nn.Identity()
    return model


def load_feature_model(base_model: str, layer: int, device: torch.device) -> WavLMModel:
    model = WavLMModel.from_pretrained(base_model)
    model = truncate_to_layer(model, layer)
    model.eval().requires_grad_(False)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    return model.to(device)


@torch.no_grad()
def layer_features(
    model: WavLMModel, waveform: torch.Tensor, normalize: bool, device: torch.device
) -> np.ndarray:
    """(T', D) float32 features for one un-padded utterance."""
    if normalize:
        waveform = normalize_waveform(waveform)
    x = waveform.unsqueeze(0).to(device=device, dtype=model.dtype)
    hidden = model(x).last_hidden_state[0]
    expected = int(model._get_feat_extract_output_lengths(waveform.shape[-1]))
    if hidden.shape[0] != expected:
        raise RuntimeError(f"got {hidden.shape[0]} frames, the CNN arithmetic says {expected}")
    return hidden.float().cpu().numpy()


def load_shar_cuts(shar_dir: str, shuffle: bool = False, seed: int = 0):
    """Lazily streams the cuts pipeline.shard wrote, audio included.

    Reading Shar (not data/raw + the manifest) means training devices only
    need `dvc pull shard`, and the labels are keyed by exactly the cut ids the
    dataloader will look up -- nothing to reconstruct or keep in sync with the
    shard seed. With shuffle=True the shard order is random (cuts inside a
    shard are already shuffled across sources), so the first N hours are a
    random sample of the corpus.
    """
    return CutSet.from_shar(
        in_dir=shar_dir, shuffle_shards=shuffle, seed=seed, stateful_shuffle=False
    )


def load_waveform(cut) -> torch.Tensor:
    audio = cut.load_audio()[0]
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def base_normalizes(base_model: str) -> bool:
    """The base checkpoint's own do_normalize flag -- read, never hard-coded."""
    return bool(Wav2Vec2FeatureExtractor.from_pretrained(base_model).do_normalize)


__all__ = [
    "SAMPLE_RATE",
    "base_normalizes",
    "layer_features",
    "load_feature_model",
    "load_shar_cuts",
    "load_waveform",
    "truncate_to_layer",
]
