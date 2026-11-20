"""Small pieces the HuBERT-style trainers (hubert, wavlm) have in common.

Not a framework: each of those trainers still owns its own loop and model. These are
just the file name and the one metric both of them use.
"""

import torch

# The linear classifier over cluster ids, saved next to the encoder in a checkpoint. The
# encoder itself is saved as a plain HubertModel / WavLMModel, which is what
# `slsb --upstream` loads; the head is only needed to resume training.
HEAD_FILE = "masked_prediction_head.pt"


def prediction_perplexity(prob_sum: torch.Tensor, n_frames: int) -> float:
    """exp(entropy) of the average predicted distribution over masked frames:
    ~1 means the model predicts one cluster for everything (collapse), up to
    num_clusters when it uses them all evenly."""
    mean_probs = prob_sum / max(n_frames, 1)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-10))).sum()
    return float(torch.exp(entropy).item())
