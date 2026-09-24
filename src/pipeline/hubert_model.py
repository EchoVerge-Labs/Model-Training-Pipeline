"""HuBERT masked-prediction model.

transformers has no HubertForPreTraining (unlike Wav2Vec2ForPreTraining) --
HuBERT's real pretraining objective is masked prediction of k-means cluster
ids, not contrastive learning with a quantizer, so there's no ready-made
pretraining head to load. This wraps the plain HubertModel encoder with a
linear classifier over cluster ids, the actual HuBERT setup.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import HubertConfig, HubertModel


class HubertForMaskedPrediction(nn.Module):
    def __init__(self, base_model: str, num_clusters: int):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(base_model)
        self.final_proj = nn.Linear(self.hubert.config.hidden_size, num_clusters)

    @classmethod
    def from_config(cls, config: HubertConfig, num_clusters: int) -> "HubertForMaskedPrediction":
        """Builds a randomly-initialised model from a HubertConfig, with no
        from_pretrained download -- for tests and phase0-style smoke runs."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.hubert = HubertModel(config)
        obj.final_proj = nn.Linear(config.hidden_size, num_clusters)
        return obj

    @property
    def config(self):
        return self.hubert.config

    def freeze_feature_encoder(self):
        self.hubert.feature_extractor._freeze_parameters()

    def _get_feat_extract_output_lengths(self, input_lengths):
        return self.hubert._get_feat_extract_output_lengths(input_lengths)

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        return self.hubert._get_feature_vector_attention_mask(feature_vector_length, attention_mask)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mask_time_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits (B, T, num_clusters) over the CNN-downsampled sequence."""
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
        )
        return self.final_proj(outputs.last_hidden_state)

    def save_pretrained(self, save_dir: str):
        """Saves the encoder as a standalone HubertModel checkpoint (what
        select_checkpoint.py / slsb load as an upstream), plus the
        masked-prediction head separately for training resumption."""
        self.hubert.save_pretrained(save_dir)
        torch.save(self.final_proj.state_dict(), Path(save_dir) / "masked_prediction_head.pt")
