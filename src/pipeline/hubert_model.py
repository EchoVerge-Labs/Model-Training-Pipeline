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

from pipeline.checkpoints import HEAD_FILE


class HubertForMaskedPrediction(nn.Module):
    def __init__(self, base_model: str, num_clusters: int, layerdrop: float = 0.0):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(base_model, layerdrop=layerdrop)
        self._check_maskable()
        self.final_proj = nn.Linear(self.hubert.config.hidden_size, num_clusters)

    @classmethod
    def from_config(cls, config: HubertConfig, num_clusters: int) -> "HubertForMaskedPrediction":
        """Builds a randomly-initialised model from a HubertConfig, with no
        from_pretrained download -- for tests and phase0-style smoke runs."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.hubert = HubertModel(config)
        obj._check_maskable()
        obj.final_proj = nn.Linear(config.hidden_size, num_clusters)
        return obj

    def _check_maskable(self):
        """HubertModel only applies masked_spec_embed to mask_time_indices when
        apply_spec_augment is on; otherwise the "masked" frames would be fed
        through unmasked and the objective would be trivial."""
        if not self.hubert.config.apply_spec_augment or not hasattr(
            self.hubert, "masked_spec_embed"
        ):
            raise ValueError(
                "base model has no masked_spec_embed / apply_spec_augment is off -- "
                "masked prediction would silently see unmasked input"
            )

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
        torch.save(self.final_proj.state_dict(), Path(save_dir) / HEAD_FILE)

    def load_checkpoint(self, ckpt_dir: str | Path):
        """Restores backbone and head from a checkpoint written by save_pretrained."""
        backbone = HubertModel.from_pretrained(ckpt_dir)
        self.hubert.load_state_dict(backbone.state_dict())
        self.final_proj.load_state_dict(torch.load(Path(ckpt_dir) / HEAD_FILE, map_location="cpu"))
