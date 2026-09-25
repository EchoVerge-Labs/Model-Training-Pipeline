"""WavLM masked-prediction model.

WavLM keeps HuBERT's objective -- masked prediction of k-means cluster ids, not
contrastive learning with a quantizer -- and transformers has no pretraining
class for it (nor a pretrained prediction head). This wraps the plain WavLMModel
encoder with a linear classifier over cluster ids. Architecture and objective
are unchanged; utterance mixing is a data augmentation (see datamodule).
"""

from pathlib import Path

import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel

from pipeline.checkpoints import HEAD_FILE


class WavLMForMaskedPrediction(nn.Module):
    def __init__(self, base_model: str, num_clusters: int, layerdrop: float = 0.0):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(base_model, layerdrop=layerdrop)
        self._check_maskable()
        self.final_proj = nn.Linear(self.wavlm.config.hidden_size, num_clusters)

    @classmethod
    def from_config(cls, config: WavLMConfig, num_clusters: int) -> "WavLMForMaskedPrediction":
        """Builds a randomly-initialised model from a WavLMConfig, with no
        from_pretrained download -- for tests and phase0-style smoke runs."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.wavlm = WavLMModel(config)
        obj._check_maskable()
        obj.final_proj = nn.Linear(config.hidden_size, num_clusters)
        return obj

    def _check_maskable(self):
        """WavLMModel only applies masked_spec_embed to mask_time_indices when
        apply_spec_augment is on; otherwise the "masked" frames would be fed
        through unmasked and the objective would be trivial."""
        if not self.wavlm.config.apply_spec_augment or not hasattr(self.wavlm, "masked_spec_embed"):
            raise ValueError(
                "base model has no masked_spec_embed / apply_spec_augment is off -- "
                "masked prediction would silently see unmasked input"
            )

    @property
    def config(self):
        return self.wavlm.config

    def freeze_feature_encoder(self):
        self.wavlm.feature_extractor._freeze_parameters()

    def _get_feat_extract_output_lengths(self, input_lengths):
        return self.wavlm._get_feat_extract_output_lengths(input_lengths)

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        return self.wavlm._get_feature_vector_attention_mask(feature_vector_length, attention_mask)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mask_time_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits (B, T, num_clusters) over the CNN-downsampled sequence."""
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
        )
        return self.final_proj(outputs.last_hidden_state)

    def save_pretrained(self, save_dir: str):
        """Saves the encoder as a standalone WavLMModel checkpoint (what
        select_checkpoint.py / slsb load as an upstream), plus the
        masked-prediction head separately for training resumption."""
        self.wavlm.save_pretrained(save_dir)
        torch.save(self.final_proj.state_dict(), Path(save_dir) / HEAD_FILE)

    def load_checkpoint(self, ckpt_dir: str | Path):
        """Restores backbone and head from a checkpoint written by save_pretrained."""
        backbone = WavLMModel.from_pretrained(ckpt_dir)
        self.wavlm.load_state_dict(backbone.state_dict())
        self.final_proj.load_state_dict(torch.load(Path(ckpt_dir) / HEAD_FILE, map_location="cpu"))
