"""Masked-prediction model for the HuBERT family (HuBERT, mHuBERT-147, WavLM).

transformers has no pretraining class for these models (unlike
Wav2Vec2ForPreTraining) -- their objective is masked prediction of k-means
cluster ids, not contrastive learning with a quantizer, so there is no
ready-made pretraining head to load. This wraps the plain encoder with a linear
classifier over cluster ids, the actual HuBERT setup. WavLM keeps the same
objective; its utterance mixing is a data augmentation (see pipeline.common.datamodule).

HuBERT, mHuBERT-147 and WavLM only differ in which transformers encoder class
backs the wrapper, chosen by `model_type` (the checkpoint's config.model_type).
"""

from pathlib import Path

import torch
from torch import nn
from transformers import HubertConfig, HubertModel, WavLMConfig, WavLMModel

from pipeline.common.checkpoints import HEAD_FILE

# config.model_type -> (config class, encoder class). mHuBERT-147 is model_type "hubert".
BACKBONES: dict[str, tuple[type, type]] = {
    "hubert": (HubertConfig, HubertModel),
    "wavlm": (WavLMConfig, WavLMModel),
}


def _backbone_class(model_type: str) -> type:
    if model_type not in BACKBONES:
        raise ValueError(
            f"unsupported model_type {model_type!r}; expected one of {sorted(BACKBONES)}"
        )
    return BACKBONES[model_type][1]


class MaskedPredictionModel(nn.Module):
    def __init__(self, model_type: str, base_model: str, num_clusters: int, layerdrop: float = 0.0):
        super().__init__()
        self.model_type = model_type
        self.backbone = _backbone_class(model_type).from_pretrained(base_model, layerdrop=layerdrop)
        self._check_maskable()
        # A plain linear classifier over the k-means cluster ids. The HuBERT/WavLM papers
        # use a cosine-similarity head with learnable codeword embeddings scaled by tau=0.1,
        # but that head existed for from-scratch pretraining; in CoPT the head is new and
        # trained from scratch against fresh offline cluster targets either way, and a linear
        # classifier predicts those ids just as well.
        self.final_proj = nn.Linear(self.backbone.config.hidden_size, num_clusters)

    @classmethod
    def from_config(cls, config, num_clusters: int) -> "MaskedPredictionModel":
        """Builds a randomly-initialised model from a HubertConfig / WavLMConfig, with
        no from_pretrained download -- for tests and phase0-style smoke runs."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.model_type = config.model_type
        obj.backbone = _backbone_class(config.model_type)(config)
        obj._check_maskable()
        obj.final_proj = nn.Linear(config.hidden_size, num_clusters)
        return obj

    def _check_maskable(self):
        """The encoder only applies masked_spec_embed to mask_time_indices when
        apply_spec_augment is on; otherwise the "masked" frames would be fed
        through unmasked and the objective would be trivial."""
        if not self.backbone.config.apply_spec_augment or not hasattr(
            self.backbone, "masked_spec_embed"
        ):
            raise ValueError(
                "base model has no masked_spec_embed / apply_spec_augment is off -- "
                "masked prediction would silently see unmasked input"
            )

    @property
    def config(self):
        return self.backbone.config

    def freeze_feature_encoder(self):
        self.backbone.feature_extractor._freeze_parameters()

    def _get_feat_extract_output_lengths(self, input_lengths):
        return self.backbone._get_feat_extract_output_lengths(input_lengths)

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        return self.backbone._get_feature_vector_attention_mask(
            feature_vector_length, attention_mask
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        mask_time_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits (B, T, num_clusters) over the CNN-downsampled sequence."""
        outputs = self.backbone(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
        )
        return self.final_proj(outputs.last_hidden_state)

    def save_pretrained(self, save_dir: str):
        """Saves the encoder as a standalone checkpoint of its own class (what
        pipeline.eval.select_checkpoint / slsb load as an upstream), plus the
        masked-prediction head separately for training resumption."""
        self.backbone.save_pretrained(save_dir)
        torch.save(self.final_proj.state_dict(), Path(save_dir) / HEAD_FILE)

    def load_checkpoint(self, ckpt_dir: str | Path):
        """Restores backbone and head from a checkpoint written by save_pretrained."""
        backbone = _backbone_class(self.model_type).from_pretrained(ckpt_dir)
        self.backbone.load_state_dict(backbone.state_dict())
        self.final_proj.load_state_dict(torch.load(Path(ckpt_dir) / HEAD_FILE, map_location="cpu"))
