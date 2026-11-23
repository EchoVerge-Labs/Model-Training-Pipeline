"""WavLM Large: masked prediction plus utterance mixing (pretrain.utterance_mix_prob),
continued from microsoft/wavlm-large."""

from pipeline.models.spec import masked_spec

SPEC = masked_spec(
    "wavlm",
    "microsoft/wavlm-large",
    "wavlm",
    "WavLMConfig",
    "WavLMModel",
    1024,
    "params-wavlm.yaml",
    "wavlm-large",
)
