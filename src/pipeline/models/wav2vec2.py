"""wav2vec2 Large (mono): wav2vec 2.0 contrastive, continued from the English
facebook/wav2vec2-large-lv60."""

from pipeline.models.spec import contrastive_spec

SPEC = contrastive_spec(
    "wav2vec2",
    "facebook/wav2vec2-large-lv60",
    "params-wav2vec2.yaml",
    "wav2vec2-mono",
)
