"""XLS-R 300M: wav2vec 2.0 contrastive, continued from the multilingual
facebook/wav2vec2-xls-r-300m."""

from pipeline.models.spec import contrastive_spec

SPEC = contrastive_spec(
    "xlsr",
    "facebook/wav2vec2-xls-r-300m",
    "params-xlsr.yaml",
    "xlsr300m",
)
