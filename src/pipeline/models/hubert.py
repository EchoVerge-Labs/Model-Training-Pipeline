"""HuBERT Large: masked prediction, continued from facebook/hubert-large-ll60k."""

from pipeline.models.spec import masked_spec

SPEC = masked_spec(
    "hubert",
    "facebook/hubert-large-ll60k",
    "hubert",
    "HubertConfig",
    "HubertModel",
    1024,
    "params-hubert.yaml",
    "hubert-large",
)
