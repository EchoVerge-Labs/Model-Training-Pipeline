"""mHuBERT-147: masked prediction, continued from the already-multilingual checkpoint
(the only base already pretrained on Sinhala/Tamil). 12-layer encoder, hidden 768."""

from pipeline.models.spec import masked_spec

SPEC = masked_spec(
    "mhubert147",
    "utter-project/mHuBERT-147",
    "hubert",
    "HubertConfig",
    "HubertModel",
    768,
    "params-mhubert147.yaml",
    "mhubert147",
    "mhubert",
)
