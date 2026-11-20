"""The models this pipeline trains, and which trainer runs each.

Three trainers exist because the training objectives differ:
  xlsr    wav2vec2 contrastive loss with a quantizer (Wav2Vec2ForPreTraining)
  hubert  masked prediction of k-means cluster ids (HubertModel + linear head)
  wavlm   the same objective on WavLMModel, plus WavLM's utterance mixing

Models whose code is the same share a trainer and differ only by their config:
wav2vec2 runs the xlsr trainer, mHuBERT-147 the hubert trainer. Each model's
configs/models/<name>.yaml says which trainer it uses (`model.trainer`).
"""

import importlib

# model name (--model) -> trainer module. The name is also the config file name.
MODELS: dict[str, str] = {
    "xlsr": "pipeline.models.xlsr.trainer",
    "wav2vec2": "pipeline.models.xlsr.trainer",
    "wavlm": "pipeline.models.wavlm.trainer",
    "hubert": "pipeline.models.hubert.trainer",
    "mhubert": "pipeline.models.hubert.trainer",
}

TRAINERS = {"xlsr", "hubert", "wavlm"}


def trainer_module(trainer: str):
    """Import pipeline.models.<trainer>.trainer lazily, so a run only loads its own."""
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    return importlib.import_module(f"pipeline.models.{trainer}.trainer")
