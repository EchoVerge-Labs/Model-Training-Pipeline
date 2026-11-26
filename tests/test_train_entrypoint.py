"""pipeline.train: picks the family trainer from the model and forwards the other arguments."""

import sys

import pytest

from pipeline import train
from pipeline.training import contrastive as train_contrastive
from pipeline.training import masked_prediction as train_masked_prediction


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_masked_prediction, "main", lambda: calls.append(("masked", sys.argv[1:]))
    )
    monkeypatch.setattr(
        train_contrastive, "main", lambda: calls.append(("contrastive", sys.argv[1:]))
    )
    return calls


@pytest.mark.parametrize(
    ("config", "family"),
    [
        ("params-mhubert147.yaml", "masked"),
        ("params-hubert.yaml", "masked"),
        ("params-wavlm.yaml", "masked"),
        ("params-xlsr.yaml", "contrastive"),
        ("params-wav2vec2.yaml", "contrastive"),
    ],
)
def test_the_family_trainer_is_chosen_from_the_params_files_model(recorded, config, family):
    train.main(["--config", config])
    assert recorded == [(family, ["--config", config])]


def test_model_alone_selects_that_models_params_file(recorded):
    train.main(["--model", "wavlm"])
    assert recorded == [("masked", ["--config", "params-wavlm.yaml"])]


def test_no_arguments_trains_mhubert147(recorded):
    train.main([])
    assert recorded == [("masked", ["--config", "params-mhubert147.yaml"])]


def test_other_arguments_reach_the_family_trainer(recorded):
    train.main(["--model", "xlsr", "--resume-from", "models/x/checkpoint-3000"])
    assert recorded == [
        (
            "contrastive",
            ["--config", "params-xlsr.yaml", "--resume-from", "models/x/checkpoint-3000"],
        )
    ]


def test_model_that_does_not_match_the_config_is_an_error(recorded):
    with pytest.raises(SystemExit, match="model_key=hubert"):
        train.main(["--model", "wavlm", "--config", "params-hubert.yaml"])
    assert recorded == []
