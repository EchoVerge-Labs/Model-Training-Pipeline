"""Every trainer logs to the experiment params.yaml names, under the run name its model config
gives, so all models' runs land side by side (on DagsHub: the `ssl-pretraining` experiment).

A local sqlite store stands in for DagsHub; nothing here touches the network.
"""

import mlflow
import pytest
import torch
from mlflow.tracking import MlflowClient

from pipeline.schema import MlflowConfig, ModelConfig
from tests import test_masked_trainers as masked
from tests import test_resume as xlsr


def _runs(uri, experiment):
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(experiment)
    assert exp is not None, f"experiment {experiment!r} was not created"
    return client, client.search_runs([exp.experiment_id])


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("DAGSHUB_USER", "")
    monkeypatch.setenv("DAGSHUB_TOKEN", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    mlflow.end_run()


def test_xlsr_trainer_logs_run_name_params_and_metrics(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    base = xlsr._make_base(tmp_path)
    params = xlsr._params(tmp_path, base, max_updates=2, save_every_updates=2)
    params = params.model_copy(
        update={
            "model": ModelConfig(trainer="xlsr", tag="wav2vec2"),
            "mlflow": MlflowConfig(tracking_uri=uri, experiment_name="ssl-pretraining"),
        }
    )
    from pipeline.models.xlsr.trainer import train

    train(params, dataloader=xlsr._batches(), reports_dir=tmp_path / "r", device="cpu")
    _, runs = _runs(uri, "ssl-pretraining")
    (run,) = runs
    assert run.info.run_name == "wav2vec2-si-ta-2.0s"  # <tag>-si-ta-<target_batch_seconds>s
    assert run.data.params["base_model"] == str(base)
    assert "train/codebook_perplexity" in run.data.metrics
    assert run.info.status == "FINISHED"


@pytest.mark.parametrize("fam", masked.FAMILIES, ids=lambda f: f.name)
def test_masked_prediction_trainers_log_to_the_same_experiment(fam, tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    base, shars, labels = masked._setup(fam, tmp_path)
    params = masked._params(fam, tmp_path, base, shars, labels, max_updates=2)
    params = params.model_copy(
        update={
            "model": ModelConfig(trainer=fam.name, tag=f"{fam.name}-large"),
            "mlflow": MlflowConfig(tracking_uri=uri, experiment_name="ssl-pretraining"),
        }
    )
    fam.trainer.train(params)
    _, runs = _runs(uri, "ssl-pretraining")
    (run,) = runs
    assert run.info.run_name == f"{fam.name}-large-si-ta-2.0s"
    assert run.data.params["num_clusters"] == str(masked.K)
    # only WavLM mixes utterances, and only its runs record the setting (as on its branch)
    assert ("utterance_mix_prob" in run.data.params) == (fam.name == "wavlm")
    assert "train/masked_accuracy" in run.data.metrics
    assert run.info.status == "FINISHED"


def test_a_resumed_run_is_named_so_in_the_same_experiment(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    fam = masked.FAMILIES[0]
    base, shars, labels = masked._setup(fam, tmp_path)

    def run(updates):
        p = masked._params(fam, tmp_path, base, shars, labels, max_updates=updates)
        p = p.model_copy(
            update={
                "model": ModelConfig(trainer=fam.name, tag="wavlm-large"),
                "mlflow": MlflowConfig(tracking_uri=uri, experiment_name="ssl-pretraining"),
            }
        )
        fam.trainer.train(p)
        mlflow.end_run()

    run(2)
    run(4)
    _, runs = _runs(uri, "ssl-pretraining")
    assert sorted(r.info.run_name for r in runs) == [
        "wavlm-large-si-ta-2.0s",
        "wavlm-large-si-ta-2.0s-resume2",
    ]


def test_runs_carry_the_leaderboard_name_and_the_tags_that_link_them_to_it(tmp_path):
    """run_name is also the model's name in the SLSB leaderboard; tags say what the run is."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    fam = masked.FAMILIES[0]
    base, shars, labels = masked._setup(fam, tmp_path)
    params = masked._params(fam, tmp_path, base, shars, labels, max_updates=2)
    params = params.model_copy(
        update={
            "model": ModelConfig(
                trainer="wavlm",
                tag="wavlm-large",
                family="wavlm",
                run_name="wavlm-large-copt200h",
            ),
            "mlflow": MlflowConfig(tracking_uri=uri, experiment_name="ssl-pretraining"),
        }
    )
    fam.trainer.train(params)
    _, runs = _runs(uri, "ssl-pretraining")
    (run,) = runs
    assert run.info.run_name == "wavlm-large-copt200h"
    tags = run.data.tags
    assert tags["slsb.model"] == "wavlm-large-copt200h"
    assert (tags["family"], tags["kind"], tags["method"], tags["data"]) == (
        "wavlm",
        "adapted",
        "copt",
        "si-ta-200h",
    )
    assert tags["base_model"] == str(base)
