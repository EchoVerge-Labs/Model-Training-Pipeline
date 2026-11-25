"""The DagsHub record of a training run (pipeline.common.tracking), against a local MLflow
file store: one run per training run across resumes, named after the leaderboard row, ended
FINISHED only when training completed."""

from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow.tracking import MlflowClient

from pipeline.common import tracking
from pipeline.common.schema import Params

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mlflow": {
                    "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                    "experiment_name": "e",
                }
            }
        )
    )
    params = Params.from_yaml(str(REPO / "params-wavlm.yaml"))
    params.pretrain.output_dir = str(tmp_path / "models" / params.run)
    yield params, str(config)
    if mlflow.active_run():
        mlflow.end_run()


def test_a_resumed_run_continues_the_same_mlflow_run(setup, tmp_path):
    params, config = setup
    first = tracking.start_training_run(params, config, 0, {"max_updates": 13500})
    first.log_metrics({"train/loss": 5.0}, step=500)
    run_id = first.run_id
    mlflow.end_run(status="KILLED")  # e.g. a power cut / kill signal

    again = tracking.start_training_run(params, config, 500, {"max_updates": 13500})
    assert again.run_id == run_id
    again.log_metrics({"train/loss": 4.0}, step=1000)
    curves = tmp_path / "curves.csv"
    curves.write_text("step,loss\n500,5\n1000,4\n")
    tracking.finish_training_run(again, config, True, {"pretrain_curves.csv": curves})

    run = MlflowClient().get_run(run_id)
    assert run.info.status == "FINISHED" and run.info.run_name == "wavlm-large-copt200h"
    assert run.data.tags["slsb.model"] == "wavlm-large-copt200h"
    assert run.data.tags["kind"] == "adapted" and run.data.tags["family"] == "wavlm"
    assert run.data.params["experiment"] == "si-ta-200h" and "trained" in run.data.params
    steps = [m.step for m in MlflowClient().get_metric_history(run_id, "train/loss")]
    assert steps == [500, 1000]
    artifacts = {a.path for a in MlflowClient().list_artifacts(run_id)}
    assert artifacts == {"params.yaml", "pretrain_curves.csv"}


def test_a_restart_from_scratch_supersedes_the_unfinished_run(setup):
    params, config = setup
    first = tracking.start_training_run(params, config, 0, {})
    old = first.run_id
    mlflow.end_run(status="RUNNING")  # the process died without ending it
    MlflowClient().update_run(old, status="RUNNING")
    second = tracking.start_training_run(params, config, 0, {})
    assert second.run_id != old
    assert MlflowClient().get_run(old).info.status == "KILLED"


def test_an_unfinished_run_ends_killed_and_params_are_never_overwritten(setup):
    params, config = setup
    logger = tracking.start_training_run(params, config, 0, {"max_updates": 13500})
    with pytest.warns(UserWarning, match="max_updates"):
        logger.log_params({"max_updates": 20000})
    tracking.finish_training_run(logger, config, False, {})
    run = MlflowClient().get_run(logger.run_id)
    assert run.info.status == "KILLED" and run.data.params["max_updates"] == "13500"
    assert "trained" not in run.data.params
