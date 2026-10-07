"""MLflowLogger must put the DagsHub credentials in the environment before its
first MLflow call (set_experiment creates the experiment; unauthenticated, DagsHub
answers 403 and nothing is logged)."""
from unittest import mock

from pipeline import callbacks


def test_credentials_are_set_before_the_first_mlflow_call(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    monkeypatch.setenv("DAGSHUB_USER", "user")
    monkeypatch.setenv("DAGSHUB_TOKEN", "token")
    seen = {}

    def set_experiment(name):
        seen["auth"] = (callbacks.os.environ.get("MLFLOW_TRACKING_USERNAME"),
                        callbacks.os.environ.get("MLFLOW_TRACKING_PASSWORD"))

    with mock.patch.object(callbacks.mlflow, "set_tracking_uri"), \
         mock.patch.object(callbacks.mlflow, "set_experiment", side_effect=set_experiment), \
         mock.patch.object(callbacks.mlflow, "start_run"):
        callbacks.MLflowLogger("https://example.invalid/x.mlflow", "ssl-pretraining")
    assert seen["auth"] == ("user", "token")
