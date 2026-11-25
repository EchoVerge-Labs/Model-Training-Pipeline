"""The DagsHub record of a pre-training run.

Every run is ONE MLflow run in the `ssl-pretraining` experiment of the Model-Training-Pipeline
repository (`mlflow.tracking_uri` / `mlflow.experiment_name` in the params file), named after
the model's SLSB leaderboard row (Params.model_name, e.g. wavlm-large-copt200h), so a run and
its benchmark result share a name:

  tags       kind=adapted, family, base_model, data=<experiment>, slsb.model=<model name>, note
  params     experiment, output_dir, params_commit, the training settings; at the end `trained`
             (start to end), and from `make ckpt` `selected_checkpoint`
  metrics    the training curves (train/*); from `make ckpt` checkpoint_selection/<checkpoint>
  artifacts  params.yaml, pretrain_curves.csv, pretrain_metrics.json, kmeans_stats.json (HuBERT
             family), checkpoint_selection.json

A resumed run continues the same MLflow run: its id is kept in <output_dir>/mlflow_run_id.txt
next to the checkpoints it resumes from. A run that stops before max_updates ends KILLED, one
that crashes FAILED, so FINISHED means trained to the end. Downstream results go to the SLSB
leaderboard instead (pipeline.eval.benchmark).
"""

import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline.common.callbacks import MLflowLogger
from pipeline.common.schema import Params
from pipeline.models.registry import get_model_spec

RUN_ID_FILE = "mlflow_run_id.txt"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def mlflow_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f).get("mlflow", {}) or {}


def run_tags(params: Params) -> dict:
    spec = get_model_spec(params.model_key)
    return {
        "kind": "adapted",
        "family": spec.slsb_family,
        "base_model": params.pretrain.base_model,
        "data": params.experiment,
        "slsb.model": params.model_name,
        "mlflow.note.content": (
            f"{spec.slsb_name}: {params.pretrain_hours} h continued pre-training from "
            f"{params.pretrain.base_model} (experiment {params.experiment}, "
            f"{'frozen' if params.pretrain.freeze_feature_encoder else 'trainable'} CNN feature "
            f"encoder, {params.pretrain.max_updates} updates). Checkpoints: "
            f"{params.pretrain.output_dir}."
        ),
    }


def nondefault_method_params(cfg) -> dict:
    """The training-method knobs, logged only when a run changes them from the default (full
    CoPT, one learning rate): otherwise they are the same constant column in every run."""
    out = {}
    if cfg.pecopt.method != "full_copt":
        out["pecopt_method"] = cfg.pecopt.method
    if cfg.head_lr_mult != 1.0:
        out["head_lr_mult"] = cfg.head_lr_mult
    return out


def start_training_run(
    params: Params, config_path: str, resume_step: int, run_params: dict
) -> MLflowLogger | None:
    """Open (or, on resume, reopen) the run's MLflow run and log its tags and params."""
    cfg = mlflow_config(config_path)
    if not cfg.get("tracking_uri"):
        return None
    id_file = Path(params.pretrain.output_dir) / RUN_ID_FILE
    previous = id_file.read_text().strip() if id_file.is_file() else None
    run_id = previous if resume_step > 0 else None
    if resume_step > 0 and not previous:
        print(f"WARNING: no {id_file}; logging this resumed run as a new MLflow run")

    def open_run(rid):
        return MLflowLogger(
            tracking_uri=cfg["tracking_uri"],
            experiment_name=cfg.get("experiment_name", "ssl-pretraining"),
            run_name=params.model_name,
            run_id=rid,
        )

    logger = open_run(run_id)
    if run_id and not logger.enabled:  # e.g. the run was deleted: start a fresh one
        logger = open_run(None)
    if not logger.enabled:
        return None
    if previous and previous != logger.run_id:
        _supersede(previous)
    id_file.parent.mkdir(parents=True, exist_ok=True)
    id_file.write_text(logger.run_id + "\n")
    logger.set_tags(run_tags(params))
    logger.log_params(
        {
            "experiment": params.experiment,
            "output_dir": params.pretrain.output_dir,
            "params_commit": git_commit(),
            **run_params,
        }
    )
    return logger


def _supersede(run_id: str) -> None:
    """A run restarted from scratch replaces its earlier, unfinished MLflow run: end that one as
    KILLED rather than leave it RUNNING forever."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        if client.get_run(run_id).info.status == "RUNNING":
            client.set_terminated(run_id, status="KILLED")
    except Exception as e:  # noqa: BLE001 -- bookkeeping only
        warnings.warn(f"could not end the superseded MLflow run {run_id}: {e}")


def finish_training_run(
    logger: MLflowLogger | None, config_path: str, completed: bool, artifacts: dict[str, Path]
) -> None:
    """Upload the run's reports, record when it trained, and end it FINISHED (reached
    max_updates) or KILLED (stopped early, e.g. by the stall/collapse detector)."""
    if logger is None:
        return
    if completed:
        try:
            import mlflow

            started_ms = mlflow.get_run(logger.run_id).info.start_time
            start = datetime.fromtimestamp(started_ms / 1000, tz=timezone.utc).astimezone()
            end = datetime.now(tz=timezone.utc).astimezone()
            logger.log_params({"trained": f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"})
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"could not record the training period: {e}")
    logger.log_artifact(config_path, "params.yaml")
    for name, path in artifacts.items():
        if Path(path).is_file():
            logger.log_artifact(str(path), name)
    logger.end("FINISHED" if completed else "KILLED")


def end_active_run(status: str) -> None:
    """End a still-open MLflow run as FAILED/KILLED (train.py, on a crash or a kill signal).
    Without this, MLflow's exit hook would end it FINISHED."""
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except Exception as e:  # noqa: BLE001 -- the run is already failing; don't mask why
        warnings.warn(f"could not end the MLflow run as {status}: {e}")


def reopen_training_run(params: Params, config_path: str) -> MLflowLogger | None:
    """The run's MLflow run, for `make ckpt` to add its checkpoint selection to."""
    cfg = mlflow_config(config_path)
    id_file = Path(params.pretrain.output_dir) / RUN_ID_FILE
    if not cfg.get("tracking_uri") or not id_file.is_file():
        print(f"NOTE: no {id_file}; the checkpoint selection is not logged to MLflow")
        return None
    run_id = id_file.read_text().strip()
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(cfg["tracking_uri"])
        status = MlflowClient().get_run(run_id).info.status
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"could not reach MLflow run {run_id}: {e}")
        return None
    logger = MLflowLogger(
        tracking_uri=cfg["tracking_uri"],
        experiment_name=cfg.get("experiment_name", "ssl-pretraining"),
        run_id=run_id,
    )
    logger.status = status  # reopening sets RUNNING; end() puts this back
    return logger if logger.enabled else None
