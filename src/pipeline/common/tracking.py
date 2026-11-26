"""The DagsHub record of a pre-training run.

Every run is ONE MLflow run in the `ssl-pretraining` experiment of the Model-Training-Pipeline
repository (`mlflow.tracking_uri` / `mlflow.experiment_name` in the params file), named with
the run's one name (Params.run, e.g. wavlm-large-copt-si-ta-200h; see pipeline.common.naming),
which is also its SLSB leaderboard row:

  tags       kind=adapted, family, base_model, slsb.model=<run name>, the name's slots (model,
             method, languages, hours, cnn, passes, batching, data, tag), and a note
  params     output_dir, params_commit, the training settings; at the end `trained`
             (start to end), and from `make ckpt` `selected_checkpoint`
  metrics    the training curves (train/*); from `make ckpt` checkpoint_selection/<checkpoint>
  artifacts  params.yaml (the merged config), pretrain_curves.csv, pretrain_metrics.json,
             pretrain_language_curves.csv (homog), kmeans_stats.json (HuBERT family),
             checkpoint_selection.json

The run's folder models/<run name>/ holds run.json: the name's slots and a fingerprint of
the config the name stands for. claim_run_folder() refuses to train, select or benchmark when
the folder belongs to a different config, so one name never covers two configs.

A resumed run continues the same MLflow run: its id is kept in <output_dir>/mlflow_run_id.txt
next to the checkpoints it resumes from. A run that stops before max_updates ends KILLED, one
that crashes FAILED, so FINISHED means trained to the end. Downstream results go to the SLSB
leaderboard instead (pipeline.eval.benchmark).
"""

import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline.common.callbacks import MLflowLogger
from pipeline.common.checkpoints import checkpoint_step
from pipeline.common.naming import changed_keys, fingerprint
from pipeline.common.schema import Params
from pipeline.models.registry import get_model_spec

RUN_ID_FILE = "mlflow_run_id.txt"
RUN_FILE = "run.json"
CONFIG_FILE = "config.yaml"


def run_record(params: Params) -> dict:
    config = params.training_config
    return {
        "name": params.run,
        "parts": params.name_parts,
        "data_id": params.data_id,
        "fingerprint": fingerprint(config),
        "config": config,
    }


def claim_run_folder(params: Params, write: bool = True) -> None:
    """Refuse to use models/<run name>/ for a config other than the one it was made for.

    Its run.json records the config the name stands for. A folder whose run.json differs, or
    that holds checkpoints but no run.json (so they cannot be shown to be this run's), stops
    the stage. A fresh folder is claimed by writing run.json (write: one process per node).
    """
    folder = params.model_dir
    path = folder / RUN_FILE
    record = run_record(params)
    if path.exists():
        found = json.loads(path.read_text())
        if found.get("fingerprint") != record["fingerprint"]:
            diff = changed_keys(found.get("config", {}), record["config"])
            raise SystemExit(
                f"{folder} belongs to a different config than this {params.run} run "
                f"(differs in {diff}). A run with these settings needs its own name: give it a "
                f"tag, or remove {folder} deliberately."
            )
        return
    if any(checkpoint_step(p) is not None for p in folder.glob("checkpoint-*")):
        raise SystemExit(
            f"{folder} holds checkpoints but no {RUN_FILE}, so they cannot be shown to belong "
            f"to this config; refusing to resume them. Move them away, or write {path} by hand "
            f"if you are sure."
        )
    if write:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")


def write_run_config(params: Params, raw: dict) -> Path:
    """reports/<run>/config.yaml: the merged config (params file + variants + tag) the run
    was launched with, uploaded to DagsHub as params.yaml."""
    path = params.reports_dir / CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def mlflow_config(config_path: str, params: Params | None = None) -> dict:
    """The run's `mlflow:` settings: from its merged config (reports/<run>/config.yaml, which
    `make train` writes first, so a VARIANT overlay's settings count), else the params file."""
    merged = params.reports_dir / CONFIG_FILE if params is not None else None
    path = merged if merged is not None and merged.is_file() else Path(config_path)
    if not path.is_file():
        return {}
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("mlflow", {}) or {}


def run_tags(params: Params) -> dict:
    """The tags both DagsHub records of a run carry (its ssl-pretraining run here, its
    leaderboard row via pipeline.eval.benchmark), so either can be filtered by any slot."""
    spec = get_model_spec(params.model_key)
    parts = params.name_parts
    return {
        "kind": "adapted",
        "family": spec.slsb_family,
        "base_model": params.pretrain.base_model,
        "slsb.model": params.run,
        **parts,
        "mlflow.note.content": (
            f"{spec.slsb_name}: {parts['method']} on {parts['hours']} h of {parts['languages']} "
            f"({params.data_id}), {parts['passes']} passes, CNN {parts['cnn']}, "
            f"{parts['batching']} batches"
            + (f", tag {params.tag}" if params.tag else "")
            + f". Checkpoints: {params.pretrain.output_dir}."
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


def language_batching_params(lb) -> dict:
    """The language-batching settings of a homogeneous run (the same params the existing
    homogeneous runs on DagsHub carry); nothing for a mixed-batch run."""
    if not lb.enabled:
        return {}
    return {
        "language_batching": True,
        "language_batching_mode": lb.mode,
        "language_batching_languages": ",".join(lb.languages),
        "language_batching_probabilities": json.dumps(lb.probabilities, sort_keys=True),
        "language_batching_window": lb.window,
        "language_batching_seed": lb.seed,
    }


def start_training_run(
    params: Params, config_path: str, resume_step: int, run_params: dict
) -> MLflowLogger | None:
    """Open (or, on resume, reopen) the run's MLflow run and log its tags and params."""
    cfg = mlflow_config(config_path, params)
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
            run_name=params.run,
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
    if "params.yaml" not in artifacts:
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
    cfg = mlflow_config(config_path, params)
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
