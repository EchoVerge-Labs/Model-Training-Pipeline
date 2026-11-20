"""Training callbacks: MLflow logging, masked-prediction stall detection."""

import os
import warnings

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowLogger:
    """Log training metrics to MLflow."""

    def __init__(self, tracking_uri: str, experiment_name: str, run_name: str | None = None):
        if not HAS_MLFLOW:
            warnings.warn("mlflow not installed, logging disabled")
            return

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # DagsHub auth via env vars
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.environ.get("DAGSHUB_USER", ""))
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ.get("DAGSHUB_TOKEN", ""))

        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        if HAS_MLFLOW:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int):
        if HAS_MLFLOW:
            mlflow.log_metrics(metrics, step=step)

    def end(self):
        if HAS_MLFLOW:
            mlflow.end_run()


class MaskedAccuracyStallDetector:
    """Monitor masked-prediction quality and kill the run if it stays stuck.

    Stuck means either accuracy at the random-guess floor (~1/num_clusters:
    the model learns nothing) or prediction perplexity near 1 (the model
    predicts one cluster for everything -- the collapse failure mode of this
    objective; accuracy alone can hide it when one cluster, e.g. silence,
    dominates). Counted in consecutive *checks* (one per eval_every_updates),
    so max_consecutive_before_kill must be reachable within max_updates.
    """

    def __init__(
        self,
        num_clusters: int,
        max_consecutive_before_kill: int,
        alert_margin: float = 1.5,
        min_perplexity: float = 2.0,
    ):
        self.floor = alert_margin / max(num_clusters, 1)
        self.min_perplexity = min_perplexity
        self.consecutive_low = 0
        self.max_consecutive_before_kill = max_consecutive_before_kill

    def check(self, accuracy: float, step: int, perplexity: float | None = None) -> bool:
        """Returns True if training should continue, False if stalled."""
        stalled = accuracy < self.floor or (
            perplexity is not None and perplexity < self.min_perplexity
        )
        if not stalled:
            self.consecutive_low = 0
            return True

        self.consecutive_low += 1
        if self.consecutive_low >= self.max_consecutive_before_kill:
            print(f"\n{'=' * 60}")
            print(f"MASKED PREDICTION STALLED at step {step}")
            print(
                f"Accuracy {accuracy:.4f} (random-guess floor {self.floor:.4f}), "
                f"perplexity {perplexity if perplexity is None else round(perplexity, 2)} "
                f"(collapse below {self.min_perplexity}) for {self.consecutive_low} consecutive checks."
            )
            print("Recommended: check the labels, normalisation and masking config.")
            print(f"{'=' * 60}\n")
            return False
        warnings.warn(
            f"Step {step}: masked accuracy {accuracy:.4f} / perplexity {perplexity} "
            f"stalled for {self.consecutive_low} checks"
        )
        return True
