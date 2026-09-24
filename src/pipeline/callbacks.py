"""Training callbacks: MLflow logging, codebook collapse detection."""

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
    """Monitor masked-prediction accuracy and kill the run if it never rises
    above the random-guess floor.

    For HuBERT's masked cluster-id prediction, a model that isn't learning
    anything sits at ~1/num_clusters accuracy indefinitely -- the equivalent
    of wav2vec2's codebook-collapse failure mode for this objective.
    """

    def __init__(self, num_clusters: int, alert_margin: float = 1.5):
        self.floor = alert_margin / max(num_clusters, 1)
        self.consecutive_low = 0
        self.max_consecutive_before_kill = 500

    def check(self, accuracy: float, step: int) -> bool:
        """Returns True if training should continue, False if stalled."""
        if accuracy < self.floor:
            self.consecutive_low += 1
            if self.consecutive_low >= self.max_consecutive_before_kill:
                print(f"\n{'=' * 60}")
                print(f"MASKED-PREDICTION ACCURACY STALLED at step {step}")
                print(
                    f"Accuracy {accuracy:.4f} has stayed near the random-guess floor "
                    f"({self.floor:.4f}) for {self.consecutive_low} consecutive checks."
                )
                print("Recommended: lower the learning rate or check masking config.")
                print(f"{'=' * 60}\n")
                return False
            elif self.consecutive_low % 100 == 0:
                warnings.warn(
                    f"Step {step}: masked accuracy {accuracy:.4f} near floor "
                    f"({self.floor:.4f}) for {self.consecutive_low} checks"
                )
        else:
            self.consecutive_low = 0

        return True
