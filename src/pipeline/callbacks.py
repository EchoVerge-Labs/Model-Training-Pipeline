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


class CodebookCollapseDetector:
    """Monitor codebook perplexity and warn on collapse.

    Codebook collapse is the classic failure of wav2vec2 pre-training:
    perplexity falls, contrastive loss looks fine, representations are garbage.
    """

    def __init__(self, num_codebooks: int = 2, alert_threshold: float = 10.0):
        self.num_codebooks = num_codebooks
        self.alert_threshold = alert_threshold
        self.consecutive_low = 0
        self.max_consecutive_before_kill = 500

    def check(self, perplexity: float, step: int) -> bool:
        """Returns True if training should continue, False if collapse detected."""
        if perplexity < self.alert_threshold:
            self.consecutive_low += 1
            if self.consecutive_low >= self.max_consecutive_before_kill:
                print(f"\n{'=' * 60}")
                print(f"CODEBOOK COLLAPSE DETECTED at step {step}")
                print(
                    f"Perplexity {perplexity:.2f} has been below {self.alert_threshold} "
                    f"for {self.consecutive_low} consecutive steps."
                )
                print("The codebook is not being used effectively.")
                print("Recommended: lower the learning rate or check masking config.")
                print(f"{'=' * 60}\n")
                return False
            elif self.consecutive_low % 100 == 0:
                warnings.warn(
                    f"Step {step}: Codebook perplexity {perplexity:.2f} below threshold "
                    f"({self.alert_threshold}) for {self.consecutive_low} steps"
                )
        else:
            self.consecutive_low = 0

        return True
