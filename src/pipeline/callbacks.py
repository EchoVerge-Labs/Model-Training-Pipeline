"""Training callbacks: MLflow logging and the per-family training-health detectors.

MaskedAccuracyStallDetector is for the HuBERT family (masked prediction);
CodebookCollapseDetector is for XLS-R / wav2vec2 (contrastive)."""

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

        # DagsHub auth via env vars -- set before the first MLflow call: set_experiment
        # creates the experiment if needed, and without credentials DagsHub answers 403.
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.environ.get("DAGSHUB_USER", ""))
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ.get("DAGSHUB_TOKEN", ""))

        # A tracking outage must never kill a multi-hour training run: if the experiment
        # can't be reached, disable logging for the rest of the run instead of raising.
        self.enabled = False
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run(run_name=run_name)
            self.enabled = True
        except Exception as e:  # noqa: BLE001 -- any MLflow/network error, logging is optional
            warnings.warn(f"MLflow logging disabled: could not start a run ({e})")

    def log_params(self, params: dict):
        self._safe(mlflow.log_params, params)

    def log_metrics(self, metrics: dict, step: int):
        self._safe(mlflow.log_metrics, metrics, step=step)

    def end(self):
        self._safe(mlflow.end_run)

    def _safe(self, fn, *args, **kwargs):
        """Run an MLflow call, but a tracking error only warns -- it never stops training."""
        if not (HAS_MLFLOW and self.enabled):
            return
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- tracking is best-effort, training is not
            warnings.warn(f"MLflow call {fn.__name__} failed (continuing): {e}")


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


class CodebookCollapseDetector:
    """Monitor codebook perplexity and warn on collapse.

    Codebook collapse is the classic failure of wav2vec2 pre-training:
    perplexity falls, contrastive loss looks fine, representations are garbage.
    """

    def __init__(
        self,
        num_codebooks: int = 2,
        alert_threshold: float = 10.0,
        consecutive_alerts: int = 5,
    ):
        """alert_threshold: a perplexity below this is a "low" reading.
        consecutive_alerts: this many low READINGS in a row stops training.

        check() is called once per logging interval (pretrain.eval_every_updates
        updates), not once per update, so 5 readings at an interval of 500 is
        2,500 updates of collapsed training.
        """
        self.num_codebooks = num_codebooks
        self.alert_threshold = alert_threshold
        self.consecutive_alerts = consecutive_alerts
        self.consecutive_low = 0

    def check(self, perplexity: float, step: int) -> bool:
        """Returns True if training should continue, False if collapse detected."""
        if perplexity < self.alert_threshold:
            self.consecutive_low += 1
            if self.consecutive_low >= self.consecutive_alerts:
                print(f"\n{'=' * 60}")
                print(f"CODEBOOK COLLAPSE DETECTED at step {step}")
                print(
                    f"Perplexity {perplexity:.2f} has been below {self.alert_threshold} "
                    f"for {self.consecutive_low} consecutive readings."
                )
                print("The codebook is not being used effectively.")
                print("Recommended: lower the learning rate or check masking config.")
                print(f"{'=' * 60}\n")
                return False
            warnings.warn(
                f"Step {step}: codebook perplexity {perplexity:.2f} is below the floor "
                f"({self.alert_threshold}) -- low reading {self.consecutive_low}/{self.consecutive_alerts}"
            )
        else:
            self.consecutive_low = 0

        return True
