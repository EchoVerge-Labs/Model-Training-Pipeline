"""Pydantic schema for the params*.yaml files (one per model, all validated here).

The top-level `model_key` names the model and fixes its objective family:
masked prediction on k-means cluster ids (HuBERT family) or wav2vec 2.0
contrastive learning with a Gumbel quantiser. Fields that only one family uses are
optional here, and `Params` requires them exactly when the family does.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from pipeline.adapters.config import PECoPTConfig
from pipeline.model_registry import MODEL_REGISTRY, ModelFamily


class SelectConfig(BaseModel):
    seed: int
    target_hours: float = Field(gt=0)
    language_mix: dict[str, float]
    min_segment_seconds: float = Field(gt=0)
    max_segment_seconds: float = Field(gt=0)
    max_hours_per_channel: float = Field(gt=0)
    holdout_hours: float = Field(gt=0)
    catalog_path: str
    exclude_channels_file: str
    drive_root: str
    drive_index_path: str

    @field_validator("language_mix")
    @classmethod
    def mix_sums_to_one(cls, v):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"language_mix must sum to 1.0, got {total}")
        return v


class ShardConfig(BaseModel):
    format: Literal["flac", "wav"] = "flac"
    shard_size: int = Field(gt=0)
    min_cut_seconds: float = Field(gt=0)
    target_cut_seconds: float = Field(gt=0)
    max_cut_seconds: float = Field(gt=0)
    output_dir: str

    @model_validator(mode="after")
    def cut_lengths_are_consistent(self):
        # Long clips are split into equal windows, each longer than max/2, so
        # max >= 2*min keeps every window above the minimum. Short clips are
        # merged until they reach target, overshooting it by less than min, so
        # max >= target + min keeps merged cuts within the cap.
        if self.max_cut_seconds < 2 * self.min_cut_seconds:
            raise ValueError(
                f"max_cut_seconds ({self.max_cut_seconds}) must be >= 2 * min_cut_seconds "
                f"({self.min_cut_seconds}) or split windows can fall below the minimum"
            )
        if self.max_cut_seconds < self.target_cut_seconds + self.min_cut_seconds:
            raise ValueError(
                f"max_cut_seconds ({self.max_cut_seconds}) must be >= target_cut_seconds + "
                f"min_cut_seconds ({self.target_cut_seconds + self.min_cut_seconds})"
            )
        return self


class ClusterConfig(BaseModel):
    layer: int = Field(gt=0)  # transformer layer of the base model whose output is clustered
    num_clusters: int = Field(gt=0)
    seed: int
    sample_hours: float = Field(gt=0)  # audio streamed for the k-means fit
    max_frames: int = Field(gt=0)  # cap on frames kept for the fit
    kmeans_output: str
    labels_output: str


ModelKey = Literal["mhubert147", "hubert", "wavlm", "xlsr", "wav2vec2"]


class PretrainConfig(BaseModel):
    base_model: str
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    # CNN feature encoder. false = full CoPT (the CNN trains too, as in fairseq pre-training);
    # true = transformer-only. Works for every model; a trainable CNN needs much more memory.
    freeze_feature_encoder: bool = True
    # Stock value of the HuBERT-family base models. Layerdrop leaves parameters
    # unused in a step, so those trainers turn on DDP's find_unused_parameters when
    # it's > 0. The contrastive trainers don't read it.
    layerdrop: float = Field(default=0.1, ge=0, le=1)

    max_updates: int = Field(gt=0)
    warmup_updates: int = Field(ge=0)
    peak_lr: float = Field(gt=0)
    lr_schedule: str = "tri_stage"
    hold_ratio: float = Field(ge=0, le=1)
    optimizer: str = "adamw"
    adam_betas: list[float]
    adam_eps: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    # LR multiplier for the (new, randomly initialised) masked-prediction head;
    # 1.0 = same LR as the encoder, matching the wav2vec2 run on main
    head_lr_mult: float = Field(default=1.0, gt=0)

    target_batch_seconds: float = Field(gt=0)
    per_device_max_seconds: float = Field(gt=0)
    num_workers: int = Field(ge=0)
    # GB10 memory is unified: whatever the CUDA allocator caches is taken from the
    # OS. Capping it makes a spike raise torch.OutOfMemoryError instead of starving
    # sshd and hanging the whole host. Same setting as the wav2vec2 run on main.
    max_gpu_memory_fraction: float = Field(default=0.8, gt=0, le=1)

    mask_time_prob: float = Field(ge=0, le=1)
    mask_time_length: int = Field(gt=0)

    # WavLM utterance mixing: probability that an utterance gets another one
    # from its batch mixed in (0 disables it; WavLM uses 0.2). A data
    # augmentation of the method, not a training hyperparameter.
    utterance_mix_prob: float = Field(default=0.0, ge=0, le=1)

    # ── masked prediction only (required for that family) ──
    labels_path: str | None = None

    # ── contrastive only (required for that family) ──
    diversity_loss_weight: float | None = Field(default=None, ge=0)
    max_gumbel_temperature: float | None = Field(default=None, gt=0)
    min_gumbel_temperature: float | None = Field(default=None, gt=0)
    gumbel_temperature_decay: float | None = Field(default=None, gt=0, le=1)

    save_every_updates: int = Field(gt=0)
    eval_every_updates: int = Field(gt=0)
    keep_last_n_checkpoints: int = Field(gt=0)
    milestone_checkpoints: list[int]

    output_dir: str

    # Training method: full_copt (default), or a PE-CoPT template (not implemented yet)
    pecopt: PECoPTConfig = Field(default_factory=PECoPTConfig)

    @field_validator("precision")
    @classmethod
    def no_fp16_on_blackwell(cls, v):
        if v == "fp16":
            import warnings

            warnings.warn(
                "fp16 is not recommended on GB10 Blackwell — use bf16 instead. "
                "Continuing, but you may see instability."
            )
        return v


class MonitoringConfig(BaseModel):
    """Training-health checks for the contrastive family (defaults apply if omitted)."""

    # A codebook-perplexity reading below this counts as "low".
    perplexity_floor: float = Field(default=10.0, gt=0)
    # This many low readings in a row stop training. One reading is taken every
    # pretrain.eval_every_updates updates, so the default is 5 x 500 = 2,500 updates.
    consecutive_alerts: int = Field(default=5, gt=0)


_CONTRASTIVE_FIELDS = (
    "diversity_loss_weight",
    "max_gumbel_temperature",
    "min_gumbel_temperature",
    "gumbel_temperature_decay",
)


DEFAULT_EXPERIMENT = "si-ta-200h"


def load_raw_params(path: str = "params.yaml", experiment: str | None = None) -> dict:
    """The params file as a dict with the run name resolved.

    A run is a model plus an experiment name (si-ta-200h, si-100h, ...). Its name,
    `<model prefix>-<experiment>` (e.g. wavlm-large-si-100h), replaces `{run}` in every
    key and value, so models/{run}, reports/*-{run}.json, the selected checkpoint, the
    benchmark folder and the MLflow run all carry it and two runs never share a path.
    `experiment` overrides the file's own `experiment:` (the --experiment flag).
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    experiment = experiment or raw.get("experiment") or DEFAULT_EXPERIMENT
    run = raw.get("run_name") or f"{MODEL_REGISTRY[raw['model_key']].run_name_prefix}-{experiment}"
    if "{" in experiment or "/" in experiment:
        raise ValueError(f"experiment name {experiment!r} must be a plain name")

    def fill(x):
        if isinstance(x, str):
            return x.replace("{run}", run)
        if isinstance(x, dict):
            return {fill(k): fill(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fill(v) for v in x]
        return x

    raw = fill(raw)
    raw["experiment"], raw["run_name"] = experiment, run
    return raw


class Params(BaseModel):
    model_key: ModelKey
    experiment: str = DEFAULT_EXPERIMENT  # names this run's outputs; see load_raw_params
    run_name: str | None = None  # default "<model prefix>-<experiment>"
    select: SelectConfig
    shard: ShardConfig
    cluster: ClusterConfig | None = None
    pretrain: PretrainConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @property
    def run(self) -> str:
        """Name shared by this run's checkpoints, reports, benchmark folder and MLflow run."""
        return (
            self.run_name or f"{MODEL_REGISTRY[self.model_key].run_name_prefix}-{self.experiment}"
        )

    @property
    def model_family(self) -> ModelFamily:
        return MODEL_REGISTRY[self.model_key].family

    @model_validator(mode="after")
    def family_fields_are_present(self):
        if self.model_family == "masked_prediction":
            if self.cluster is None:
                raise ValueError(f"model_key={self.model_key!r} needs a `cluster` section")
            if self.pretrain.labels_path is None:
                raise ValueError(f"model_key={self.model_key!r} needs pretrain.labels_path")
        else:
            missing = [f for f in _CONTRASTIVE_FIELDS if getattr(self.pretrain, f) is None]
            if missing:
                raise ValueError(
                    f"model_key={self.model_key!r} needs pretrain.{', pretrain.'.join(missing)}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str = "params.yaml", experiment: str | None = None) -> "Params":
        return cls(**load_raw_params(path, experiment))
