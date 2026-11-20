"""Pydantic schema for params.yaml validation."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class ModelConfig(BaseModel):
    """Which model this run trains. The default is the original XLS-R run, so a
    params.yaml without a `model` block keeps meaning what it always meant."""

    # selects pipeline/models/<trainer>/trainer.py. mHuBERT-147 uses the hubert trainer
    # and wav2vec2 the xlsr trainer: their code is the same, only their config differs.
    trainer: Literal["xlsr", "hubert", "wavlm"] = "xlsr"
    # prefix of the MLflow run name ("<tag>-si-ta-<target_batch_seconds>s")
    tag: str = "xlsr300m"
    # where a run writes its loss curves, final metrics and checkpoint-selection report;
    # one directory per model keeps their reports apart
    reports_dir: str = "reports"
    # MLflow run name, and the name of the model's run in the SLSB leaderboard (tag `slsb.model`).
    # Default: <tag>-si-ta-<target_batch_seconds>s
    run_name: str | None = None
    # e.g. xlsr, wav2vec2, wavlm, hubert, mhubert -- tag `family` on the pre-training run
    family: str | None = None
    # the pre-training data, tag `data`
    data: str = "si-ta-200h"
    # the model's baseline (frozen, not continued-pre-trained) row in the SLSB leaderboard
    frozen_name: str | None = None


class ClusterConfig(BaseModel):
    """k-means pseudo-labels: the masked-prediction targets of HuBERT-style models."""

    layer: int = Field(gt=0)  # transformer layer of the base model whose output is clustered
    num_clusters: int = Field(gt=0)
    seed: int
    sample_hours: float = Field(gt=0)  # audio streamed for the k-means fit
    max_frames: int = Field(gt=0)  # cap on frames kept for the fit
    kmeans_output: str
    labels_output: str


class MlflowConfig(BaseModel):
    tracking_uri: str = ""
    experiment_name: str = "ssl-pretraining"


class PretrainConfig(BaseModel):
    base_model: str
    # How the model is adapted. Only full continued pre-training exists so far;
    # the PEFT methods (adapters, LoRA, GC-LoRA) will be further values.
    method: Literal["copt"] = "copt"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    freeze_feature_encoder: bool = True
    # None = keep the checkpoint's own value (what the XLS-R / wav2vec2 runs do).
    # Layerdrop skips layers at random, leaving their parameters without a gradient in
    # that step, so train.py turns on DDP's find_unused_parameters when it is > 0.
    layerdrop: float | None = Field(default=None, ge=0, le=1)

    max_updates: int = Field(gt=0)
    warmup_updates: int = Field(ge=0)
    peak_lr: float = Field(gt=0)
    lr_schedule: str = "tri_stage"
    hold_ratio: float = Field(ge=0, le=1)
    adam_betas: list[float]
    adam_eps: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    # LR multiplier for the (new, randomly initialised) masked-prediction head;
    # 1.0 = same LR as the encoder. Masked-prediction trainers only.
    head_lr_mult: float = Field(default=1.0, gt=0)

    target_batch_seconds: float = Field(gt=0)
    per_device_max_seconds: float = Field(gt=0)
    num_workers: int = Field(ge=0)
    # GB10 memory is unified: whatever the CUDA allocator caches is taken from the
    # OS. Capping it makes a spike raise torch.OutOfMemoryError instead of starving
    # sshd and hanging the whole host.
    max_gpu_memory_fraction: float = Field(default=0.8, gt=0, le=1)

    mask_time_prob: float = Field(ge=0, le=1)
    mask_time_length: int = Field(gt=0)

    # wav2vec2 contrastive objective (xlsr trainer): quantizer / Gumbel settings
    diversity_loss_weight: float | None = Field(default=None, ge=0)
    max_gumbel_temperature: float | None = Field(default=None, gt=0)
    min_gumbel_temperature: float | None = Field(default=None, gt=0)
    gumbel_temperature_decay: float | None = Field(default=None, gt=0, le=1)

    # WavLM utterance mixing: probability that an utterance gets another one from its
    # batch mixed in (0 disables it; WavLM uses 0.2). A data augmentation of the
    # method, not a training hyperparameter.
    utterance_mix_prob: float = Field(default=0.0, ge=0, le=1)

    # masked-prediction objective (hubert / wavlm trainers): per-frame cluster ids
    labels_path: str | None = None

    save_every_updates: int = Field(gt=0)
    eval_every_updates: int = Field(gt=0)
    keep_last_n_checkpoints: int = Field(gt=0)
    milestone_checkpoints: list[int]

    output_dir: str

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
    """Training-health checks. Optional in params.yaml (defaults apply)."""

    # A codebook-perplexity reading below this counts as "low".
    perplexity_floor: float = Field(default=10.0, gt=0)
    # This many low readings in a row stop training. One reading is taken every
    # pretrain.eval_every_updates updates, so the default is 5 x 500 = 2,500 updates.
    consecutive_alerts: int = Field(default=5, gt=0)


class Params(BaseModel):
    select: SelectConfig
    shard: ShardConfig
    pretrain: PretrainConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cluster: ClusterConfig | None = None
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

    @model_validator(mode="after")
    def trainer_has_its_settings(self):
        """Each trainer needs the settings of its own objective; the ones it does not
        use are optional, so a config only has to spell out what its model uses."""
        pt = self.pretrain
        if self.model.trainer == "xlsr":
            missing = [
                name
                for name in (
                    "diversity_loss_weight",
                    "max_gumbel_temperature",
                    "min_gumbel_temperature",
                    "gumbel_temperature_decay",
                )
                if getattr(pt, name) is None
            ]
            if missing:
                raise ValueError(f"the xlsr trainer needs pretrain.{', pretrain.'.join(missing)}")
        else:
            if self.cluster is None:
                raise ValueError(f"the {self.model.trainer} trainer needs a `cluster` section")
            if not pt.labels_path:
                raise ValueError(f"the {self.model.trainer} trainer needs pretrain.labels_path")
        return self

    @classmethod
    def from_yaml(cls, path: str = "params.yaml") -> "Params":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
