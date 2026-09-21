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


class PretrainConfig(BaseModel):
    base_model: str
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    freeze_feature_encoder: bool = True

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

    target_batch_seconds: float = Field(gt=0)
    per_device_max_seconds: float = Field(gt=0)
    num_workers: int = Field(ge=0)

    mask_time_prob: float = Field(ge=0, le=1)
    mask_time_length: int = Field(gt=0)

    diversity_loss_weight: float = Field(ge=0)
    max_gumbel_temperature: float = Field(gt=0)
    min_gumbel_temperature: float = Field(gt=0)
    gumbel_temperature_decay: float = Field(gt=0, le=1)

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

    @classmethod
    def from_yaml(cls, path: str = "params.yaml") -> "Params":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
