"""Pydantic schema for params.yaml validation."""

import re
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
    # GB10 memory is unified: whatever the CUDA allocator caches is taken from the
    # OS. Capping it makes a spike raise torch.OutOfMemoryError instead of starving
    # sshd and hanging the whole host.
    max_gpu_memory_fraction: float = Field(default=0.8, gt=0, le=1)

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


# pretrain settings a run may override: how long it trains, and when it saves/is judged.
SCHEDULE_KEYS = {"max_updates", "warmup_updates", "milestone_checkpoints", "save_every_updates"}


class ExperimentRun(BaseModel):
    """One single-language ablation run: the joint selection restricted to one language."""

    language: str
    # Nominal hours of that language in the joint selection (select.target_hours x its mix
    # share). The hours actually realised can fall short; selection_stats.json records them.
    target_hours: float = Field(gt=0)
    # Schedule length only -- the one thing an equal-exposure variant changes (e.g. half the
    # updates, so each language gets the passes over its audio the joint run gave it).
    # Anything else (LR, batch, masking, ...) is refused: it would confound the comparison.
    pretrain_overrides: dict[str, int | list[int]] = Field(default_factory=dict)

    @field_validator("pretrain_overrides")
    @classmethod
    def only_schedule_length(cls, v):
        bad = sorted(set(v) - SCHEDULE_KEYS)
        if bad:
            raise ValueError(f"pretrain_overrides may only set {sorted(SCHEDULE_KEYS)}, got {bad}")
        return v

    @field_validator("language")
    @classmethod
    def lowercase_language(cls, v):
        if v != v.strip().lower() or not v:
            raise ValueError(f"language must be lowercase, as in the catalog: {v!r}")
        return v


class ExperimentsConfig(BaseModel):
    """Single-language runs. Optional: without it the pipeline is the joint one."""

    backbone_tag: str  # prefix of every output name, e.g. xlsr300m / wavlm-large
    runs: dict[str, ExperimentRun]

    @field_validator("runs")
    @classmethod
    def run_names_are_path_safe(cls, v):
        for name in v:
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
                raise ValueError(f"experiment name {name!r} must match [a-z0-9][a-z0-9-]*")
        return v


class Params(BaseModel):
    select: SelectConfig
    shard: ShardConfig
    pretrain: PretrainConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    experiments: ExperimentsConfig | None = None

    @model_validator(mode="after")
    def experiments_slice_the_joint_selection(self):
        # A run keeps one language of the joint selection, so the language must be in
        # the mix and its nominal hours must be what the mix gives it.
        for name, run in (self.experiments.runs if self.experiments else {}).items():
            share = self.select.language_mix.get(run.language)
            if share is None:
                raise ValueError(
                    f"experiments.runs.{name}: language {run.language!r} is not in "
                    f"select.language_mix {sorted(self.select.language_mix)}"
                )
            expected = self.select.target_hours * share
            if abs(run.target_hours - expected) > 1e-6:
                raise ValueError(
                    f"experiments.runs.{name}: target_hours {run.target_hours} != "
                    f"select.target_hours x language_mix[{run.language}] = {expected}"
                )
            if run.pretrain_overrides:
                self.effective_pretrain(name)  # validates the combined schedule
        return self

    def effective_pretrain(self, name: str) -> "PretrainConfig":
        """The pretrain config of run `name`: the shared recipe plus its schedule overrides.
        Raises if the schedule is inconsistent (warmup, milestones, save interval)."""
        run = self.experiments.runs[name]
        cfg = self.pretrain.model_copy(update=run.pretrain_overrides)
        if cfg.warmup_updates >= cfg.max_updates:
            raise ValueError(f"{name}: warmup_updates must be below max_updates")
        late = [m for m in cfg.milestone_checkpoints if m > cfg.max_updates]
        if late:
            raise ValueError(f"{name}: milestone_checkpoints {late} exceed max_updates")
        off = [m for m in cfg.milestone_checkpoints if m % cfg.save_every_updates]
        if off:
            raise ValueError(
                f"{name}: milestones {off} are not multiples of save_every_updates "
                f"({cfg.save_every_updates}), so they would never be saved"
            )
        return cfg

    @classmethod
    def from_yaml(cls, path: str = "params.yaml") -> "Params":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
