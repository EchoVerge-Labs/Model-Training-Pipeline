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


class ClusterConfig(BaseModel):
    layer: int = Field(gt=0)  # transformer layer of the base model whose output is clustered
    num_clusters: int = Field(gt=0)
    seed: int
    sample_hours: float = Field(gt=0)  # audio streamed for the k-means fit
    max_frames: int = Field(gt=0)  # cap on frames kept for the fit
    kmeans_output: str
    labels_output: str


class LanguageBatchingConfig(BaseModel):
    """Language-homogeneous optimizer updates (see pipeline.language_batching).

    Off by default, which leaves the original mixed-language batching untouched.
    When on, every optimizer update -- all of its gradient-accumulation
    micro-batches, on every DDP rank -- is built from ONE language, chosen by a
    schedule that is a pure function of (seed, update index).
    """

    enabled: bool = False
    mode: Literal["balanced_random", "alternate"] = "balanced_random"
    # lower-case, as in the manifests' `language` field
    languages: list[str] = Field(default_factory=lambda: ["sinhala", "tamil"])
    probabilities: dict[str, float] | None = None  # default: uniform over `languages`
    # balanced_random: every block of `window` consecutive updates holds each
    # language in proportion to its probability (50/50 at 0.5/0.5), in random order.
    window: int = Field(default=20, gt=0)
    seed: int = 1337
    # These two are the point of the experiment; they exist as settings so the
    # config documents them, but turning either off is refused rather than
    # silently producing mixed-language gradients.
    synchronize_across_ranks: bool = True
    keep_language_across_gradient_accumulation: bool = True
    # Optional guard: refuse to train unless data/manifests/train.jsonl has exactly
    # the same clip ids as this manifest -- a path, or "git:<rev>:<path>".
    reference_manifest: str | None = None
    dataset_check_report: str = "reports/language_batching_dataset_check.json"

    @field_validator("languages")
    @classmethod
    def _normalise_languages(cls, v):
        langs = [x.strip().lower() for x in v]
        if len(langs) < 2 or len(set(langs)) != len(langs):
            raise ValueError(f"languages needs at least two distinct entries, got {v}")
        return langs

    @model_validator(mode="after")
    def _consistent(self):
        if self.probabilities is None:
            self.probabilities = {lang: 1 / len(self.languages) for lang in self.languages}
        else:
            self.probabilities = {
                k.strip().lower(): float(v) for k, v in self.probabilities.items()
            }
        if set(self.probabilities) != set(self.languages):
            raise ValueError(
                f"probabilities keys {sorted(self.probabilities)} must match languages "
                f"{sorted(self.languages)}"
            )
        if any(p <= 0 for p in self.probabilities.values()) or not (
            0.999 <= sum(self.probabilities.values()) <= 1.001
        ):
            raise ValueError(f"probabilities must be positive and sum to 1: {self.probabilities}")
        if self.enabled and not (
            self.synchronize_across_ranks and self.keep_language_across_gradient_accumulation
        ):
            raise ValueError(
                "language_batching.enabled requires synchronize_across_ranks and "
                "keep_language_across_gradient_accumulation to be true -- without them an "
                "optimizer update can still mix languages"
            )
        return self


class ExperimentConfig(BaseModel):
    """Where a run writes its reports and what it is called. The defaults are the
    original joint run's, so params.yaml needs no such section."""

    reports_dir: str = "reports"
    mlflow_run_name: str | None = None  # default: the original xlsr300m-si-ta-<batch>s name


class PretrainConfig(BaseModel):
    base_model: str
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    freeze_feature_encoder: bool = True
    # Stock value of all three base models (XLS-R on main runs with 0.1). Layerdrop
    # leaves parameters unused in a step, so train.py turns on DDP's
    # find_unused_parameters when it's > 0.
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

    labels_path: str

    save_every_updates: int = Field(gt=0)
    eval_every_updates: int = Field(gt=0)
    keep_last_n_checkpoints: int = Field(gt=0)
    milestone_checkpoints: list[int]

    output_dir: str

    language_batching: LanguageBatchingConfig = Field(default_factory=LanguageBatchingConfig)

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


class Params(BaseModel):
    select: SelectConfig
    shard: ShardConfig
    cluster: ClusterConfig
    pretrain: PretrainConfig
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, path: str = "params.yaml", overlay: str | None = None) -> "Params":
        """Load params.yaml; `overlay` (a YAML holding only the keys a variant changes)
        is deep-merged on top -- see pipeline.variant."""
        from pipeline.variant import load_raw

        return cls(**load_raw(path, overlay))
