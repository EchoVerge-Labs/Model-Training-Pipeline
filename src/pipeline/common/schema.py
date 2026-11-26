"""Pydantic schema for the params*.yaml files (one per model, all validated here).

The top-level `model_key` names the model and fixes its objective family:
masked prediction on k-means cluster ids (HuBERT family) or wav2vec 2.0
contrastive learning with a Gumbel quantiser. Fields that only one family uses are
optional here, and `Params` requires them exactly when the family does.
"""

import copy
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from pipeline.common.naming import (
    METHOD_TOKENS,
    changed_keys,
    check_tag,
    data_id,
    hours_str,
    languages_token,
    passes,
    run_name,
)
from pipeline.models.registry import MODEL_REGISTRY, ModelFamily
from pipeline.pecopt.config import PECoPTConfig


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
    # Derived from the data id (data/manifests/<data id>) unless set.
    manifest_dir: str | None = None

    @field_validator("language_mix")
    @classmethod
    def mix_sums_to_one(cls, v):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"language_mix must sum to 1.0, got {total}")
        return v

    @property
    def languages(self) -> list[str]:
        """The languages a run trains on: every language in the mix (joint si-ta runs)."""
        return [lang.strip().lower() for lang in self.language_mix]


class ShardConfig(BaseModel):
    format: Literal["flac", "wav"] = "flac"
    shard_size: int = Field(gt=0)
    min_cut_seconds: float = Field(gt=0)
    target_cut_seconds: float = Field(gt=0)
    max_cut_seconds: float = Field(gt=0)
    output_dir: str | None = None  # derived: data/shars/<data id>

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
    # The selection k-means is fitted on (a data id). Default: the run's own selection. A
    # small smoke selection can point at a bigger one's k-means (configs/variants/smoke-1h).
    kmeans_data: str | None = None
    # Derived unless set: models/kmeans/<model>-L<layer>-k<k>-<kmeans data>.joblib, and
    # data/labels/<same>[-on-<data id>].jsonl.gz for the run's own shards.
    kmeans_output: str | None = None
    labels_output: str | None = None


ModelKey = Literal["mhubert147", "hubert", "wavlm", "xlsr", "wav2vec2"]


class LanguageBatchingConfig(BaseModel):
    """Language-homogeneous optimizer updates (pipeline.training.language_batching).

    Off by default (mixed-language batches). When on, every optimizer update -- all of its
    gradient-accumulation micro-batches, on every DDP rank -- is built from ONE language,
    chosen by a schedule that is a pure function of (seed, update index). Runs with it on are
    named `...-homog`.
    """

    enabled: bool = False
    mode: Literal["balanced_random", "alternate"] = "balanced_random"
    # lower-case, as in the manifests' `language` field; default: the languages of language_mix
    languages: list[str] | None = None
    probabilities: dict[str, float] | None = None  # default: uniform over `languages`
    # balanced_random: every block of `window` consecutive updates holds each language in
    # proportion to its probability (50/50 at 0.5/0.5), in random order.
    window: int = Field(default=20, gt=0)
    seed: int = 1337

    def resolve(self, languages: list[str]) -> "LanguageBatchingConfig":
        """Fill in the languages (the run's) and uniform probabilities, and check them."""
        langs = [x.strip().lower() for x in (self.languages or languages)]
        if self.enabled and (len(langs) < 2 or len(set(langs)) != len(langs)):
            raise ValueError(
                f"language_batching needs at least two distinct languages, got {langs} "
                f"(a single-language run has nothing to batch by language)"
            )
        if self.enabled and set(langs) != set(languages):
            raise ValueError(
                f"language_batching.languages {langs} must be the run's languages {languages}"
            )
        probs = self.probabilities or {lang: 1 / len(langs) for lang in langs}
        probs = {k.strip().lower(): float(v) for k, v in probs.items()}
        if set(probs) != set(langs):
            raise ValueError(f"probabilities keys {sorted(probs)} must match languages {langs}")
        if any(v <= 0 for v in probs.values()) or not 0.999 <= sum(probs.values()) <= 1.001:
            raise ValueError(f"probabilities must be positive and sum to 1: {probs}")
        return self.model_copy(update={"languages": langs, "probabilities": probs})


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
    # distractors per masked position; default = the base model's own (100 for XLS-R)
    num_negatives: int | None = Field(default=None, gt=0)
    max_gumbel_temperature: float | None = Field(default=None, gt=0)
    min_gumbel_temperature: float | None = Field(default=None, gt=0)
    gumbel_temperature_decay: float | None = Field(default=None, gt=0, le=1)

    save_every_updates: int = Field(gt=0)
    eval_every_updates: int = Field(gt=0)
    keep_last_n_checkpoints: int = Field(gt=0)
    milestone_checkpoints: list[int]

    output_dir: str | None = None  # derived: models/<run name>

    language_batching: LanguageBatchingConfig = Field(default_factory=LanguageBatchingConfig)

    # Training method: full_copt (default), or a PE-CoPT template (not implemented yet)
    pecopt: PECoPTConfig = Field(default_factory=PECoPTConfig)

    @model_validator(mode="after")
    def schedule_is_consistent(self):
        """Catch a schedule that would silently never produce its checkpoints."""
        if self.warmup_updates >= self.max_updates:
            raise ValueError(
                f"warmup_updates ({self.warmup_updates}) must be below max_updates "
                f"({self.max_updates})"
            )
        for m in self.milestone_checkpoints:
            if m > self.max_updates:
                raise ValueError(
                    f"milestone checkpoint {m} is beyond max_updates ({self.max_updates}), "
                    "so it would never be written"
                )
            if m % self.save_every_updates:
                raise ValueError(
                    f"milestone checkpoint {m} is not a multiple of save_every_updates "
                    f"({self.save_every_updates}); checkpoints are only written on those steps"
                )
        lo, hi = self.min_gumbel_temperature, self.max_gumbel_temperature
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"min_gumbel_temperature ({lo}) must not exceed max_gumbel_temperature ({hi})"
            )
        return self

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


DATA_CONFIG = "params-data.yaml"

# Settings a run's name shows: changing one renames the run, so an overlay may change them
# without a tag. (warmup_updates counts when it keeps its share of max_updates.)
NAMED_KEYS = {
    "select.target_hours",
    "pretrain.freeze_feature_encoder",
    "pretrain.max_updates",
    "pretrain.pecopt.method",
    "pretrain.language_batching.enabled",
}
# Settings that do not change what is trained: where things are written, how often the run
# logs and saves, and the names themselves.
BOOKKEEPING_PREFIXES = (
    "tag",
    "run_name",
    "data_tag",
    "monitoring",
    "select.manifest_dir",
    "select.catalog_path",
    "select.drive_root",
    "select.drive_index_path",
    "shard.output_dir",
    "shard.format",
    "cluster.kmeans_output",
    "cluster.labels_output",
    "pretrain.output_dir",
    "pretrain.labels_path",
    "pretrain.num_workers",
    "pretrain.max_gpu_memory_fraction",
    "pretrain.eval_every_updates",
    "pretrain.save_every_updates",
    "pretrain.keep_last_n_checkpoints",
    "pretrain.milestone_checkpoints",
    "pretrain.language_batching.languages",  # resolved from select.language_mix
    "pretrain.language_batching.probabilities",  # resolved: uniform by default
)


def _read_yaml(path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    """overlay's keys win; nested dicts merge recursively. Returns a new dict."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _with_data_config(raw: dict, path: str) -> dict:
    """select/shard are shared by every model and live in params-data.yaml (`data_config:`
    names another file); a file that has its own sections keeps them."""
    if "select" in raw and "shard" in raw:
        return raw
    data_file = Path(path).parent / raw.get("data_config", DATA_CONFIG)
    if not data_file.exists():
        data_file = Path(raw.get("data_config", DATA_CONFIG))
    data = _read_yaml(data_file)
    raw = dict(raw)
    for key in ("select", "shard"):
        raw.setdefault(key, data[key])
    return raw


def unnamed_changes(base, merged) -> list[str]:
    """Settings an overlay changed that the run's name does not show (they need a tag)."""
    before, after = base.model_dump(), merged.model_dump()
    out = []
    for key in changed_keys(before, after):
        if key in NAMED_KEYS or key.startswith(BOOKKEEPING_PREFIXES):
            continue
        if key == "pretrain.warmup_updates" and "max_updates" in (before.get("pretrain") or {}):
            # scaled with max_updates (the same warmup share): part of the passes change
            b, a = before["pretrain"], after["pretrain"]
            share_before = b["warmup_updates"] / b["max_updates"]
            share_after = a["warmup_updates"] / a["max_updates"]
            if abs(share_after - share_before) <= 0.1 * share_before:
                continue
        out.append(key)
    return out


def _load(path: str, variants, tag: str | None, model) -> tuple[dict, BaseModel]:
    """Read `path` (+ data config), merge the variant overlays, and validate with `model`.

    An overlay may change any setting, but one the name does not show (see NAMED_KEYS)
    needs a tag, or two different runs would share a name. Such a change to the data
    (select/shard) also tags the data id, so the selection gets its own folder.
    """
    base_raw = _read_yaml(path)
    if model is Params:
        base_raw = _with_data_config(base_raw, path)
    raw = base_raw
    for variant in variants or ():
        raw = deep_merge(raw, _read_yaml(variant))
    tag = check_tag(tag or raw.get("tag"))
    changes = []
    if variants:
        changes = unnamed_changes(model(**base_raw), model(**{**raw, "tag": tag}))
        if changes and not tag:
            raise ValueError(
                f"the variant changes {changes}, which the run name does not show -- give the "
                f"run a tag (TAG=... / --tag / `tag:` in the overlay) so it gets its own name"
            )
    data_tag = tag if any(k.startswith(("select.", "shard.")) for k in changes) else None
    raw = {**raw, "tag": tag, "data_tag": data_tag}
    return raw, model(**raw)


def _model_paths(model_key: str, cluster, data_id_: str) -> tuple[str, str]:
    kmeans_data = cluster.kmeans_data or data_id_
    stem = f"{MODEL_REGISTRY[model_key].slsb_name}-L{cluster.layer}-k{cluster.num_clusters}"
    kmeans = cluster.kmeans_output or f"models/kmeans/{stem}-{kmeans_data}.joblib"
    on = "" if kmeans_data == data_id_ else f"-on-{data_id_}"
    labels = cluster.labels_output or f"data/labels/{stem}-{kmeans_data}{on}.jsonl.gz"
    return kmeans, labels


class DataParams(BaseModel):
    """params-data.yaml (+ variant overlays): what the data stages read."""

    select: SelectConfig
    shard: ShardConfig
    tag: str | None = None
    data_tag: str | None = None  # set by the loader when a variant changes the data

    @model_validator(mode="after")
    def data_paths(self):
        if self.select.manifest_dir is None:
            self.select.manifest_dir = f"data/manifests/{self.data_id}"
        if self.shard.output_dir is None:
            self.shard.output_dir = f"data/shars/{self.data_id}"
        return self

    @property
    def data_id(self) -> str:
        return data_id(self.select.languages, self.select.target_hours, self.data_tag)

    @property
    def reports_dir(self) -> Path:
        return Path("reports/data") / self.data_id

    @classmethod
    def from_yaml(cls, path: str = DATA_CONFIG, variants=(), tag: str | None = None):
        return _load(path, variants, tag, cls)[1]


class Params(BaseModel):
    model_key: ModelKey
    tag: str | None = None
    data_tag: str | None = None  # set by the loader when a variant changes the data
    run_name: str | None = None  # overrides the derived name (e.g. to reach an older folder)
    select: SelectConfig
    shard: ShardConfig
    cluster: ClusterConfig | None = None
    pretrain: PretrainConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="before")
    @classmethod
    def contrastive_models_have_no_kmeans(cls, data):
        """A shared overlay (e.g. smoke-1h) may set cluster keys for the HuBERT family; the
        contrastive models have no k-means, so the section does not apply to them."""
        if isinstance(data, dict) and data.get("cluster") is not None:
            key = data.get("model_key")
            if key in MODEL_REGISTRY and MODEL_REGISTRY[key].family == "contrastive":
                data = {**data, "cluster": None}
        return data

    @model_validator(mode="after")
    def derive_names_and_paths(self):
        check_tag(self.tag)
        self.pretrain.language_batching = self.pretrain.language_batching.resolve(
            self.select.languages
        )
        if self.select.manifest_dir is None:
            self.select.manifest_dir = f"data/manifests/{self.data_id}"
        if self.shard.output_dir is None:
            self.shard.output_dir = f"data/shars/{self.data_id}"
        if self.pretrain.output_dir is None:
            self.pretrain.output_dir = f"models/{self.run}"
        if self.cluster is not None:
            kmeans, labels = _model_paths(self.model_key, self.cluster, self.data_id)
            self.cluster.kmeans_output = self.cluster.kmeans_output or kmeans
            self.cluster.labels_output = self.cluster.labels_output or labels
            if self.pretrain.labels_path is None:
                self.pretrain.labels_path = self.cluster.labels_output
        return self

    # ── names ──

    @property
    def model_family(self) -> ModelFamily:
        return MODEL_REGISTRY[self.model_key].family

    @property
    def hours(self) -> float:
        """Hours of audio the run trains on."""
        return self.select.target_hours

    @property
    def passes(self) -> float:
        p = self.pretrain
        return passes(p.max_updates, p.target_batch_seconds, self.hours)

    @property
    def data_id(self) -> str:
        return data_id(self.select.languages, self.hours, self.data_tag)

    @property
    def run(self) -> str:
        """The run's one name: its folders, its ssl-pretraining run and its leaderboard row."""
        return self.run_name or run_name(
            MODEL_REGISTRY[self.model_key].slsb_name,
            self.pretrain.pecopt.method,
            self.select.languages,
            self.hours,
            cnn_trains=not self.pretrain.freeze_feature_encoder,
            n_passes=self.passes,
            homogeneous=self.pretrain.language_batching.enabled,
            tag=self.tag,
        )

    @property
    def name_parts(self) -> dict[str, str]:
        """The name's slots, as DagsHub tags on the training run and the leaderboard row."""
        return {
            "model": MODEL_REGISTRY[self.model_key].slsb_name,
            "method": METHOD_TOKENS[self.pretrain.pecopt.method],
            "languages": languages_token(self.select.languages),
            "hours": hours_str(self.hours),
            "cnn": "trained" if not self.pretrain.freeze_feature_encoder else "frozen",
            "passes": f"{round(self.passes, 1):g}",
            "batching": "homogeneous" if self.pretrain.language_batching.enabled else "mixed",
            "data": self.data_id,
            **({"tag": self.tag} if self.tag else {}),
        }

    # ── where the run's files go ──

    @property
    def kmeans_data_id(self) -> str:
        """The selection the k-means is fitted on (default: the run's own)."""
        return self.cluster.kmeans_data or self.data_id

    @property
    def kmeans_stats_path(self) -> Path:
        return (
            Path("reports/data")
            / self.kmeans_data_id
            / f"kmeans-{Path(self.cluster.kmeans_output).stem}.json"
        )

    @property
    def model_dir(self) -> Path:
        return Path(self.pretrain.output_dir)

    @property
    def selected_dir(self) -> Path:
        return self.model_dir / "selected"

    @property
    def reports_dir(self) -> Path:
        return Path("reports") / self.run

    @property
    def logs_dir(self) -> Path:
        return Path("logs") / self.run

    @property
    def bench_dir(self) -> Path:
        return self.reports_dir / "bench"

    @property
    def training_config(self) -> dict:
        """What the run trains on and how -- the part of the config its name stands for
        (paths and logging cadence left out). run.json stores its fingerprint."""
        dump = self.model_dump()
        for key in ("tag", "data_tag", "run_name", "monitoring"):
            dump.pop(key, None)
        for section, keys in (
            ("select", ("manifest_dir", "catalog_path", "drive_root", "drive_index_path")),
            ("shard", ("output_dir",)),
            ("cluster", ("kmeans_output", "labels_output")),
            (
                "pretrain",
                (
                    "output_dir",
                    "labels_path",
                    "num_workers",
                    "max_gpu_memory_fraction",
                    "eval_every_updates",
                    "save_every_updates",
                    "keep_last_n_checkpoints",
                    "milestone_checkpoints",
                ),
            ),
        ):
            for key in keys:
                (dump.get(section) or {}).pop(key, None)
        return dump

    @model_validator(mode="after")
    def family_fields_are_present(self):
        if self.model_family == "masked_prediction":
            if self.cluster is None:
                raise ValueError(f"model_key={self.model_key!r} needs a `cluster` section")
        else:
            missing = [f for f in _CONTRASTIVE_FIELDS if getattr(self.pretrain, f) is None]
            if missing:
                raise ValueError(
                    f"model_key={self.model_key!r} needs pretrain.{', pretrain.'.join(missing)}"
                )
        return self

    @classmethod
    def from_yaml(
        cls, path: str = "params-mhubert147.yaml", variants=(), tag: str | None = None
    ) -> "Params":
        return load_params(path, variants, tag)[1]


def load_params(path: str, variants=(), tag: str | None = None) -> tuple[dict, Params]:
    """(raw dict, Params) for a model params file plus variant overlays and a tag. The raw
    dict carries the sections Params does not model (checkpoint_selection, benchmark,
    mlflow) and the derived run name and data id."""
    raw, params = _load(path, variants, tag, Params)
    raw = {
        **raw,
        "run": params.run,  # derived; `run_name` stays the (optional) manual override
        "data_id": params.data_id,
    }
    return raw, params


def load_raw_params(path: str = "params-mhubert147.yaml", variants=(), tag=None) -> dict:
    return load_params(path, variants, tag)[0]


def add_run_args(parser) -> None:
    """--config / --variant / --tag: how every stage names the run it works on."""
    parser.add_argument("--config", default="params-mhubert147.yaml")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="overlay YAML deep-merged over --config (repeatable; e.g. configs/variants/cnn.yaml)",
    )
    parser.add_argument("--tag", default=None, help="free-text name suffix, e.g. smoke")


def variants_arg(values) -> list[str]:
    """--variant may be repeated or comma-separated (make passes VARIANT=a.yaml,b.yaml)."""
    return [v for value in values or [] for v in value.split(",") if v]


def add_data_args(parser) -> None:
    """--config / --variant / --tag for the data stages (select, pull, shard)."""
    parser.add_argument("--config", default=DATA_CONFIG)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="overlay YAML (repeatable); its select/shard keys decide the selection",
    )
    parser.add_argument("--tag", default=None, help="the run's tag (names a changed selection)")
