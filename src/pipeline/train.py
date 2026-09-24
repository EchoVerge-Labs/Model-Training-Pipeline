"""Continued pre-training of wav2vec2-xls-r-300m on Sinhala/Tamil.

Adapted from HuggingFace's run_wav2vec2_pretraining_no_trainer.py.
Key changes from the HF example:
  - Loads from Lhotse Shar tarballs (not HF datasets)
  - Uses raw DDP + torch.amp (not HF Accelerator)
  - Tri-stage LR schedule (warmup / hold / exponential decay)
  - MLflow logging to DagsHub
  - Codebook collapse detection
  - Asserts quantizer weights are present in checkpoint
  - Reads all config from params.yaml via Pydantic schema

Mask generation and negative sampling reuse transformers' own
`_compute_mask_indices` / `_sample_negative_indices` (the same helpers the HF
reference script's DataCollator uses) rather than a hand-rolled version:
`Wav2Vec2ForPreTraining.forward()` returns `loss=None` unless
`sampled_negative_indices` is supplied (see its forward() docstring --
"Required input for pre-training"), so negative sampling isn't optional.

Resuming: checkpoints (model + optimizer + step) are written every
pretrain.save_every_updates. Relaunching the SAME command after a crash resumes
automatically from the newest complete checkpoint in pretrain.output_dir; pass
--resume-from CKPT_DIR to pick a specific one, or --no-resume to start over.

Usage:
  # Single node
  torchrun --nproc_per_node=1 -m pipeline.train --config params.yaml

  # Two nodes
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
           --rdzv_backend=c10d --rdzv_endpoint=10.0.0.1:29500 \
           -m pipeline.train --config params.yaml
"""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

# Must be set before torch initialises CUDA. Batches vary in shape every step, and
# with the default allocator the cache fragments and keeps growing (measured: 46GB
# live -> 75GB+ reserved within ~20 micro-batches). On GB10's unified memory that
# growth eats system RAM until the host locks up. Expandable segments hold it flat.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)

from pipeline.callbacks import CodebookCollapseDetector, MLflowLogger
from pipeline.datamodule import create_dataloader
from pipeline.schema import Params

# ─── Tri-stage LR schedule ──────────────────────────────────────────────
#
# Phase 1: linear warmup from 0 → peak_lr         (0 → warmup_updates)
# Phase 2: hold at peak_lr                        (warmup_updates → hold_end)
# Phase 3: exponential decay from peak_lr → 0     (hold_end → max_updates)
#
# hold_end = warmup_updates + hold_ratio * (max_updates - warmup_updates)
#
# This is the schedule fairseq uses for wav2vec2 pre-training, not a cosine.


def get_tri_stage_lr(
    step: int,
    peak_lr: float,
    warmup_updates: int,
    hold_ratio: float,
    max_updates: int,
) -> float:
    """Compute learning rate at a given step using the tri-stage schedule."""
    hold_updates = int(hold_ratio * (max_updates - warmup_updates))
    hold_end = warmup_updates + hold_updates
    decay_updates = max_updates - hold_end

    if step < warmup_updates:
        # Phase 1: linear warmup
        if warmup_updates == 0:
            return peak_lr
        return peak_lr * step / warmup_updates

    elif step < hold_end:
        # Phase 2: hold
        return peak_lr

    else:
        # Phase 3: exponential decay
        if decay_updates == 0:
            return peak_lr
        decay_step = step - hold_end
        # Exponential decay to ~0 at max_updates
        # lr = peak_lr * gamma^decay_step, where gamma chosen so final lr ≈ peak_lr * 0.01
        gamma = (0.01) ** (1.0 / max(decay_updates, 1))
        return peak_lr * (gamma**decay_step)


# ─── Mask generation + negative sampling ────────────────────────────────
#
# Reuses transformers' own `_compute_mask_indices` / `_sample_negative_indices`
# (numpy-based) exactly as HuggingFace's DataCollatorForWav2Vec2Pretraining
# does. Both are computed against the *feature-extractor output* length (the
# transformer's input length after the CNN downsamples), using the reduced
# attention mask so padded frames are never masked/sampled.


def compute_mask_and_negatives(
    model,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float,
    mask_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (mask_time_indices, sampled_negative_indices, sub_attention_mask),
    all as torch tensors on `device`."""
    batch_size = input_values.shape[0]
    seq_length = int(model._get_feat_extract_output_lengths(input_values.shape[-1]))

    sub_attention_mask = model._get_feature_vector_attention_mask(seq_length, attention_mask)

    features_shape = (batch_size, seq_length)
    mask_time_indices = _compute_mask_indices(
        features_shape,
        mask_prob=mask_prob,
        mask_length=mask_length,
        attention_mask=sub_attention_mask,
        min_masks=2,
    )
    sampled_negative_indices = _sample_negative_indices(
        features_shape,
        model.config.num_negatives,
        mask_time_indices=mask_time_indices,
    )

    mask_time_indices = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
    sampled_negative_indices = torch.tensor(
        sampled_negative_indices, dtype=torch.long, device=device
    )
    return mask_time_indices, sampled_negative_indices, sub_attention_mask


# ─── Gumbel temperature ────────────────────────────────────────────────


def get_gumbel_temperature(
    step: int,
    max_temp: float,
    min_temp: float,
    decay: float,
) -> float:
    """Exponential decay of Gumbel softmax temperature.

    For continued pre-training from a checkpoint with a trained codebook,
    start at a LOWER max_temp (1.0 vs 2.0) because the codebook is already
    meaningful — high temperature would noise out the existing structure.
    """
    return max(min_temp, max_temp * (decay**step))


# ─── Gradient rescaling by mask count ───────────────────────────────────
#
# Wav2Vec2ForPreTraining's contrastive_loss (reduction="sum") and
# diversity_loss (scaled by mask_time_indices.sum()) are UNNORMALIZED sums
# over masked positions -- dividing only by grad_accum_steps would let
# gradient magnitude swing with batch size / mask density between steps.
# The HF reference rescales each micro-batch's just-computed gradient by
# 1/num_losses (that micro-batch's masked-position count) right after its
# backward() call. Under DDP, gradients are already averaged across ranks by
# DDP's backward-time all-reduce, so the multiplier becomes
# world_size/total_num_losses instead of 1/num_losses to compensate.


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


# ─── Checkpointing and resume ──────────────────────────────────────────
#
# A checkpoint is a directory checkpoint-<step>/ holding the model
# (save_pretrained), the feature extractor, and training_state.pt (optimizer
# state, global_step, Gumbel temperature). Everything else the loop needs -- the
# LR, the Gumbel temperature, the data-loader seed -- is a pure function of
# global_step, so restoring the optimizer and the step is all a resume takes.

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
TRAINING_STATE_FILE = "training_state.pt"
CURVES_HEADER = (
    "step,loss,contrastive_loss,diversity_loss,codebook_perplexity,"
    "grad_norm,lr,gumbel_temp,audio_sec_per_wall_sec"
)


def checkpoint_step(path: Path) -> int | None:
    """The step in a checkpoint-<step> directory name, or None if it isn't one."""
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def _is_complete_checkpoint(path: Path) -> bool:
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and has_weights
        and (path / TRAINING_STATE_FILE).exists()
    )


def list_checkpoints(output_dir: Path) -> list[Path]:
    """Complete checkpoint directories under output_dir, newest (highest step) first."""
    found = [
        (step, path)
        for path in Path(output_dir).glob("checkpoint-*")
        if (step := checkpoint_step(path)) is not None and _is_complete_checkpoint(path)
    ]
    return [path for _, path in sorted(found, reverse=True)]


def save_checkpoint(
    raw_model, feature_extractor, optimizer, global_step: int, gumbel_temp: float, output_dir: Path
) -> Path:
    """Write checkpoint-<global_step>/ atomically.

    Everything is built under a hidden temp name and renamed into place only
    when complete, so a crash mid-save (a checkpoint is several GB) can never
    leave a directory that looks like a valid checkpoint.
    """
    output_dir = Path(output_dir)
    final = output_dir / f"checkpoint-{global_step}"
    tmp = output_dir / f".tmp-checkpoint-{global_step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    raw_model.save_pretrained(tmp)
    feature_extractor.save_pretrained(tmp)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "gumbel_temperature": gumbel_temp,
        },
        tmp / TRAINING_STATE_FILE,
    )

    if final.exists():
        shutil.rmtree(final)
    os.rename(tmp, final)
    return final


def load_training_state(ckpt_dir: Path, device) -> dict:
    """Read training_state.pt. weights_only: it holds tensors and plain numbers, so
    nothing needs to be unpickled as code."""
    state = torch.load(Path(ckpt_dir) / TRAINING_STATE_FILE, map_location=device, weights_only=True)
    missing = {"optimizer", "global_step"} - set(state)
    if missing:
        raise ValueError(f"{ckpt_dir / TRAINING_STATE_FILE} is missing {sorted(missing)}")
    return state


def find_resume(
    resume_from: str | Path | None, output_dir: Path, device, auto_resume: bool = True
) -> tuple[Path, dict] | None:
    """Decide what to resume from. Returns (checkpoint dir, training state) or None.

    An explicit resume_from must be a complete, readable checkpoint (error
    otherwise). Otherwise, with auto_resume, the newest complete checkpoint in
    output_dir is used -- and if it turns out to be unreadable (e.g. a crash
    truncated it) the next-newest is tried, so one bad file can't block a relaunch.
    """
    if resume_from is not None:
        path = Path(resume_from)
        if not _is_complete_checkpoint(path):
            raise FileNotFoundError(
                f"--resume-from {path}: not a complete checkpoint "
                f"(needs config.json, model weights and {TRAINING_STATE_FILE})"
            )
        return path, load_training_state(path, device)
    if not auto_resume:
        return None
    for path in list_checkpoints(output_dir):
        try:
            return path, load_training_state(path, device)
        except Exception as e:  # noqa: BLE001 -- any unreadable file just means "try the previous one"
            print(f"WARNING: skipping unreadable checkpoint {path}: {e}")
    return None


def prune_checkpoints(output_dir: Path, milestones: list[int], keep_last_n: int) -> None:
    """Keep every milestone checkpoint, plus the newest keep_last_n of the others."""
    numbered = sorted(
        (step, path)
        for path in Path(output_dir).glob("checkpoint-*")
        if (step := checkpoint_step(path)) is not None
    )
    rotating = [path for step, path in numbered if step not in milestones]
    for old in rotating[: max(0, len(rotating) - keep_last_n)]:
        print(f"  Removing old checkpoint: {old.name}")
        shutil.rmtree(old)


def prepare_curves(path: Path, resume_step: int) -> None:
    """Fresh run: start the loss-curve CSV. Resume: keep the rows up to the step
    being resumed from and drop later ones, which get re-logged when those steps
    re-run -- so the history has no gaps and no duplicates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if resume_step > 0 and path.exists() else []
    if not lines:
        path.write_text(CURVES_HEADER + "\n")
        return
    kept = [row for row in lines[1:] if row and int(row.split(",", 1)[0]) <= resume_step]
    path.write_text("\n".join([lines[0], *kept]) + "\n")


def _assert_ranks_agree_on_resume(local_step: int, device) -> None:
    """A checkpoint only exists on the node that saved it. If ranks resumed from
    different steps their optimizer states would silently diverge, so fail every
    rank together (a one-sided error would leave the others hanging)."""
    reference = torch.tensor([local_step], device=device)
    dist.broadcast(reference, src=0)
    disagree = torch.tensor([int(local_step != int(reference.item()))], device=device)
    dist.all_reduce(disagree, op=dist.ReduceOp.MAX)
    if disagree.item():
        raise RuntimeError(
            f"ranks disagree on the resume checkpoint (this rank: step {local_step}, rank 0: "
            f"step {int(reference.item())}). Copy the checkpoint directory to every node, or "
            f"pass --no-resume to start fresh."
        )


# ─── Training ──────────────────────────────────────────────────────────


def train(
    params: Params,
    *,
    config_path: str = "params.yaml",
    resume_from: str | Path | None = None,
    auto_resume: bool = True,
    dataloader=None,
    reports_dir: str | Path = "reports",
    use_mlflow: bool = True,
    device: str | None = None,
) -> int:
    """Run (or resume) pre-training. Returns the global step reached.

    resume_from: restart from this checkpoint directory.
    auto_resume: with no resume_from, restart from the newest complete
        checkpoint in pretrain.output_dir if there is one (so a crash followed by
        the same launch command just continues).
    dataloader / device / reports_dir / use_mlflow: seams for tests; in a real run
        they default to the shard loader, CUDA if present, ./reports, and MLflow.
    """
    cfg = params.pretrain
    reports_dir = Path(reports_dir)

    # ── Distributed setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.max_gpu_memory_fraction, device)

    # ── Output dir + resume ──
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        for stale in output_dir.glob(".tmp-checkpoint-*"):  # a save that never finished
            shutil.rmtree(stale, ignore_errors=True)

    resume = find_resume(resume_from, output_dir, device, auto_resume)
    resume_step = int(resume[1]["global_step"]) if resume else 0
    if world_size > 1:
        _assert_ranks_agree_on_resume(resume_step if resume else -1, device)

    if resume and resume_step >= cfg.max_updates:
        if is_main:
            print(
                f"{resume[0].name} is at step {resume_step}, which already reaches "
                f"max_updates ({cfg.max_updates}): nothing to train. "
                f"(Pass --no-resume to start over.)"
            )
        if world_size > 1:
            dist.destroy_process_group()
        return resume_step

    model_source = str(resume[0]) if resume else cfg.base_model

    if is_main:
        print(f"World size: {world_size}, device: {device}")
        print(f"Base model: {cfg.base_model}")
        if resume:
            how = "explicitly" if resume_from is not None else "automatically"
            print(f"Resuming {how} from {resume[0]} at step {resume_step}")
        else:
            print("Starting from the base model (no checkpoint resumed)")
        print(f"Precision: {cfg.precision}")
        print(f"Max updates: {cfg.max_updates}")
        print(f"Peak LR: {cfg.peak_lr}")
        print(f"Target batch seconds: {cfg.target_batch_seconds}")

    # ── Load model ──
    model = Wav2Vec2ForPreTraining.from_pretrained(model_source)

    # CRITICAL: verify quantizer weights are present
    state_keys = set(model.state_dict().keys())
    quantizer_keys = [k for k in state_keys if "quantizer" in k]
    project_q_keys = [k for k in state_keys if "project_q" in k]
    project_hid_keys = [k for k in state_keys if "project_hid" in k]

    if is_main:
        print(f"Quantizer keys: {len(quantizer_keys)}")
        print(f"project_q keys: {len(project_q_keys)}")
        print(f"project_hid keys: {len(project_hid_keys)}")

    if not quantizer_keys:
        raise RuntimeError(
            "FATAL: No quantizer weights found in checkpoint. "
            "This checkpoint cannot be used for continued pre-training. "
            "Make sure you are loading Wav2Vec2ForPreTraining, not Wav2Vec2Model. "
            f"Loaded from: {model_source}"
        )

    # Freeze CNN feature encoder — standard for continued pre-training
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()
        if is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable "
                f"(CNN frozen)"
            )

    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Unwrap for accessing config/methods not proxied by DDP (mask/negative
    # sampling helpers, save_pretrained, set_gumbel_temperature)
    raw_model = model.module if hasattr(model, "module") else model

    # ── Feature extractor (for computing output lengths) ──
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.base_model)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.peak_lr,
        betas=tuple(cfg.adam_betas),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    if resume:
        try:
            optimizer.load_state_dict(resume[1]["optimizer"])
        except ValueError as e:
            raise RuntimeError(
                f"cannot restore the optimizer from {resume[0]}: {e} -- do "
                f"freeze_feature_encoder or the model differ from when it was saved?"
            ) from e
        if is_main:
            saved_temp = resume[1].get("gumbel_temperature")
            print(f"Restored optimizer state (saved Gumbel temperature: {saved_temp})")

    # ── Data ──
    # Seeded with the resume step, so a resumed run doesn't replay the exact
    # shard order of the first epoch (a fresh run has step 0, i.e. seed 0 as before).
    if dataloader is None:
        dataloader = create_dataloader(
            shar_dir="data/shars",
            per_device_max_seconds=cfg.per_device_max_seconds,
            num_workers=cfg.num_workers,
            seed=resume_step,
        )
    data_iter = iter(dataloader)

    # ── Gradient accumulation ──
    # target_batch_seconds / (per_device_max_seconds * world_size)
    grad_accum_steps = max(
        1, int(cfg.target_batch_seconds / (cfg.per_device_max_seconds * world_size))
    )
    if is_main:
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(
            f"Effective batch: ~{cfg.per_device_max_seconds * world_size * grad_accum_steps:.0f}s "
            f"audio per update"
        )

    # ── Logging ──
    mlflow_logger = None
    collapse_detector = None
    if is_main:
        if use_mlflow:
            try:
                with open(config_path) as f:
                    mlflow_cfg = yaml.safe_load(f).get("mlflow", {})
                run_name = f"xlsr300m-si-ta-{cfg.target_batch_seconds}s"
                if resume:
                    run_name += f"-resume{resume_step}"
                mlflow_logger = MLflowLogger(
                    tracking_uri=mlflow_cfg.get("tracking_uri", ""),
                    experiment_name=mlflow_cfg.get("experiment_name", "ssl-pretraining"),
                    run_name=run_name,
                )
                # Log all params
                flat_params = {
                    "base_model": cfg.base_model,
                    "precision": cfg.precision,
                    "max_updates": cfg.max_updates,
                    "warmup_updates": cfg.warmup_updates,
                    "peak_lr": cfg.peak_lr,
                    "lr_schedule": cfg.lr_schedule,
                    "hold_ratio": cfg.hold_ratio,
                    "target_batch_seconds": cfg.target_batch_seconds,
                    "per_device_max_seconds": cfg.per_device_max_seconds,
                    "grad_accum_steps": grad_accum_steps,
                    "world_size": world_size,
                    "mask_time_prob": cfg.mask_time_prob,
                    "mask_time_length": cfg.mask_time_length,
                    "diversity_loss_weight": cfg.diversity_loss_weight,
                    "max_gumbel_temperature": cfg.max_gumbel_temperature,
                    "min_gumbel_temperature": cfg.min_gumbel_temperature,
                    "gumbel_temperature_decay": cfg.gumbel_temperature_decay,
                    "freeze_feature_encoder": cfg.freeze_feature_encoder,
                    "resumed_from_step": resume_step,
                }
                mlflow_logger.log_params(flat_params)
            except Exception as e:  # noqa: BLE001 -- experiment tracking must never kill a training run
                print(f"WARNING: MLflow setup failed: {e}")
                mlflow_logger = None

        collapse_detector = CodebookCollapseDetector(
            num_codebooks=raw_model.config.num_codevector_groups,
            alert_threshold=params.monitoring.perplexity_floor,
            consecutive_alerts=params.monitoring.consecutive_alerts,
        )

    # ── Precision ──
    use_amp = cfg.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16

    # ── Training loop ──
    model.train()
    global_step = resume_step
    accum_loss = 0.0
    accum_contrastive_loss = 0.0
    accum_diversity_loss = 0.0
    accum_grad_norm = 0.0
    accum_num_losses = 0
    accum_audio_seconds = 0.0
    log_start_time = time.perf_counter()

    # CSV for loss curves (kept across a resume, see prepare_curves)
    curves_path = reports_dir / "pretrain_curves.csv"
    if is_main:
        prepare_curves(curves_path, resume_step)

    if is_main:
        print(f"\n{'=' * 60}")
        print(
            "Starting continued pre-training" if not resume else f"Resuming at step {resume_step}"
        )
        print(f"{'=' * 60}\n")

    # Rank-0-only decision (codebook collapse) that every rank must agree to
    # act on before the next collective op (DDP's backward all-reduce) --
    # otherwise a `break` gated by `if is_main:` alone would leave non-main
    # ranks waiting forever on a gradient sync that rank 0 never issues again.
    stop_signal = torch.zeros(1, device=device)

    while global_step < cfg.max_updates:
        optimizer.zero_grad()

        for accum_step in range(grad_accum_steps):
            # Get batch (restart iterator if exhausted = new epoch)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Compute Gumbel temperature for this step
            gumbel_temp = get_gumbel_temperature(
                global_step,
                cfg.max_gumbel_temperature,
                cfg.min_gumbel_temperature,
                cfg.gumbel_temperature_decay,
            )
            raw_model.set_gumbel_temperature(gumbel_temp)

            # Mask + negatives (required -- forward() returns loss=None without them)
            mask_time_indices, sampled_negative_indices, _sub_attention_mask = (
                compute_mask_and_negatives(
                    raw_model,
                    input_values,
                    attention_mask,
                    cfg.mask_time_prob,
                    cfg.mask_time_length,
                    device,
                )
            )

            # Forward pass
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_values,
                    attention_mask=attention_mask,
                    mask_time_indices=mask_time_indices,
                    sampled_negative_indices=sampled_negative_indices,
                )

            # Loss = contrastive + diversity_weight * diversity (computed internally
            # by Wav2Vec2ForPreTraining -- do not add diversity loss a second time).
            # Both components use reduction="sum" over masked positions, so the raw
            # loss/gradient scales with mask count -- normalized below via
            # multiply_grads(), not by dividing the loss itself (matches the HF
            # reference's post-backward gradient rescale).
            loss = outputs.loss
            num_losses = mask_time_indices.sum().float()

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if world_size > 1:
                # DDP already averaged this micro-batch's gradient across ranks
                # during backward(); rescale by world_size/total_num_losses to
                # turn that average into sum(grad_i)/sum(num_losses_i).
                num_losses_total = num_losses.clone().to(device)
                dist.all_reduce(num_losses_total, op=dist.ReduceOp.SUM)
                gradient_multiplier = world_size / num_losses_total.clamp(min=1)
            else:
                gradient_multiplier = 1.0 / num_losses.clamp(min=1)
            multiply_grads(trainable_params, gradient_multiplier)

            # Track losses for logging
            with torch.no_grad():
                num_losses_safe = num_losses.clamp(min=1)
                accum_loss += loss.item()
                accum_contrastive_loss += (
                    (outputs.contrastive_loss / num_losses_safe).item()
                    if outputs.contrastive_loss is not None
                    else 0.0
                )
                accum_diversity_loss += (
                    (outputs.diversity_loss / num_losses_safe).item()
                    if outputs.diversity_loss is not None
                    else 0.0
                )
                accum_num_losses += 1
                accum_audio_seconds += attention_mask.sum().item() / 16000.0  # 16kHz

        # Gradient clipping (grads are already rescaled by mask count above, so
        # this clips the properly-normalized gradient, matching get_grad_norm's
        # role as a monitoring-only stat in the HF reference)
        if cfg.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            accum_grad_norm += grad_norm.item()
        else:
            accum_grad_norm += 0.0

        # LR schedule
        lr = get_tri_stage_lr(
            global_step,
            cfg.peak_lr,
            cfg.warmup_updates,
            cfg.hold_ratio,
            cfg.max_updates,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Optimizer step
        optimizer.step()
        global_step += 1

        # ── Logging ──
        if is_main and global_step % cfg.eval_every_updates == 0:
            elapsed = time.perf_counter() - log_start_time
            avg_loss = accum_loss / max(accum_num_losses, 1)
            avg_contrastive = accum_contrastive_loss / max(accum_num_losses, 1)
            avg_diversity = accum_diversity_loss / max(accum_num_losses, 1)
            avg_grad_norm = accum_grad_norm / (cfg.eval_every_updates)
            audio_throughput = accum_audio_seconds / elapsed if elapsed > 0 else 0

            # Codebook perplexity from the model
            # wav2vec2 computes this as exp(entropy) over the codebook usage
            codebook_perplexity = 0.0
            if (
                hasattr(outputs, "codevector_perplexity")
                and outputs.codevector_perplexity is not None
            ):
                codebook_perplexity = outputs.codevector_perplexity.item()

            metrics = {
                "train/loss": round(avg_loss, 4),
                "train/contrastive_loss": round(avg_contrastive, 4),
                "train/diversity_loss": round(avg_diversity, 4),
                "train/codebook_perplexity": round(codebook_perplexity, 2),
                "train/grad_norm": round(avg_grad_norm, 4),
                "train/lr": lr,
                "train/gumbel_temp": round(gumbel_temp, 6),
                "train/audio_sec_per_wall_sec": round(audio_throughput, 1),
                "train/global_step": global_step,
            }

            print(
                f"Step {global_step}/{cfg.max_updates} | "
                f"loss={avg_loss:.4f} (contr={avg_contrastive:.4f} div={avg_diversity:.4f}) | "
                f"ppl={codebook_perplexity:.1f} | "
                f"gnorm={avg_grad_norm:.3f} | "
                f"lr={lr:.2e} | "
                f"temp={gumbel_temp:.4f} | "
                f"{audio_throughput:.0f} aud-s/wall-s"
            )

            if mlflow_logger:
                mlflow_logger.log_metrics(metrics, step=global_step)

            # Write to CSV
            with open(curves_path, "a") as f:
                f.write(
                    f"{global_step},{avg_loss:.6f},{avg_contrastive:.6f},"
                    f"{avg_diversity:.6f},{codebook_perplexity:.2f},"
                    f"{avg_grad_norm:.4f},{lr:.8f},{gumbel_temp:.6f},"
                    f"{audio_throughput:.1f}\n"
                )

            # Codebook collapse check
            if collapse_detector:
                should_continue = collapse_detector.check(codebook_perplexity, global_step)
                if not should_continue:
                    print("KILLING TRAINING DUE TO CODEBOOK COLLAPSE")
                    stop_signal[0] = 1.0
                    if mlflow_logger:
                        mlflow_logger.log_metrics({"train/killed_collapse": 1}, step=global_step)

            # Reset accumulators
            accum_loss = 0.0
            accum_contrastive_loss = 0.0
            accum_diversity_loss = 0.0
            accum_grad_norm = 0.0
            accum_num_losses = 0
            accum_audio_seconds = 0.0
            log_start_time = time.perf_counter()

        # Every rank must agree to stop before any rank stops calling
        # collective ops (DDP's backward-time gradient all-reduce), or the
        # surviving ranks hang forever waiting on rank 0.
        if world_size > 1:
            dist.broadcast(stop_signal, src=0)
        if stop_signal.item() > 0:
            break

        # ── Checkpointing ──
        if is_main and global_step % cfg.save_every_updates == 0:
            ckpt_dir = save_checkpoint(
                raw_model, feature_extractor, optimizer, global_step, gumbel_temp, output_dir
            )
            print(f"  Saved checkpoint: {ckpt_dir}")
            prune_checkpoints(output_dir, cfg.milestone_checkpoints, cfg.keep_last_n_checkpoints)

    # ── Final save ──
    if is_main:
        raw_model.save_pretrained(output_dir)
        feature_extractor.save_pretrained(output_dir)

        # Final metrics
        final_metrics = {
            "final_step": global_step,
            "final_loss": round(accum_loss / max(accum_num_losses, 1), 4)
            if accum_num_losses
            else None,
            "completed": global_step >= cfg.max_updates,
        }
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "pretrain_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        if mlflow_logger:
            mlflow_logger.log_metrics(
                {k: v for k, v in final_metrics.items() if isinstance(v, (int, float))},
                step=global_step,
            )
            mlflow_logger.end()

        print(f"\nPre-training complete. {global_step} updates.")
        print(f"Model saved to {output_dir}")

    if world_size > 1:
        dist.destroy_process_group()
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume-from",
        metavar="CKPT_DIR",
        default=None,
        help="resume from this checkpoint directory (e.g. models/.../checkpoint-3000)",
    )
    resume.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore checkpoints already in pretrain.output_dir and start from the base model "
        "(by default the newest complete checkpoint there is resumed automatically)",
    )
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    train(
        params,
        config_path=args.config,
        resume_from=args.resume_from,
        auto_resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
