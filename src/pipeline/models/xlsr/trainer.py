"""Continued pre-training of the wav2vec2 family (XLS-R 300M, wav2vec2 Large) on Sinhala/Tamil.

The xlsr trainer: wav2vec2's contrastive objective with a Gumbel quantizer, via
`Wav2Vec2ForPreTraining`. XLS-R and wav2vec2 Large run this same code; their configs
(params.yaml, configs/models/wav2vec2.yaml) differ in the base model and output paths.

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
  # Single node (XLS-R; add --model wav2vec2 for wav2vec2 Large)
  torchrun --nproc_per_node=1 -m pipeline.train --config params.yaml

  # Two nodes
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
           --rdzv_backend=c10d --rdzv_endpoint=10.0.0.1:29500 \
           -m pipeline.train --config params.yaml
"""

import os
import time
from pathlib import Path

# Must be set before torch initialises CUDA. Batches vary in shape every step, and
# with the default allocator the cache fragments and keeps growing (measured: 46GB
# live -> 75GB+ reserved within ~20 micro-batches). On GB10's unified memory that
# growth eats system RAM until the host locks up. Expandable segments hold it flat.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)

from pipeline.callbacks import CodebookCollapseDetector
from pipeline.common.checkpoints import (  # noqa: F401 -- re-exported for callers and tests
    TRAINING_STATE_FILE,
    checkpoint_step,
    find_resume,
    list_checkpoints,
    load_training_state,
    prune_checkpoints,
    remove_stale_saves,
    write_checkpoint_atomically,
)
from pipeline.common.checkpoints import prepare_curves as _prepare_curves
from pipeline.common.schedule import get_tri_stage_lr
from pipeline.common.training import (
    already_finished,
    begin_run,
    finish_run,
    print_run_header,
    start_tracking,
    wrap_ddp,
)
from pipeline.common.training import grad_accum_steps as _grad_accum_steps
from pipeline.datamodule import create_dataloader
from pipeline.schema import Params

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

    For continued pre-training from a checkpoint whose codebook is already trained,
    max_temp should start at or near the floor the base model ended on (fairseq's LARGE
    runs end near 0.1-0.5), not at 2.0: a high temperature re-noises the existing codebook.
    params.yaml sets this.
    """
    return max(min_temp, max_temp * (decay**step))


# ─── Gradient normalisation by mask count ───────────────────────────────
#
# Wav2Vec2ForPreTraining's contrastive_loss (reduction="sum") and diversity_loss
# (scaled by mask_time_indices.sum()) are UNNORMALIZED sums over masked positions.
# The gradients of every micro-batch in an update accumulate in .grad, and are
# normalised ONCE, after the loop, by the masked-position count of the whole update
# (like fairseq). Rescaling after each micro-batch -- the HF example's pattern, which
# is only correct without gradient accumulation -- shrank the gradients already
# accumulated, so only the last micro-batch counted. Under DDP, DDP's backward-time
# all-reduce already averages across ranks, so the multiplier is world_size/total.


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


# ─── Checkpointing and resume ──────────────────────────────────────────
#
# The generic machinery (atomic writes, resume, pruning, curve trimming) lives in
# pipeline.common.checkpoints; what is specific to this trainer is the training-state
# payload (it also carries the Gumbel temperature) and the loss-curve columns.

CURVES_HEADER = (
    "step,loss,contrastive_loss,diversity_loss,codebook_perplexity,"
    "grad_norm,lr,gumbel_temp,audio_sec_per_wall_sec"
)


def save_checkpoint(
    raw_model, feature_extractor, optimizer, global_step: int, gumbel_temp: float, output_dir: Path
) -> Path:
    """Write checkpoint-<global_step>/ atomically (model, feature extractor, training state)."""

    def write(tmp: Path) -> None:
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

    return write_checkpoint_atomically(Path(output_dir), global_step, write)


def prepare_curves(path: Path, resume_step: int, header: str = CURVES_HEADER) -> None:
    _prepare_curves(path, resume_step, header)


# ─── Training ──────────────────────────────────────────────────────────


def train(
    params: Params,
    *,
    resume_from: str | Path | None = None,
    auto_resume: bool = True,
    dataloader=None,
    reports_dir: str | Path | None = None,
    use_mlflow: bool = True,
    device: str | None = None,
) -> int:
    """Run (or resume) pre-training. Returns the global step reached.

    resume_from: restart from this checkpoint directory.
    auto_resume: with no resume_from, restart from the newest complete
        checkpoint in pretrain.output_dir if there is one (so a crash followed by
        the same launch command just continues).
    dataloader / device / reports_dir / use_mlflow: seams for tests; in a real run
        they default to the shard loader, CUDA if present, model.reports_dir, and MLflow.
    """
    cfg = params.pretrain
    reports_dir = Path(params.model.reports_dir if reports_dir is None else reports_dir)

    # ── Distributed setup, output dir, resume point ──
    setup = begin_run(cfg, resume_from, auto_resume, device, extra_files=())
    world_size, is_main = setup.world_size, setup.is_main
    device, output_dir, resume, resume_step = (
        setup.device,
        setup.output_dir,
        setup.resume,
        setup.resume_step,
    )

    if already_finished(setup, cfg):
        return resume_step

    model_source = str(resume[0]) if resume else cfg.base_model

    print_run_header(setup, cfg)

    # ── Load model ──
    # layerdrop unset = the checkpoint's own value, which is what the original run used
    load_kwargs = {} if cfg.layerdrop is None else {"layerdrop": cfg.layerdrop}
    model = Wav2Vec2ForPreTraining.from_pretrained(model_source, **load_kwargs)

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

    # Full CoPT trains every weight, the CNN feature encoder included (as in fairseq
    # wav2vec 2.0 pre-training and Getman et al. 2024). freeze_feature_encoder: true is
    # the transformer-only (partial-CoPT) variant.
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        cnn = sum(p.numel() for p in model.wav2vec2.feature_extractor.parameters())
        print(
            f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable; "
            f"CNN feature encoder ({cnn / 1e6:.1f}M) "
            f"{'frozen' if cfg.freeze_feature_encoder else 'TRAINABLE'}"
        )

    # Apply the configured diversity-loss weight to the model config the loss reads from
    # (model is still unwrapped here, before DDP). Without this, pretrain.diversity_loss_weight
    # was logged but never used: the model kept the checkpoint's own value (0.1, which
    # happened to match the YAML).
    model.config.diversity_loss_weight = cfg.diversity_loss_weight

    model.to(device)

    model = wrap_ddp(model, setup)

    # Unwrap for accessing config/methods not proxied by DDP (mask/negative
    # sampling helpers, save_pretrained, set_gumbel_temperature)
    raw_model = model.module if hasattr(model, "module") else model

    # ── Feature extractor (saved with checkpoints; its do_normalize flag decides
    #    whether utterances are normalised, exactly as the base model expects) ──
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.base_model)
    normalize = bool(feature_extractor.do_normalize)
    if is_main:
        print(f"Waveform normalisation (do_normalize): {normalize}")

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
            shar_dir=params.shard.output_dir,
            per_device_max_seconds=cfg.per_device_max_seconds,
            num_workers=cfg.num_workers,
            seed=resume_step,
            normalize=normalize,
        )
    data_iter = iter(dataloader)

    # ── Gradient accumulation ──
    grad_accum_steps = _grad_accum_steps(cfg, setup)

    # ── Logging ──
    mlflow_logger = None
    collapse_detector = None
    if is_main:
        if use_mlflow:
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
                "normalize": normalize,
                "resumed_from_step": resume_step,
            }
            mlflow_logger = start_tracking(params, setup, flat_params)

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

    last_logged_loss: float | None = None

    while global_step < cfg.max_updates:
        optimizer.zero_grad()
        update_num_losses = torch.zeros((), device=device)

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
            # Both components use reduction="sum" over masked positions. The gradients of
            # every micro-batch accumulate in .grad and are normalised ONCE, after the loop,
            # by the masked-position count of the whole update (see below).
            loss = outputs.loss
            num_losses = mask_time_indices.sum().float()

            loss.backward()
            update_num_losses += num_losses.detach().to(device)

            # Track losses for logging
            with torch.no_grad():
                num_losses_safe = num_losses.clamp(min=1)
                # per masked frame, so it is comparable across steps and with contrastive_loss
                accum_loss += (loss / num_losses_safe).item()
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

        # Normalise the accumulated gradient ONCE per update by the masked-position count
        # of the whole update (all micro-batches, all ranks), as fairseq does. DDP averages
        # each backward across ranks, so the multiplier is world_size / total_num_losses.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if world_size > 1:
            dist.all_reduce(update_num_losses, op=dist.ReduceOp.SUM)
        multiply_grads(trainable_params, world_size / update_num_losses.clamp(min=1))

        # Gradient clipping (grads are already normalised by mask count above, so this
        # clips the properly-normalised gradient)
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
            last_logged_loss = avg_contrastive  # per-masked-frame, for pretrain_metrics.json

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

    finish_run(
        setup,
        raw_model,
        feature_extractor,
        cfg,
        reports_dir,
        global_step,
        last_logged_loss,
        mlflow_logger,
    )
    return global_step
