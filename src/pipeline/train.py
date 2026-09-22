"""Continued pre-training of HuBERT Large on Sinhala/Tamil.

Adapted from HuggingFace's wav2vec2/HuBERT pretraining example.
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
`HubertForPreTraining.forward()` returns `loss=None` unless
`sampled_negative_indices` is supplied (see its forward() docstring --
"Required input for pre-training"), so negative sampling isn't optional.

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
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    HubertFeatureExtractor,
    HubertForPreTraining,
)
from transformers.models.hubert.modeling_hubert import (
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
# This is the schedule fairseq uses for HuBERT/Wav2Vec2 pre-training, not a cosine.


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
# (numpy-based) exactly as HuggingFace's HuBERT pretraining data collator
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
# HubertForPreTraining's contrastive_loss (reduction="sum") and
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


# ─── Training ──────────────────────────────────────────────────────────


def train(params: Params):
    cfg = params.pretrain

    # ── Distributed setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"World size: {world_size}, device: {device}")
        print(f"Base model: {cfg.base_model}")
        print(f"Precision: {cfg.precision}")
        print(f"Max updates: {cfg.max_updates}")
        print(f"Peak LR: {cfg.peak_lr}")
        print(f"Target batch seconds: {cfg.target_batch_seconds}")

    # ── Load model ──
    model = HubertForPreTraining.from_pretrained(cfg.base_model)

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
            "Make sure you are loading HubertForPreTraining, not HubertModel. "
            f"Loaded from: {cfg.base_model}"
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
    feature_extractor = HubertFeatureExtractor.from_pretrained(cfg.base_model)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.peak_lr,
        betas=tuple(cfg.adam_betas),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    # ── Data ──
    dataloader = create_dataloader(
        shar_dir="data/shars",
        per_device_max_seconds=cfg.per_device_max_seconds,
        num_workers=cfg.num_workers,
        seed=0,
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

    # ── Output dir ──
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──
    mlflow_logger = None
    collapse_detector = None
    if is_main:
        try:
            import yaml

            with open("params.yaml") as f:
                raw_params = yaml.safe_load(f)
            mlflow_cfg = raw_params.get("mlflow", {})
            mlflow_logger = MLflowLogger(
                tracking_uri=mlflow_cfg.get("tracking_uri", ""),
                experiment_name=mlflow_cfg.get("experiment_name", "ssl-pretraining"),
                run_name=f"hubert-large-si-ta-{cfg.target_batch_seconds}s",
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
            }
            mlflow_logger.log_params(flat_params)
        except Exception as e:  # noqa: BLE001 -- experiment tracking must never kill a training run
            print(f"WARNING: MLflow setup failed: {e}")
            mlflow_logger = None

        collapse_detector = CodebookCollapseDetector(
            num_codebooks=raw_model.config.num_codevector_groups,
            alert_threshold=10.0,
        )

    # ── Precision ──
    use_amp = cfg.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16

    # ── Training loop ──
    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_contrastive_loss = 0.0
    accum_diversity_loss = 0.0
    accum_grad_norm = 0.0
    accum_num_losses = 0
    accum_audio_seconds = 0.0
    log_start_time = time.perf_counter()

    # CSV for loss curves
    curves_path = Path("reports/pretrain_curves.csv")
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    if is_main:
        with open(curves_path, "w") as f:
            f.write(
                "step,loss,contrastive_loss,diversity_loss,codebook_perplexity,"
                "grad_norm,lr,gumbel_temp,audio_sec_per_wall_sec\n"
            )

    if is_main:
        print(f"\n{'=' * 60}")
        print("Starting continued pre-training")
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
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_values,
                    attention_mask=attention_mask,
                    mask_time_indices=mask_time_indices,
                    sampled_negative_indices=sampled_negative_indices,
                )

            # Loss = contrastive + diversity_weight * diversity (computed internally
            # by HubertForPreTraining -- do not add diversity loss a second time).
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
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            raw_model.save_pretrained(ckpt_dir)
            feature_extractor.save_pretrained(ckpt_dir)

            # Save optimizer and scheduler state for resumption
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "gumbel_temperature": gumbel_temp,
                },
                ckpt_dir / "training_state.pt",
            )

            print(f"  Saved checkpoint: {ckpt_dir}")

            # Manage checkpoint retention
            is_milestone = global_step in cfg.milestone_checkpoints
            if not is_milestone:
                # Remove old non-milestone checkpoints beyond keep_last_n
                all_ckpts = sorted(
                    output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
                )
                non_milestones = [
                    c
                    for c in all_ckpts
                    if int(c.name.split("-")[1]) not in cfg.milestone_checkpoints
                ]
                while len(non_milestones) > cfg.keep_last_n_checkpoints:
                    oldest = non_milestones.pop(0)
                    print(f"  Removing old checkpoint: {oldest.name}")
                    shutil.rmtree(oldest)

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
        with open("reports/pretrain_metrics.json", "w") as f:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    train(params)


if __name__ == "__main__":
    main()
