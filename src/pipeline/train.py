"""Continued pre-training of WavLM Large on Sinhala/Tamil.

WavLM's pretraining objective (HuBERT's) is masked prediction of k-means
pseudo-labels (data/labels/, produced by pipeline.fit_kmeans +
pipeline.assign_cluster_labels from layer-N features of the original
pretrained model -- see layer_features.py), not wav2vec2's contrastive learning with a quantizer/codebook. transformers
has no WavLMForPreTraining class for the latter to even target, so this
loads a plain WavLMModel encoder (pipeline.wavlm_model.WavLMForMaskedPrediction)
and trains a linear head against the precomputed cluster ids. The encoder is
the already-pretrained checkpoint; its architecture and the objective are not
changed. Re-running the launch script resumes from the newest complete
checkpoint (backbone + head + optimizer + step).

Key changes from a typical HF fine-tuning script:
  - Loads from Lhotse Shar tarballs (not HF datasets)
  - Uses raw DDP + torch.amp (not HF Accelerator)
  - Tri-stage LR schedule (warmup / hold / exponential decay)
  - MLflow logging to DagsHub

Mask generation reuses transformers' own `_compute_mask_indices` (the same
helper WavLMModel._mask_hidden_states falls back to when no
mask_time_indices is given) so training explicitly controls which frames are
masked -- masking is WavLMModel's own mechanism (via `masked_spec_embed`,
see modeling_wavlm.py's `_mask_hidden_states`), not a hand-rolled one.
`mask_time_indices` must be `torch.bool`: WavLMModel indexes
`hidden_states[mask_time_indices] = ...` directly with no internal dtype
cast (unlike Wav2Vec2ForPreTraining, which casts before use).

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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wavlm.modeling_wavlm import _compute_mask_indices

from pipeline.callbacks import MaskedAccuracyStallDetector, MLflowLogger
from pipeline.checkpoints import STATE_FILE, latest_checkpoint, list_checkpoints
from pipeline.datamodule import create_dataloader
from pipeline.schema import Params
from pipeline.wavlm_model import WavLMForMaskedPrediction

# ─── Tri-stage LR schedule ──────────────────────────────────────────────
#
# Phase 1: linear warmup from 0 → peak_lr         (0 → warmup_updates)
# Phase 2: hold at peak_lr                        (warmup_updates → hold_end)
# Phase 3: exponential decay from peak_lr → 0     (hold_end → max_updates)
#
# hold_end = warmup_updates + hold_ratio * (max_updates - warmup_updates)
#
# This is the schedule fairseq uses for HuBERT/WavLM/Wav2Vec2 pre-training, not a cosine.


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


# ─── Mask generation ─────────────────────────────────────────────────────
#
# Reuses transformers' own `_compute_mask_indices` (numpy-based), computed
# against the *feature-extractor output* length (the transformer's input
# length after the CNN downsamples), using the reduced attention mask so
# padded frames are never masked.


def compute_mask(
    model,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float,
    mask_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mask_time_indices, sub_attention_mask), both as torch tensors
    on `device`. mask_time_indices is torch.bool -- WavLMModel indexes
    hidden_states with it directly (hidden_states[mask_time_indices] = ...),
    with no internal dtype cast."""
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
    mask_time_indices = torch.tensor(mask_time_indices, dtype=torch.bool, device=device)
    return mask_time_indices, sub_attention_mask


# ─── Checkpointing ───────────────────────────────────────────────────────


def save_checkpoint(raw_model, feature_extractor, optimizer, global_step: int, ckpt_dir: Path):
    """Writes ckpt_dir atomically: everything goes to <ckpt_dir>.tmp, then one
    rename, so a crash mid-save can't leave a directory that looks complete."""
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    raw_model.save_pretrained(tmp)
    feature_extractor.save_pretrained(tmp)
    torch.save({"optimizer": optimizer.state_dict(), "global_step": global_step}, tmp / STATE_FILE)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    os.replace(tmp, ckpt_dir)


def prediction_perplexity(prob_sum: torch.Tensor, n_frames: int) -> float:
    """exp(entropy) of the average predicted distribution over masked frames:
    ~1 means the model predicts one cluster for everything (collapse), up to
    num_clusters when it uses them all evenly."""
    mean_probs = prob_sum / max(n_frames, 1)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-10))).sum()
    return float(torch.exp(entropy).item())


# ─── Training ──────────────────────────────────────────────────────────


def train(params: Params):
    cfg = params.pretrain
    num_clusters = params.cluster.num_clusters

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
        print(f"Num clusters: {num_clusters}")
        print(f"Target batch seconds: {cfg.target_batch_seconds}")

    # ── Resume point ──
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        for stale in output_dir.glob("checkpoint-*.tmp"):  # a save that never finished
            shutil.rmtree(stale, ignore_errors=True)
    resume = latest_checkpoint(output_dir)
    resume_step = resume[0] if resume else 0
    if world_size > 1:
        # Every rank reads its own output_dir; ranks on different nodes must
        # agree on where they resume or their optimizer states would diverge.
        steps = torch.tensor([resume_step], device=device)
        lo, hi = steps.clone(), steps.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if lo.item() != hi.item():
            raise RuntimeError(
                f"ranks disagree on the resume checkpoint (steps {lo.item()}..{hi.item()}) -- "
                f"copy {output_dir}/checkpoint-{hi.item()} to every node"
            )

    # ── Load model ──
    model = WavLMForMaskedPrediction(
        cfg.base_model, num_clusters=num_clusters, layerdrop=cfg.layerdrop
    )
    if resume:
        model.load_checkpoint(resume[1])
        if is_main:
            print(f"Resuming from {resume[1]} (step {resume_step})")

    # Freeze CNN feature encoder — standard for continued pre-training
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        frozen_note = " (CNN frozen)" if cfg.freeze_feature_encoder else ""
        print(
            f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable{frozen_note}"
        )

    model.to(device)

    if world_size > 1:
        # layerdrop skips layers at random, leaving their parameters without a
        # gradient in that step, which DDP only tolerates with this flag.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=cfg.layerdrop > 0)

    # Unwrap for accessing config/methods not proxied by DDP (mask helpers,
    # save_pretrained)
    raw_model = model.module if hasattr(model, "module") else model

    # ── Feature extractor (saved with checkpoints; its do_normalize flag decides
    #    whether utterances are normalised, exactly as the base model expects) ──
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.base_model)
    normalize = bool(feature_extractor.do_normalize)
    if is_main:
        print(f"Waveform normalisation (do_normalize): {normalize}")

    # ── Optimizer ──
    # The head starts random (the checkpoint has none) while the encoder is
    # already pretrained, so the head gets a larger LR (lr_mult).
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in raw_model.wavlm.parameters() if p.requires_grad],
                "lr_mult": 1.0,
            },
            {"params": list(raw_model.final_proj.parameters()), "lr_mult": cfg.head_lr_mult},
        ],
        lr=cfg.peak_lr,
        betas=tuple(cfg.adam_betas),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    if resume:
        state = torch.load(resume[1] / STATE_FILE, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])

    # ── Data ──
    dataloader = create_dataloader(
        shar_dir=params.shard.output_dir,
        per_device_max_seconds=cfg.per_device_max_seconds,
        num_workers=cfg.num_workers,
        seed=0,
        labels_path=cfg.labels_path,
        normalize=normalize,
        utterance_mix_prob=cfg.utterance_mix_prob,
    )
    data_iter = iter(dataloader)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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
    stall_detector = None
    if is_main:
        try:
            import yaml

            with open("params.yaml") as f:
                raw_params = yaml.safe_load(f)
            mlflow_cfg = raw_params.get("mlflow", {})
            mlflow_logger = MLflowLogger(
                tracking_uri=mlflow_cfg.get("tracking_uri", ""),
                experiment_name=mlflow_cfg.get("experiment_name", "ssl-pretraining"),
                run_name=f"wavlm-large-si-ta-{cfg.target_batch_seconds}s",
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
                "num_clusters": num_clusters,
                "freeze_feature_encoder": cfg.freeze_feature_encoder,
                "layerdrop": cfg.layerdrop,
                "head_lr_mult": cfg.head_lr_mult,
                "normalize": normalize,
                "utterance_mix_prob": cfg.utterance_mix_prob,
                "resumed_from_step": resume_step,
            }
            mlflow_logger.log_params(flat_params)
        except Exception as e:  # noqa: BLE001 -- experiment tracking must never kill a training run
            print(f"WARNING: MLflow setup failed: {e}")
            mlflow_logger = None

        # Kill only after ~a fifth of the run (at least 3 checks) sits at chance
        # or collapsed, so the threshold is reachable within max_updates.
        stall_detector = MaskedAccuracyStallDetector(
            num_clusters=num_clusters,
            max_consecutive_before_kill=max(3, cfg.max_updates // cfg.eval_every_updates // 5),
        )

    # ── Precision ──
    use_amp = cfg.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16

    # ── Training loop ──
    model.train()
    global_step = resume_step
    accum_loss = 0.0
    accum_correct = 0
    accum_valid = 0
    accum_un_correct = 0
    accum_un_valid = 0
    accum_probs = torch.zeros(num_clusters, device=device)
    accum_grad_norm = 0.0
    accum_num_losses = 0
    accum_audio_seconds = 0.0
    log_start_time = time.perf_counter()

    # CSV for loss curves
    curves_path = Path("reports/pretrain_curves.csv")
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    if is_main and not (resume and curves_path.exists()):
        with open(curves_path, "w") as f:
            f.write(
                "step,loss,masked_accuracy,unmasked_accuracy,pred_perplexity,"
                "grad_norm,lr,audio_sec_per_wall_sec\n"
            )

    if is_main:
        print(f"\n{'=' * 60}")
        print("Starting continued pre-training")
        print(f"{'=' * 60}\n")

    # Rank-0-only decision (masked-accuracy stall) that every rank must agree
    # to act on before the next collective op (DDP's backward all-reduce) --
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
            labels = batch["labels"].to(device)

            mask_time_indices, _sub_attention_mask = compute_mask(
                raw_model,
                input_values,
                attention_mask,
                cfg.mask_time_prob,
                cfg.mask_time_length,
                device,
            )

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(
                    input_values,
                    attention_mask=attention_mask,
                    mask_time_indices=mask_time_indices,
                )

            # Frame counts should already agree by construction (MFCC labels
            # were extracted to the same conv-output length train.py computes
            # here) -- crop to the overlap as a defensive guard against any
            # off-by-a-few edge case rather than crashing mid-run.
            common_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :common_len].float()  # CE in fp32, not bf16
            label_frames = labels[:, :common_len]
            mask_time_indices = mask_time_indices[:, :common_len]
            target = label_frames.clone()
            target[~mask_time_indices] = -100

            if (target != -100).any():
                loss = loss_fn(logits.transpose(1, 2), target)
            else:
                # no masked, labelled frame (all-padding crop): a zero loss that
                # still touches the graph, so DDP's gradient sync stays in step
                loss = logits.sum() * 0.0

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            # Track losses for logging
            with torch.no_grad():
                valid = target != -100
                n_valid = int(valid.sum().item())
                preds = logits.argmax(-1)
                if n_valid > 0:
                    accum_correct += int((preds[valid] == target[valid]).sum().item())
                    accum_valid += n_valid
                    accum_probs += torch.softmax(logits[valid], dim=-1).sum(0)
                unmasked = (label_frames != -100) & ~mask_time_indices
                n_un = int(unmasked.sum().item())
                if n_un > 0:
                    accum_un_correct += int(
                        (preds[unmasked] == label_frames[unmasked]).sum().item()
                    )
                    accum_un_valid += n_un
                accum_loss += loss.item()
                accum_num_losses += 1
                accum_audio_seconds += attention_mask.sum().item() / 16000.0  # 16kHz

        # Gradient clipping
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
            param_group["lr"] = lr * param_group["lr_mult"]

        # Optimizer step
        optimizer.step()
        global_step += 1

        # ── Logging ──
        if is_main and global_step % cfg.eval_every_updates == 0:
            elapsed = time.perf_counter() - log_start_time
            avg_loss = accum_loss / max(accum_num_losses, 1)
            masked_accuracy = accum_correct / max(accum_valid, 1)
            unmasked_accuracy = accum_un_correct / max(accum_un_valid, 1)
            perplexity = prediction_perplexity(accum_probs, accum_valid)
            avg_grad_norm = accum_grad_norm / (cfg.eval_every_updates)
            audio_throughput = accum_audio_seconds / elapsed if elapsed > 0 else 0

            metrics = {
                "train/loss": round(avg_loss, 4),
                "train/masked_accuracy": round(masked_accuracy, 4),
                "train/unmasked_accuracy": round(unmasked_accuracy, 4),
                "train/pred_perplexity": round(perplexity, 2),
                "train/grad_norm": round(avg_grad_norm, 4),
                "train/lr": lr,
                "train/audio_sec_per_wall_sec": round(audio_throughput, 1),
                "train/global_step": global_step,
            }

            print(
                f"Step {global_step}/{cfg.max_updates} | "
                f"loss={avg_loss:.4f} | "
                f"acc={masked_accuracy:.4f} | "
                f"acc_unmasked={unmasked_accuracy:.4f} | "
                f"ppl={perplexity:.1f} | "
                f"gnorm={avg_grad_norm:.3f} | "
                f"lr={lr:.2e} | "
                f"{audio_throughput:.0f} aud-s/wall-s"
            )

            if mlflow_logger:
                mlflow_logger.log_metrics(metrics, step=global_step)

            # Write to CSV
            with open(curves_path, "a") as f:
                f.write(
                    f"{global_step},{avg_loss:.6f},{masked_accuracy:.6f},"
                    f"{unmasked_accuracy:.6f},{perplexity:.3f},"
                    f"{avg_grad_norm:.4f},{lr:.8f},{audio_throughput:.1f}\n"
                )

            # Masked-accuracy stall check
            if stall_detector:
                should_continue = stall_detector.check(masked_accuracy, global_step, perplexity)
                if not should_continue:
                    print("KILLING TRAINING: masked-prediction accuracy stalled")
                    stop_signal[0] = 1.0
                    if mlflow_logger:
                        mlflow_logger.log_metrics({"train/killed_stall": 1}, step=global_step)

            # Reset accumulators
            accum_loss = 0.0
            accum_correct = 0
            accum_valid = 0
            accum_un_correct = 0
            accum_un_valid = 0
            accum_probs.zero_()
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
            save_checkpoint(raw_model, feature_extractor, optimizer, global_step, ckpt_dir)
            print(f"  Saved checkpoint: {ckpt_dir}")

            # Manage checkpoint retention
            is_milestone = global_step in cfg.milestone_checkpoints
            if not is_milestone:
                # Remove old non-milestone checkpoints beyond keep_last_n
                non_milestones = [
                    path
                    for step, path in list_checkpoints(output_dir)
                    if step not in cfg.milestone_checkpoints
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
