"""Continued pre-training of WavLM Large on Sinhala/Tamil.

WavLM's pretraining objective (HuBERT's) is masked prediction of k-means
pseudo-labels (produced by pipeline.labels.fit_kmeans + assign_cluster_labels from
layer-N features of the original pretrained model -- see labels/layer_features.py),
not wav2vec2's contrastive learning with a quantizer/codebook. transformers has no
WavLMForPreTraining class for the latter to even target, so this loads a plain
WavLMModel encoder (pipeline.models.wavlm.model.WavLMForMaskedPrediction) and trains
a linear head against the precomputed cluster ids. The encoder is the already-
pretrained checkpoint; its architecture and the objective are not changed. WavLM's
utterance mixing is a data augmentation applied by the data loader
(pretrain.utterance_mix_prob).

Key points:
  - Loads from Lhotse Shar tarballs (not HF datasets)
  - Uses raw DDP + torch.amp (not HF Accelerator)
  - Tri-stage LR schedule (warmup / hold / exponential decay)
  - MLflow logging to DagsHub
  - Masked-prediction stall detection
  - Reads all config from params.yaml (+ configs/models/wavlm.yaml) via the Pydantic schema

Mask generation reuses transformers' own `_compute_mask_indices` (the same
helper WavLMModel._mask_hidden_states falls back to when no
mask_time_indices is given) so training explicitly controls which frames are
masked -- masking is WavLMModel's own mechanism (via `masked_spec_embed`,
see modeling_wavlm.py's `_mask_hidden_states`), not a hand-rolled one.
`mask_time_indices` must be `torch.bool`: WavLMModel indexes
`hidden_states[mask_time_indices] = ...` directly with no internal dtype
cast (unlike Wav2Vec2ForPreTraining, which casts before use).

Resuming: checkpoints (backbone + head + optimizer + step) are written every
pretrain.save_every_updates. Relaunching the SAME command after a crash resumes
automatically from the newest complete checkpoint in pretrain.output_dir; pass
--resume-from CKPT_DIR to pick a specific one, or --no-resume to start over.

Usage:
  torchrun --nproc_per_node=1 -m pipeline.train --model wavlm
"""

import os
import time
from pathlib import Path

# Must be set before torch initialises CUDA. Batches vary in shape every step, and
# with the default allocator the cache fragments and keeps growing. On GB10's unified
# memory that growth eats system RAM until the host locks up. Expandable segments hold
# it flat.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch import nn
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wavlm.modeling_wavlm import _compute_mask_indices

from pipeline.callbacks import MaskedAccuracyStallDetector
from pipeline.common.checkpoints import (
    TRAINING_STATE_FILE,
    prune_checkpoints,
    write_checkpoint_atomically,
)
from pipeline.common.checkpoints import prepare_curves as _prepare_curves
from pipeline.common.masked_prediction import HEAD_FILE, prediction_perplexity
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
from pipeline.models.wavlm.model import WavLMForMaskedPrediction
from pipeline.schema import Params

CURVES_HEADER = (
    "step,loss,masked_accuracy,unmasked_accuracy,pred_perplexity,"
    "grad_norm,lr,audio_sec_per_wall_sec"
)


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
#
# The generic machinery (atomic writes, resume, pruning, curve trimming) lives in
# pipeline.common.checkpoints. A checkpoint is complete only with the head file next
# to the backbone.


def save_checkpoint(
    raw_model, feature_extractor, optimizer, global_step: int, output_dir: Path
) -> Path:
    """Write checkpoint-<global_step>/ atomically: backbone, head, feature extractor and
    training state."""

    def write(tmp: Path) -> None:
        raw_model.save_pretrained(tmp)
        feature_extractor.save_pretrained(tmp)
        torch.save(
            {"optimizer": optimizer.state_dict(), "global_step": global_step},
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
    num_clusters = params.cluster.num_clusters
    reports_dir = Path(params.model.reports_dir if reports_dir is None else reports_dir)

    # ── Distributed setup, output dir, resume point ──
    setup = begin_run(cfg, resume_from, auto_resume, device, extra_files=(HEAD_FILE,))
    world_size, is_main = setup.world_size, setup.is_main
    device, output_dir, resume, resume_step = (
        setup.device,
        setup.output_dir,
        setup.resume,
        setup.resume_step,
    )

    if already_finished(setup, cfg):
        return resume_step

    print_run_header(setup, cfg, extra_lines=(f"Num clusters: {num_clusters}",))

    # ── Load model ──
    # layerdrop unset = 0.1, the stock value of the base models
    layerdrop = 0.1 if cfg.layerdrop is None else cfg.layerdrop
    model = WavLMForMaskedPrediction(cfg.base_model, num_clusters=num_clusters, layerdrop=layerdrop)
    if resume:
        model.load_checkpoint(resume[0])

    # Full CoPT trains every weight, the CNN feature encoder included;
    # freeze_feature_encoder: true is the transformer-only (partial-CoPT) variant.
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        state = "frozen" if cfg.freeze_feature_encoder else "TRAINABLE"
        print(
            f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable "
            f"(CNN feature encoder {state})"
        )

    model.to(device)

    model = wrap_ddp(model, setup)

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
    # already pretrained, so the head gets its own LR multiplier (lr_mult).
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
        try:
            optimizer.load_state_dict(resume[1]["optimizer"])
        except ValueError as e:
            raise RuntimeError(
                f"cannot restore the optimizer from {resume[0]}: {e} -- do "
                f"freeze_feature_encoder or the model differ from when it was saved?"
            ) from e

    # ── Data ──
    # Seeded with the resume step, so a resumed run doesn't replay the exact
    # shard order of the first epoch (a fresh run has step 0, i.e. seed 0).
    if dataloader is None:
        dataloader = create_dataloader(
            shar_dir=params.shard.output_dir,
            per_device_max_seconds=cfg.per_device_max_seconds,
            num_workers=cfg.num_workers,
            seed=resume_step,
            labels_path=cfg.labels_path,
            normalize=normalize,
            utterance_mix_prob=cfg.utterance_mix_prob,
        )
    data_iter = iter(dataloader)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Gradient accumulation ──
    grad_accum_steps = _grad_accum_steps(cfg, setup)

    # ── Logging ──
    mlflow_logger = None
    stall_detector = None
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
                "num_clusters": num_clusters,
                "freeze_feature_encoder": cfg.freeze_feature_encoder,
                "layerdrop": cfg.layerdrop,
                "head_lr_mult": cfg.head_lr_mult,
                "normalize": normalize,
                "utterance_mix_prob": cfg.utterance_mix_prob,
                "resumed_from_step": resume_step,
            }
            mlflow_logger = start_tracking(params, setup, flat_params)

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
    last_logged_loss: float | None = None

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
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(
                    input_values,
                    attention_mask=attention_mask,
                    mask_time_indices=mask_time_indices,
                )

            # Frame counts should already agree by construction (the labels were
            # extracted to the same conv-output length compute_mask uses) -- crop to
            # the overlap as a defensive guard against any off-by-a-few edge case
            # rather than crashing mid-run.
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
            last_logged_loss = avg_loss  # masked-CE, for pretrain_metrics.json

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
            ckpt_dir = save_checkpoint(
                raw_model, feature_extractor, optimizer, global_step, output_dir
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
