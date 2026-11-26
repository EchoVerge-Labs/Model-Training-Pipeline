#!/usr/bin/env python3
"""Phase 0: Measure real training throughput before scoping experiments.

Run this BEFORE committing to max_updates or timeline estimates.
Records samples/sec, GPU utilisation, memory usage, estimates wall-clock.

Usage:
    python scripts/phase0_throughput.py --steps 100                    # mHuBERT-147
    python scripts/phase0_throughput.py --model wavlm --steps 100
    python scripts/phase0_throughput.py --model xlsr --precision bf16

Output:
    reports/phase0-<model>.json — machine-readable results
    stdout — human-readable summary

--model is one of mhubert147, hubert, wavlm, xlsr, wav2vec2. It picks the params file,
the model class and the training step to time: masked prediction on cluster ids for
the HuBERT family, contrastive loss with negatives for XLS-R / wav2vec2.
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.models.registry import MODEL_REGISTRY, describe_feature_encoder


def load_model(model_key: str, params, device, freeze_cnn: bool):
    """-> (model, extra info for the report). Real pretrained weights, not a toy config."""
    base_model = params.pretrain.base_model
    print(f"Loading {base_model} ...")
    if params.model_family == "masked_prediction":
        from pipeline.training.masked_model import MaskedPredictionModel

        model = MaskedPredictionModel(
            MODEL_REGISTRY[model_key].hf_model_type,
            base_model,
            num_clusters=params.cluster.num_clusters,
            layerdrop=params.pretrain.layerdrop,
        )
        extra = {"num_clusters": params.cluster.num_clusters}
    else:
        from transformers import Wav2Vec2ForPreTraining

        model = Wav2Vec2ForPreTraining.from_pretrained(base_model)
        quantizer_keys = [k for k in model.state_dict() if "quantizer" in k]
        assert len(quantizer_keys) > 0, "No quantizer weights — wrong checkpoint?"
        print(f"Quantizer keys present: {len(quantizer_keys)} ✓")
        extra = {"quantizer_keys": len(quantizer_keys)}
    if freeze_cnn:
        model.freeze_feature_encoder()
    print(describe_feature_encoder(model, MODEL_REGISTRY[model_key]))
    return model.to(device), extra


def make_train_step(model_key, model, params, trainer, batch_size, seq_len, device, amp_dtype):
    """A real pre-training step, using the same helpers as the trainer: mask generation
    (required -- the encoder applies masked_spec_embed at these positions; contrastive
    forward() returns loss=None without negatives), forward under autocast, loss,
    backward, clip, optimizer step. Synthetic audio only; no real data IO or DDP sync."""
    cfg = params.pretrain
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01
    )

    def batch():
        x = torch.randn(batch_size, seq_len, device=device)
        return x, torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    if params.model_family == "masked_prediction":
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        num_clusters = params.cluster.num_clusters

        def step():
            x, m = batch()
            mask_time_indices, _ = trainer.compute_mask(
                model, x, m, cfg.mask_time_prob, cfg.mask_time_length, device
            )
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x, attention_mask=m, mask_time_indices=mask_time_indices)
            target = torch.randint(0, num_clusters, logits.shape[:2], device=device)
            target[~mask_time_indices] = -100
            loss = loss_fn(logits.transpose(1, 2), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            return loss

        return step

    model.set_gumbel_temperature(cfg.max_gumbel_temperature)

    def step():
        x, m = batch()
        mask_time_indices, sampled_negatives, _ = trainer.compute_mask_and_negatives(
            model, x, m, cfg.mask_time_prob, cfg.mask_time_length, device
        )
        with torch.autocast("cuda", dtype=amp_dtype):
            out = model(
                x,
                attention_mask=m,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negatives,
            )
        out.loss.backward()
        num_losses = mask_time_indices.sum().float().clamp(min=1)
        trainer.multiply_grads(trainable, 1.0 / num_losses)  # per-mask-count rescale
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return out.loss

    return step


def measure_throughput(
    model_key: str = "mhubert147",
    steps: int = 100,
    batch_seconds: float = 200.0,
    precision: str = "bf16",
    freeze_cnn: bool | None = None,
):
    """Measure pre-training throughput with synthetic data on real model."""
    from pipeline.common.schema import Params

    spec = MODEL_REGISTRY[model_key]
    params = Params.from_yaml(str(REPO_ROOT / spec.params_file))
    assert params.model_key == model_key, f"{spec.params_file} is for {params.model_key}"
    trainer = importlib.import_module(spec.trainer_module)

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory: {mem_gb:.1f} GB")

    # follow pretrain.freeze_feature_encoder (a trainable CNN costs much more activation
    # memory, so measure the setup the real run uses), unless overridden on the command line
    freeze_cnn = params.pretrain.freeze_feature_encoder if freeze_cnn is None else freeze_cnn
    model, model_extra = load_model(model_key, params, device, freeze_cnn)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")

    # Synthetic batch
    sample_rate = 16000
    seq_seconds = 20.0
    seq_len = int(seq_seconds * sample_rate)
    batch_size = max(1, int(batch_seconds / seq_seconds))
    actual_batch_sec = batch_size * seq_seconds
    print(f"Batch: {batch_size} × {seq_seconds:.0f}s = {actual_batch_sec:.0f}s audio")

    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    train_step = make_train_step(
        model_key, model, params, trainer, batch_size, seq_len, device, amp_dtype
    )

    # Warmup (not timed)
    print("Warming up (5 steps) ...")
    model.train()
    for _ in range(5):
        train_step()

    # Reset peak memory after warmup
    torch.cuda.reset_peak_memory_stats()

    # Timed run
    print(f"Timing {steps} steps ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_audio_sec = 0.0

    for i in range(steps):
        loss = train_step()
        total_audio_sec += actual_batch_sec

        if (i + 1) % 20 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            rate = total_audio_sec / elapsed
            print(
                f"  {i + 1}/{steps}  {rate:.1f} aud-s/s  loss={loss.item():.4f}  "
                f"peak_mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            )

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    throughput = total_audio_sec / wall
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Estimate real pre-training time from the configured schedule, so this
    # can't drift from what the trainer will actually run.
    pretrain_cfg = params.pretrain
    target_batch = float(pretrain_cfg.target_batch_seconds)  # seconds of audio per optimizer step
    max_updates = pretrain_cfg.max_updates
    # With grad accumulation: each update processes target_batch seconds
    total_audio_for_training = max_updates * target_batch
    est_hours = total_audio_for_training / throughput / 3600

    results = {
        "model_key": model_key,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(),
        "precision": precision,
        "freeze_feature_encoder": freeze_cnn,
        "measured": {
            "batch_size": batch_size,
            "seq_seconds": seq_seconds,
            "batch_seconds": actual_batch_sec,
            "steps": steps,
            "wall_seconds": round(wall, 1),
            "throughput_audio_sec_per_wall_sec": round(throughput, 2),
            "peak_gpu_memory_gb": round(peak_mem, 2),
        },
        "model": {
            "total_params_M": round(total_params / 1e6, 1),
            "trainable_params_M": round(trainable_params / 1e6, 1),
            **model_extra,
        },
        "estimate": {
            "target_batch_seconds": target_batch,
            "max_updates": max_updates,
            "total_audio_hours": round(total_audio_for_training / 3600, 1),
            "estimated_wall_hours": round(est_hours, 1),
            "estimated_wall_days": round(est_hours / 24, 1),
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  PHASE 0 RESULTS — {model_key}, single node")
    print(f"{'=' * 60}")
    print(f"  Throughput:    {throughput:.1f} audio-sec / wall-sec")
    print(f"  Peak memory:   {peak_mem:.1f} GB")
    print(
        f"  Pre-train est: {est_hours:.0f}h ({est_hours / 24:.1f}d) "
        f"for {max_updates} updates × {target_batch:.0f}s/update"
    )
    print(f"{'=' * 60}")
    print()
    print("  ⚠  This is a SINGLE-NODE estimate with SYNTHETIC data (full train step:")
    print("  mask (+ negative sampling) + forward + backward + optimizer).")
    print("  Real throughput may differ due to:")
    print("    - Real audio IO (measure with real shards next)")
    print("    - 2-node DDP sync overhead")
    print("    - Dynamic batching variance")
    print()
    print("  Next steps:")
    print("    1. Run nccl_bandwidth_test.sh to verify inter-node link")
    print("    2. Re-run with --nnodes=2 for 2-node scaling factor")
    print("    3. Replace synthetic data with real shards for IO test")

    out_path = Path(f"reports/phase0-{model_key}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 throughput measurement")
    parser.add_argument("--model", default="mhubert147", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--steps", type=int, default=100, help="Steps to time")
    parser.add_argument(
        "--batch-seconds", type=float, default=200.0, help="Audio seconds per synthetic batch"
    )
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument(
        "--freeze-cnn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override pretrain.freeze_feature_encoder (default: follow the params file)",
    )
    args = parser.parse_args()
    measure_throughput(args.model, args.steps, args.batch_seconds, args.precision, args.freeze_cnn)
