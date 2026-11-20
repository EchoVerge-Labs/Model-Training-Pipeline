#!/usr/bin/env python3
"""Phase 0: Measure real training throughput before scoping experiments.

Run this BEFORE committing to max_updates or timeline estimates.
Records samples/sec, GPU utilisation, memory usage, estimates wall-clock.

Usage:
    python scripts/phase0_throughput.py --steps 100
    python scripts/phase0_throughput.py --steps 100 --precision bf16
    python scripts/phase0_throughput.py --steps 100 --model wavlm   # any model: xlsr wav2vec2 wavlm hubert mhubert

Output:
    <model.reports_dir>/phase0.json — machine-readable results (reports/ for XLS-R)
    stdout — human-readable summary
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def measure_throughput(
    params, steps: int = 100, batch_seconds: float = 200.0, precision: str = "bf16"
):
    """Measure pre-training throughput with synthetic data on the real model of `params`
    (params.yaml + the chosen model's config). The step is the one that model's trainer runs."""
    cfg = params.pretrain
    trainer = params.model.trainer
    MASK_TIME_PROB, MASK_TIME_LENGTH = cfg.mask_time_prob, cfg.mask_time_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory: {mem_gb:.1f} GB")

    # Load the actual model — not a toy config
    print(f"Loading {cfg.base_model} (trainer: {trainer}) ...")
    quantizer_keys = []
    if trainer == "xlsr":
        from transformers import Wav2Vec2ForPreTraining

        kwargs = {} if cfg.layerdrop is None else {"layerdrop": cfg.layerdrop}
        model = Wav2Vec2ForPreTraining.from_pretrained(cfg.base_model, **kwargs)
        if cfg.freeze_feature_encoder:  # follow params.yaml: a trainable CNN costs far more memory
            model.freeze_feature_encoder()
        model.to(device)

        # Verify quantizer
        quantizer_keys = [k for k in model.state_dict() if "quantizer" in k]
        assert len(quantizer_keys) > 0, "No quantizer weights — wrong checkpoint?"
        print(f"Quantizer keys present: {len(quantizer_keys)} ✓")
    else:
        num_clusters = params.cluster.num_clusters
        layerdrop = 0.1 if cfg.layerdrop is None else cfg.layerdrop
        if trainer == "wavlm":
            from pipeline.models.wavlm.model import WavLMForMaskedPrediction as Wrapper
        else:
            from pipeline.models.hubert.model import HubertForMaskedPrediction as Wrapper
        model = Wrapper(cfg.base_model, num_clusters=num_clusters, layerdrop=layerdrop)
        if cfg.freeze_feature_encoder:  # follow params.yaml: a trainable CNN costs far more memory
            model.freeze_feature_encoder()
        model.to(device)

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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01,
    )

    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # A real pre-training step, using the same helpers as the model's trainer: mask
    # generation (+ negative sampling for wav2vec2 -- forward() returns loss=None without
    # them), forward under autocast, the model's loss, backward, clip, optimizer step.
    # Synthetic audio only; no real data IO or DDP sync.
    trainable = [p for p in model.parameters() if p.requires_grad]

    class _Out:
        def __init__(self, loss):
            self.loss = loss

    if trainer == "xlsr":
        from pipeline.models.xlsr.trainer import compute_mask_and_negatives, multiply_grads

        model.set_gumbel_temperature(1.0)

        def train_step():
            x = torch.randn(batch_size, seq_len, device=device)
            m = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            mask_time_indices, sampled_negatives, _ = compute_mask_and_negatives(
                model, x, m, MASK_TIME_PROB, MASK_TIME_LENGTH, device,
            )
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(x, attention_mask=m, mask_time_indices=mask_time_indices,
                            sampled_negative_indices=sampled_negatives)
            out.loss.backward()
            num_losses = mask_time_indices.sum().float().clamp(min=1)
            multiply_grads(trainable, 1.0 / num_losses)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            return out
    else:
        import importlib

        from torch import nn

        compute_mask = importlib.import_module(f"pipeline.models.{trainer}.trainer").compute_mask
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        def train_step():
            x = torch.randn(batch_size, seq_len, device=device)
            m = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            mask_time_indices, _ = compute_mask(
                model, x, m, MASK_TIME_PROB, MASK_TIME_LENGTH, device
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
            return _Out(loss.detach())

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
        out = train_step()
        total_audio_sec += actual_batch_sec

        if (i + 1) % 20 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            rate = total_audio_sec / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {i+1}/{steps}  {rate:.1f} aud-s/s  loss={out.loss.item():.4f}  "
                  f"peak_mem={peak_mem:.1f}GB")

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    throughput = total_audio_sec / wall
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Estimate real pre-training time from the configured schedule, so this
    # can't drift from what train.py will actually run.
    target_batch = float(cfg.target_batch_seconds)  # seconds of audio per optimizer step
    max_updates = cfg.max_updates
    # With grad accumulation: each update processes target_batch seconds
    total_audio_for_training = max_updates * target_batch
    est_hours = total_audio_for_training / throughput / 3600

    results = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "precision": precision,
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
            "name": params.model.tag,
            "trainer": trainer,
            "base_model": cfg.base_model,
            "total_params_M": round(total_params / 1e6, 1),
            "trainable_params_M": round(trainable_params / 1e6, 1),
            "quantizer_keys": len(quantizer_keys),
        },
        "estimate": {
            "target_batch_seconds": target_batch,
            "max_updates": max_updates,
            "total_audio_hours": round(total_audio_for_training / 3600, 1),
            "estimated_wall_hours": round(est_hours, 1),
            "estimated_wall_days": round(est_hours / 24, 1),
        },
    }

    print(f"\n{'='*60}")
    print("  PHASE 0 RESULTS — single node")
    print(f"{'='*60}")
    print(f"  Throughput:    {throughput:.1f} audio-sec / wall-sec")
    print(f"  Peak memory:   {peak_mem:.1f} GB")
    print(f"  Pre-train est: {est_hours:.0f}h ({est_hours/24:.1f}d) "
          f"for {max_updates} updates × {target_batch:.0f}s/update")
    print(f"{'='*60}")
    print()
    print("  ⚠  This is a SINGLE-NODE estimate with SYNTHETIC data (full train step:")
    print("  mask/negative sampling + forward + backward + optimizer).")
    print("  Real throughput may differ due to:")
    print("    - Real audio IO (measure with real shards next)")
    print("    - 2-node DDP sync overhead")
    print("    - Dynamic batching variance")
    print()
    print("  Next steps:")
    print("    1. Run nccl_bandwidth_test.sh to verify inter-node link")
    print("    2. Re-run with --nnodes=2 for 2-node scaling factor")
    print("    3. Replace synthetic data with real shards for IO test")

    out_path = Path(params.model.reports_dir) / "phase0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 throughput measurement")
    parser.add_argument("--steps", type=int, default=100, help="Steps to time")
    parser.add_argument("--batch-seconds", type=float, default=200.0,
                        help="Audio seconds per synthetic batch")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    from pipeline.common.config import add_config_args, params_from_args

    add_config_args(parser)  # --config, --model, --overlay
    args = parser.parse_args()
    measure_throughput(params_from_args(args), args.steps, args.batch_seconds, args.precision)
