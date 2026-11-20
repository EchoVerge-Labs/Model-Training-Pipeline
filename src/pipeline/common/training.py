"""The parts of a training run that every trainer does the same way.

The three trainers (xlsr, wavlm, hubert) differ in their model, loss and metrics, and each keeps
its own loop. What is identical -- starting the run (distributed setup, output directory, resume),
wrapping the model for DDP, sizing gradient accumulation, setting up MLflow, and the final save --
lives here, so a fix to any of it is made once.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pipeline.callbacks import MLflowLogger
from pipeline.common.checkpoints import find_resume, remove_stale_saves
from pipeline.common.distributed import (
    DistInfo,
    assert_ranks_agree_on_resume,
    init_distributed,
    pick_device,
)
from pipeline.schema import Params, PretrainConfig


@dataclass
class RunSetup:
    dist: DistInfo
    device: torch.device
    output_dir: Path
    resume: tuple[Path, dict] | None  # (checkpoint dir, training state) when resuming
    resume_step: int
    resume_from: str | Path | None  # what the caller asked for, only to word the log line

    @property
    def is_main(self) -> bool:
        return self.dist.is_main

    @property
    def world_size(self) -> int:
        return self.dist.world_size


def begin_run(
    cfg: PretrainConfig,
    resume_from: str | Path | None,
    auto_resume: bool,
    device: str | None,
    extra_files=(),
) -> RunSetup:
    """Distributed setup, the output directory, and the resume point (agreed by every rank).

    extra_files: files besides config.json / weights / training state that make a checkpoint
    complete (the masked-prediction head of HuBERT and WavLM).
    """
    dist_info = init_distributed()
    dev = pick_device(dist_info, device, cfg.max_gpu_memory_fraction)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist_info.is_main:
        remove_stale_saves(output_dir)  # a save that never finished

    resume = find_resume(resume_from, output_dir, dev, auto_resume, extra_files=extra_files)
    resume_step = int(resume[1]["global_step"]) if resume else 0
    if dist_info.world_size > 1:
        assert_ranks_agree_on_resume(resume_step if resume else -1, dev)
    return RunSetup(dist_info, dev, output_dir, resume, resume_step, resume_from)


def already_finished(setup: RunSetup, cfg: PretrainConfig) -> bool:
    """True (after saying so) if the resumed checkpoint already reaches max_updates."""
    if not (setup.resume and setup.resume_step >= cfg.max_updates):
        return False
    if setup.is_main:
        print(
            f"{setup.resume[0].name} is at step {setup.resume_step}, which already reaches "
            f"max_updates ({cfg.max_updates}): nothing to train. "
            f"(Pass --no-resume to start over.)"
        )
    if setup.world_size > 1:
        dist.destroy_process_group()
    return True


def print_run_header(setup: RunSetup, cfg: PretrainConfig, extra_lines=()) -> None:
    if not setup.is_main:
        return
    print(f"World size: {setup.world_size}, device: {setup.device}")
    print(f"Base model: {cfg.base_model}")
    if setup.resume:
        how = "explicitly" if setup.resume_from is not None else "automatically"
        print(f"Resuming {how} from {setup.resume[0]} at step {setup.resume_step}")
    else:
        print("Starting from the base model (no checkpoint resumed)")
    print(f"Precision: {cfg.precision}")
    print(f"Max updates: {cfg.max_updates}")
    print(f"Peak LR: {cfg.peak_lr}")
    for line in extra_lines:
        print(line)
    print(f"Target batch seconds: {cfg.target_batch_seconds}")


def wrap_ddp(model, setup: RunSetup):
    """DDP for more than one process. Layerdrop skips layers at random, leaving their parameters
    without a gradient in that step, which DDP only tolerates with find_unused_parameters."""
    if setup.world_size == 1:
        return model
    return DDP(
        model,
        device_ids=[setup.dist.local_rank],
        find_unused_parameters=model.config.layerdrop > 0,
    )


def grad_accum_steps(cfg: PretrainConfig, setup: RunSetup) -> int:
    """target_batch_seconds / (per_device_max_seconds * world_size), at least 1."""
    steps = max(1, int(cfg.target_batch_seconds / (cfg.per_device_max_seconds * setup.world_size)))
    if setup.is_main:
        print(f"Gradient accumulation steps: {steps}")
        print(
            f"Effective batch: ~{cfg.per_device_max_seconds * setup.world_size * steps:.0f}s "
            f"audio per update"
        )
    return steps


def start_tracking(params: Params, setup: RunSetup, flat_params: dict) -> MLflowLogger | None:
    """Open this run's MLflow run, tag it, log its params. Tracking must never kill a training
    run, so any failure just means no logging.

    The run name is also the model's name in the SLSB leaderboard (tag slsb.model), so a
    pre-training run and its benchmark row can be matched.
    """
    cfg = params.pretrain
    try:
        model_name = (
            params.model.run_name or f"{params.model.tag}-si-ta-{cfg.target_batch_seconds}s"
        )
        run_name = model_name
        if setup.resume:
            run_name += f"-resume{setup.resume_step}"
        logger = MLflowLogger(
            tracking_uri=params.mlflow.tracking_uri,
            experiment_name=params.mlflow.experiment_name,
            run_name=run_name,
        )
        # copt = full continued pre-training; any other method is a PE-CoPT variant
        # (lora, adapter, gc_lora, ...). Derived from cfg.method so the tags stay
        # correct once those methods exist, rather than being hard-coded to full-copt.
        is_full = cfg.method == "copt"
        logger.set_tags(
            {
                **({"family": params.model.family} if params.model.family else {}),
                "kind": "adapted",
                "method": cfg.method,
                "training_method": "full-copt" if is_full else "pe-copt",
                "peft": "none" if is_full else cfg.method,
                "data": params.model.data,
                "base_model": cfg.base_model,
                "slsb.model": model_name,
            }
        )
        logger.log_params(flat_params)
        return logger
    except Exception as e:  # noqa: BLE001 -- experiment tracking must never kill a training run
        print(f"WARNING: MLflow setup failed: {e}")
        return None


def finish_run(
    setup: RunSetup,
    raw_model,
    feature_extractor,
    cfg: PretrainConfig,
    reports_dir: Path,
    global_step: int,
    final_loss: float | None,
    mlflow_logger: MLflowLogger | None,
) -> None:
    """Final save of the model, the final-metrics file, and closing the MLflow run.

    final_loss is the last value logged to the curves (each trainer's last avg loss). The
    running accumulators are reset at every log step, and max_updates is normally a multiple
    of eval_every_updates, so reading them here would give 0/None -- hence the last logged
    value is passed in instead.
    """
    if setup.is_main:
        raw_model.save_pretrained(setup.output_dir)
        feature_extractor.save_pretrained(setup.output_dir)

        final_metrics = {
            "final_step": global_step,
            "final_loss": round(final_loss, 4) if final_loss is not None else None,
            # provenance: tie this output dir to the model that produced it, so selection
            # and benchmarking can verify they are operating on the right model (a stale
            # metrics file left in another model's reports dir no longer passes silently).
            "output_dir": str(setup.output_dir),
            "base_model": cfg.base_model,
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
        print(f"Model saved to {setup.output_dir}")

    if setup.world_size > 1:
        dist.destroy_process_group()


__all__ = [
    "RunSetup",
    "already_finished",
    "begin_run",
    "finish_run",
    "grad_accum_steps",
    "print_run_header",
    "start_tracking",
    "wrap_ddp",
]
