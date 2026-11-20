"""Checkpoint directories, atomic saving, resume and pruning -- shared by every trainer.

A checkpoint is a directory checkpoint-<step>/ holding the model (save_pretrained),
the feature extractor, and training_state.pt (optimizer state and global_step, plus
whatever extra scalars a trainer needs). Everything else the loop needs -- the LR,
the Gumbel temperature, the data-loader seed -- is a pure function of global_step,
so restoring the optimizer and the step is all a resume takes.

Models that keep part of their state outside the HuggingFace checkpoint (the
masked-prediction head of HuBERT / WavLM) name those files in `extra_files`; a
checkpoint without them is not complete.
"""

import json
import os
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
TRAINING_STATE_FILE = "training_state.pt"


def checkpoint_step(path: Path) -> int | None:
    """The step in a checkpoint-<step> directory name, or None if it isn't one."""
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def is_complete_checkpoint(path: Path, extra_files: Iterable[str] = ()) -> bool:
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and has_weights
        and (path / TRAINING_STATE_FILE).exists()
        and all((path / name).exists() for name in extra_files)
    )


def list_checkpoints(output_dir: Path, extra_files: Iterable[str] = ()) -> list[Path]:
    """Complete checkpoint directories under output_dir, newest (highest step) first."""
    extra_files = tuple(extra_files)
    found = [
        (step, path)
        for path in Path(output_dir).glob("checkpoint-*")
        if (step := checkpoint_step(path)) is not None and is_complete_checkpoint(path, extra_files)
    ]
    return [path for _, path in sorted(found, reverse=True)]


def remove_stale_saves(output_dir: Path) -> None:
    """Delete saves that never finished: .tmp-checkpoint-<step> (this layout) and
    checkpoint-<step>.tmp (the layout the HuBERT / WavLM branches used)."""
    for stale in [*output_dir.glob(".tmp-checkpoint-*"), *output_dir.glob("checkpoint-*.tmp")]:
        shutil.rmtree(stale, ignore_errors=True)


def write_checkpoint_atomically(
    output_dir: Path, global_step: int, write: Callable[[Path], None]
) -> Path:
    """Write checkpoint-<global_step>/ atomically; `write(dir)` fills the directory.

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

    write(tmp)

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
    resume_from: str | Path | None,
    output_dir: Path,
    device,
    auto_resume: bool = True,
    extra_files: Iterable[str] = (),
) -> tuple[Path, dict] | None:
    """Decide what to resume from. Returns (checkpoint dir, training state) or None.

    An explicit resume_from must be a complete, readable checkpoint (error
    otherwise). Otherwise, with auto_resume, the newest complete checkpoint in
    output_dir is used -- and if it turns out to be unreadable (e.g. a crash
    truncated it) the next-newest is tried, so one bad file can't block a relaunch.
    """
    extra_files = tuple(extra_files)
    if resume_from is not None:
        path = Path(resume_from)
        if not is_complete_checkpoint(path, extra_files):
            needs = ", ".join(["config.json", "model weights", TRAINING_STATE_FILE, *extra_files])
            raise FileNotFoundError(
                f"--resume-from {path}: not a complete checkpoint (needs {needs})"
            )
        return path, load_training_state(path, device)
    if not auto_resume:
        return None
    for path in list_checkpoints(output_dir, extra_files):
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


def prepare_curves(path: Path, resume_step: int, header: str) -> None:
    """Fresh run: start the loss-curve CSV. Resume: keep the rows up to the step
    being resumed from and drop later ones, which get re-logged when those steps
    re-run -- so the history has no gaps and no duplicates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if resume_step > 0 and path.exists() else []
    if not lines:
        path.write_text(header + "\n")
        return
    kept = [row for row in lines[1:] if row and int(row.split(",", 1)[0]) <= resume_step]
    path.write_text("\n".join([lines[0], *kept]) + "\n")


def pretraining_problem(reports_dir: str | Path, max_updates: int) -> str | None:
    """Why this model's pre-training is not finished, or None if it is.

    Selecting a checkpoint and benchmarking only make sense after pre-training has run to
    pretrain.max_updates. A run stopped early (a collapse detector, a crash, Ctrl-C) leaves
    milestone checkpoints behind, and benchmarking those would put a half-trained model on the
    leaderboard. train writes <reports_dir>/pretrain_metrics.json only when it returns.
    """
    path = Path(reports_dir) / "pretrain_metrics.json"
    if not path.is_file():
        return f"{path} not found: pre-training has not finished (or never ran)"
    metrics = json.loads(path.read_text())
    step = metrics.get("final_step")
    if not metrics.get("completed") or step is None or step < max_updates:
        return f"pre-training stopped at step {step} of {max_updates} ({path})"
    return None


def require_pretraining_complete(reports_dir: str | Path, max_updates: int) -> None:
    problem = pretraining_problem(reports_dir, max_updates)
    if problem:
        raise SystemExit(
            f"ERROR: {problem}.\nFinish pre-training first (rerun the same train command: it "
            f"resumes), or pass --allow-incomplete to go on anyway."
        )
