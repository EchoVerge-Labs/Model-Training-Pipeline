"""Checkpoint directory conventions and helpers shared by both trainers and pipeline.eval.select_checkpoint.

A checkpoint is `<output_dir>/checkpoint-<step>/`. the trainer writes it to
`checkpoint-<step>.tmp` first and renames it, so a crash mid-save never leaves
a half-written directory that looks complete; `.tmp` dirs don't match
CHECKPOINT_RE and are ignored (and cleaned up on the next start).
"""

import os
import re
import shutil
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
HEAD_FILE = "masked_prediction_head.pt"
STATE_FILE = "training_state.pt"


def checkpoint_step(path: Path) -> int | None:
    """The step in a checkpoint-<step> directory name, or None if it isn't one."""
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def has_backbone(path: Path) -> bool:
    """A loadable encoder (what `slsb --upstream` needs): config plus weights (sharded or
    not). True for every model family."""
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return path.is_dir() and (path / "config.json").exists() and has_weights


def is_complete_checkpoint(path: Path, with_head: bool = True) -> bool:
    """Backbone and optimizer state present, plus the prediction head for the masked-
    prediction family (with_head); the contrastive family has no separate head."""
    complete = has_backbone(path) and (path / STATE_FILE).exists()
    return complete and (not with_head or (path / HEAD_FILE).exists())


def list_checkpoints(output_dir: str | Path, with_head: bool = True) -> list[tuple[int, Path]]:
    """Complete checkpoints as (step, dir), oldest first."""
    found = []
    for p in Path(output_dir).glob("checkpoint-*"):
        step = checkpoint_step(p)
        if step is not None and is_complete_checkpoint(p, with_head):
            found.append((step, p))
    return sorted(found)


def latest_checkpoint(output_dir: str | Path, with_head: bool = True) -> tuple[int, Path] | None:
    found = list_checkpoints(output_dir, with_head)
    return found[-1] if found else None


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


def save_checkpoint(
    raw_model, feature_extractor, optimizer, global_step: int, ckpt_dir: Path, **extra_state
) -> Path:
    """Write ckpt_dir atomically: everything goes to <ckpt_dir>.tmp, then one rename, so a
    crash mid-save (a checkpoint is several GB) can't leave a directory that looks complete.
    extra_state is saved next to the optimizer (e.g. the contrastive Gumbel temperature)."""
    import torch

    ckpt_dir = Path(ckpt_dir)
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    raw_model.save_pretrained(tmp)
    feature_extractor.save_pretrained(tmp)
    torch.save(
        {"optimizer": optimizer.state_dict(), "global_step": global_step, **extra_state},
        tmp / STATE_FILE,
    )
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    os.replace(tmp, ckpt_dir)
    return ckpt_dir


def load_training_state(ckpt_dir: Path, device) -> dict:
    """Read training_state.pt. weights_only: it holds tensors and plain numbers, so
    nothing needs to be unpickled as code."""
    import torch

    state = torch.load(Path(ckpt_dir) / STATE_FILE, map_location=device, weights_only=True)
    missing = {"optimizer", "global_step"} - set(state)
    if missing:
        raise ValueError(f"{Path(ckpt_dir) / STATE_FILE} is missing {sorted(missing)}")
    return state


def find_resume(
    resume_from: str | Path | None,
    output_dir: Path,
    device,
    auto_resume: bool = True,
    with_head: bool = True,
) -> tuple[Path, dict] | None:
    """Decide what to resume from. Returns (checkpoint dir, training state) or None.

    An explicit resume_from must be a complete, readable checkpoint (error
    otherwise). Otherwise, with auto_resume, the newest complete checkpoint in
    output_dir is used -- and if it turns out to be unreadable (e.g. a crash
    truncated it) the next-newest is tried, so one bad file can't block a relaunch.
    """
    if resume_from is not None:
        path = Path(resume_from)
        if not is_complete_checkpoint(path, with_head):
            raise FileNotFoundError(
                f"--resume-from {path}: not a complete checkpoint "
                f"(needs config.json, model weights and {STATE_FILE})"
            )
        return path, load_training_state(path, device)
    if not auto_resume:
        return None
    for _, path in reversed(list_checkpoints(output_dir, with_head)):
        try:
            return path, load_training_state(path, device)
        except Exception as e:  # noqa: BLE001 -- any unreadable file just means "try the previous one"
            print(f"WARNING: skipping unreadable checkpoint {path}: {e}")
    return None


def prepare_curves(path: Path, resume_step: int, header: str) -> None:
    """Fresh run: start the loss-curve CSV. Resume: keep the rows up to the step
    being resumed from and drop later ones, which get re-logged when those steps
    re-run -- so the history has no gaps and no duplicates. `header` is the family's
    own column list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if resume_step > 0 and path.exists() else []
    if not lines:
        path.write_text(header + "\n")
        return
    kept = [row for row in lines[1:] if row and int(row.split(",", 1)[0]) <= resume_step]
    path.write_text("\n".join([lines[0], *kept]) + "\n")
