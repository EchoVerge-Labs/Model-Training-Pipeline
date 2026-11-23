"""Checkpoint directory conventions shared by the trainer and select_checkpoint.py.

A checkpoint is `<output_dir>/checkpoint-<step>/`. the trainer writes it to
`checkpoint-<step>.tmp` first and renames it, so a crash mid-save never leaves
a half-written directory that looks complete; `.tmp` dirs don't match
CHECKPOINT_RE and are ignored (and cleaned up on the next start).
"""

import re
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
HEAD_FILE = "masked_prediction_head.pt"
STATE_FILE = "training_state.pt"


def has_backbone(path: Path) -> bool:
    """A loadable encoder (what `slsb --upstream` needs): config plus weights. True for
    every model family, unlike is_complete_checkpoint, which also needs the head."""
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    return path.is_dir() and (path / "config.json").exists() and has_weights


def is_complete_checkpoint(path: Path) -> bool:
    """Backbone (loadable by `slsb --upstream`), head and optimizer state all present."""
    return has_backbone(path) and (path / HEAD_FILE).exists() and (path / STATE_FILE).exists()


def list_checkpoints(output_dir: str | Path) -> list[tuple[int, Path]]:
    """Complete checkpoints as (step, dir), oldest first."""
    found = []
    for p in Path(output_dir).glob("checkpoint-*"):
        m = CHECKPOINT_RE.match(p.name)
        if m and is_complete_checkpoint(p):
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_checkpoint(output_dir: str | Path) -> tuple[int, Path] | None:
    found = list_checkpoints(output_dir)
    return found[-1] if found else None


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
