"""Checkpoint directory conventions shared by train.py and select_checkpoint.py.

A checkpoint is `<output_dir>/checkpoint-<step>/`. train.py writes it to
`checkpoint-<step>.tmp` first and renames it, so a crash mid-save never leaves
a half-written directory that looks complete; `.tmp` dirs don't match
CHECKPOINT_RE and are ignored (and cleaned up on the next start).
"""

import re
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
HEAD_FILE = "masked_prediction_head.pt"
STATE_FILE = "training_state.pt"


def is_complete_checkpoint(path: Path) -> bool:
    """Backbone (loadable by `slsb --upstream`), head and optimizer state all present."""
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and has_weights
        and (path / HEAD_FILE).exists()
        and (path / STATE_FILE).exists()
    )


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
