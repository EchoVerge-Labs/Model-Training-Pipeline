"""select_checkpoint: one selector for all five models."""

import json
import sys

import pytest
import yaml

from pipeline import select_checkpoint
from pipeline.select_checkpoint import find_milestone_checkpoints


def _backbone(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"")


def test_milestones_need_only_a_backbone_so_contrastive_checkpoints_qualify(tmp_path):
    for step in (4500, 9000):
        _backbone(tmp_path / f"checkpoint-{step}")  # no masked-prediction head, as for XLS-R
    _backbone(tmp_path / "checkpoint-1500")  # rolling, not a milestone
    found = find_milestone_checkpoints(str(tmp_path), [9000, 4500, 13500])
    assert [p.name for p in found] == ["checkpoint-4500", "checkpoint-9000"]


def test_a_milestone_without_weights_is_skipped(tmp_path, capsys):
    (tmp_path / "checkpoint-4500").mkdir()
    (tmp_path / "checkpoint-4500" / "config.json").write_text("{}")
    assert find_milestone_checkpoints(str(tmp_path), [4500]) == []
    assert "checkpoint-4500" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config", "suffix_arg", "expected"),
    [
        ("params-xlsr.yaml", [], "checkpoint_selection-xlsr.json"),
        ("params-wav2vec2.yaml", [], "checkpoint_selection-wav2vec2.json"),
        ("params-wavlm.yaml", [], "checkpoint_selection-wavlm.json"),
        ("params.yaml", ["--report-suffix=-custom"], "checkpoint_selection-custom.json"),
    ],
)
def test_report_suffix_defaults_to_the_model_key(
    tmp_path, monkeypatch, config, suffix_arg, expected
):
    with open(config) as f:
        raw = yaml.safe_load(f)
    raw["pretrain"]["output_dir"] = str(tmp_path / "model")
    raw["checkpoint_selection"]["output_dir"] = str(tmp_path / "selected")
    cfg = tmp_path / "params.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    for step in raw["pretrain"]["milestone_checkpoints"]:
        _backbone(tmp_path / "model" / f"checkpoint-{step}")

    scores = iter([3.0, 1.0, 2.0])
    monkeypatch.setattr(select_checkpoint, "evaluate_checkpoint", lambda *a, **k: next(scores))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), *suffix_arg])
    select_checkpoint.main()

    report = json.loads((tmp_path / "reports" / expected).read_text())
    assert report["best_checkpoint"] == "checkpoint-9000"
    assert (tmp_path / "selected" / "config.json").exists()
