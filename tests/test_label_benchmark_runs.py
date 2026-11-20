"""scripts/label_benchmark_runs.py: the pure helpers that build run names and tags."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "label_benchmark_runs.py"
spec = importlib.util.spec_from_file_location("label_benchmark_runs", SCRIPT)
lbr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lbr)


def test_tags_parse_key_value_pairs():
    assert lbr.parse_tags("model=wavlm-large,base_model=microsoft/wavlm-large, a = b ") == {
        "model": "wavlm-large",
        "base_model": "microsoft/wavlm-large",
        "a": "b",
    }
    assert lbr.parse_tags("") == {}


def test_a_tag_without_a_value_is_an_error():
    with pytest.raises(argparse.ArgumentTypeError):
        lbr.parse_tags("model")


def test_a_local_model_is_tagged_with_how_it_was_evaluated_and_which_checkpoint(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "preprocessor_config.json").write_text(json.dumps({"do_normalize": True}))
    report = tmp_path / "checkpoint_selection.json"
    report.write_text(json.dumps({"best_checkpoint": "checkpoint-9000"}))
    tags = lbr.local_upstream_tags(str(tmp_path / "ckpt"), report)
    assert tags == {"eval_normalize": "true", "checkpoint": "9000"}


def test_a_hub_model_has_no_local_tags():
    assert lbr.local_upstream_tags("microsoft/wavlm-large", None) == {}
