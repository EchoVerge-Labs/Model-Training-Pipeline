"""The scripts at least parse and start: a broken import or shell syntax error fails here, not
hours into a GPU job."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_phase0_throughput_starts_and_lists_the_model_options():
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / "phase0_throughput.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--model" in out.stdout and "--steps" in out.stdout


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
def test_shell_scripts_have_valid_syntax(script):
    subprocess.run(["bash", "-n", str(SCRIPTS / script)], check=True)


def test_the_launch_script_passes_the_model_to_the_trainer():
    text = (SCRIPTS / "launch_pretrain.sh").read_text()
    assert 'MODEL="${MODEL:-}"' in text and '${MODEL:+--model "$MODEL"}' in text
