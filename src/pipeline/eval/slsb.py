"""Finding the SLSB-benchmark install this pipeline benchmarks with, and checking its version.

SLSB is a separate repository (EchoVerge-Labs/SLSB-benchmark) with its own environment. Its
scores are only comparable within one protocol version, so the pipeline refuses to benchmark
against an slsb other than the version the params file names (`benchmark.slsb_version`).

Where slsb is looked for, in order:
  1. $SLSB_BIN                           the slsb executable itself
  2. $SLSB_DIR/.venv/bin/slsb            a checkout with its own virtualenv ($SLSB_DIR defaults
                                         to ../SLSB-benchmark, the sibling checkout)
  3. `slsb` on PATH
SLSB's own scripts/log_results.py (the leaderboard logger) is taken from $SLSB_DIR.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class SlsbNotFound(RuntimeError):
    pass


def slsb_dir() -> Path:
    return Path(os.environ.get("SLSB_DIR") or REPO_ROOT.parent / "SLSB-benchmark")


def slsb_command() -> list[str]:
    """The command that runs slsb (a one-element list), or SlsbNotFound."""
    candidates = [os.environ.get("SLSB_BIN"), str(slsb_dir() / ".venv" / "bin" / "slsb")]
    for c in candidates:
        if c and Path(c).exists():
            return [c]
    found = shutil.which("slsb")
    if found:
        return [found]
    raise SlsbNotFound(
        "slsb not found. Install SLSB-benchmark next to this repo "
        f"({slsb_dir()}) or set SLSB_DIR / SLSB_BIN, or put `slsb` on PATH."
    )


def slsb_python() -> str | None:
    """The Python of slsb's own environment (to run its scripts with), if it has one."""
    py = slsb_dir() / ".venv" / "bin" / "python"
    return str(py) if py.exists() else None


def log_results_script() -> Path:
    script = slsb_dir() / "scripts" / "log_results.py"
    if not script.is_file():
        raise SlsbNotFound(f"{script} not found -- set SLSB_DIR to the SLSB-benchmark checkout")
    return script


def slsb_version(command: list[str] | None = None) -> str:
    """e.g. "0.5.0" from `slsb --version`."""
    out = subprocess.run(
        [*(command or slsb_command()), "--version"], capture_output=True, text=True, check=False
    )
    match = re.search(r"(\d+\.\d+\.\d+)", out.stdout + out.stderr)
    if not match:
        raise SlsbNotFound(f"could not read a version from `slsb --version`: {out.stdout!r}")
    return match.group(1)


def require_version(expected: str, command: list[str] | None = None) -> str:
    """Return the installed version, or raise if it is not `expected`."""
    have = slsb_version(command)
    if have != expected:
        raise SlsbNotFound(
            f"slsb {have} is installed but benchmark.slsb_version is {expected}: scores from "
            f"different protocol versions are not comparable. Check out the matching tag of "
            f"SLSB-benchmark (git checkout v{expected}) or change benchmark.slsb_version."
        )
    return have
