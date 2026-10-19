"""Config overlays: run a variant of params.yaml without copying it.

An overlay (e.g. configs/language_homogeneous.yaml) holds only the keys a variant
changes and is deep-merged over params.yaml, so everything it does not mention --
data, preprocessing, optimizer, schedule, masking, batch size, max_updates --
is the original run's by construction.

CLI (used by the Makefile / scripts to read a merged value):
  python -m pipeline.variant get benchmark.mlflow_run_prefix --overlay configs/X.yaml
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

# Outputs of the original joint run and of other experiments. A variant must never
# write here: its checkpoints, selected model and reports would overwrite the baseline.
PROTECTED_MODEL_DIRS = frozenset(
    {
        "models/wavlm-large-si-ta-200h",
        "models/selected",
    }
)
PROTECTED_REPORT_DIRS = frozenset({"reports"})
PROTECTED_BENCH_DIRS = frozenset(
    {
        "reports/bench/wavlm_large_adapted",
    }
)


def deep_merge(base: dict, overlay: dict) -> dict:
    """overlay's keys win; nested dicts merge recursively. Returns a new dict."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_raw(path: str = "params.yaml", overlay: str | None = None) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overlay:
        with open(overlay) as f:
            raw = deep_merge(raw, yaml.safe_load(f) or {})
    return raw


def _norm(p: str) -> str:
    return str(Path(p).as_posix()).rstrip("/").removeprefix("./")


def assert_isolated(raw: dict) -> None:
    """Refuse a language-homogeneous config whose outputs collide with the baseline's."""
    pre = raw.get("pretrain", {})
    if not pre.get("language_batching", {}).get("enabled", False):
        return
    problems = []
    if _norm(pre["output_dir"]) in PROTECTED_MODEL_DIRS:
        problems.append(f"pretrain.output_dir {pre['output_dir']} belongs to another run")
    sel = raw.get("checkpoint_selection", {}).get("output_dir", "models/selected")
    if _norm(sel) in PROTECTED_MODEL_DIRS:
        problems.append(f"checkpoint_selection.output_dir {sel} belongs to another run")
    reports = raw.get("experiment", {}).get("reports_dir", "reports")
    if _norm(reports) in PROTECTED_REPORT_DIRS:
        problems.append(f"experiment.reports_dir {reports} would overwrite the baseline's curves")
    if _norm(
        raw.get("checkpoint_selection", {}).get("report_path", "reports/checkpoint_selection.json")
    ) == ("reports/checkpoint_selection.json"):
        problems.append("checkpoint_selection.report_path would overwrite the baseline's report")
    if problems:
        raise ValueError(
            "language_batching output paths would overwrite the original joint run:\n  - "
            + "\n  - ".join(problems)
        )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    get = sub.add_parser("get", help="print one merged config value (dotted key)")
    get.add_argument("key")
    get.add_argument("--config", default="params.yaml")
    get.add_argument("--overlay", default=None)
    args = parser.parse_args()

    value = load_raw(args.config, args.overlay)
    for part in args.key.split("."):
        if not isinstance(value, dict) or part not in value:
            sys.exit(f"ERROR: {args.key} not found in the merged config")
        value = value[part]
    print(value)


if __name__ == "__main__":
    main()
