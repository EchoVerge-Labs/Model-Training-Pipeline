"""Aggregate benchmark results into a comparison table.

Reads reports/<name>/bench/metrics.json for the run and for each frozen baseline in the
params file's benchmark.baselines (reports/<model>-frozen/bench/) -- a copy of slsb's own
results_<upstream>.json (see pipeline.eval.benchmark), a list of
per-(task,language,seed) result entries, not a flat {task: score} dict.
This groups entries by task family (matching slsb's own data/<prefix>*
directory convention) and reports mean +/- std of the primary metric across
languages and seeds for each family.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from pipeline.common.schema import add_run_args, load_params, variants_arg

# Mirrors slsb.tasks.FAMILY_DIR_PREFIXES (SLSB-benchmark's own family->dirname
# prefix convention) so a result entry's "name" (e.g. "asr_omni_sinhala") maps
# back to the task family it belongs to.
FAMILY_DIR_PREFIXES = {
    "asr": ("asr",),
    "sid": ("sid",),
    "er": ("er_",),
    "sd": ("sd_",),
    "asv": ("asv",),
    "ic": ("ic_",),
}

# The TEST metric reported per family (select_checkpoint.py ranks on the dev score instead).
PRIMARY_METRIC = {
    "asr": "wer",
    "sid": "accuracy",
    "er": "accuracy",
    "asv": "eer",
    "sd": "der",
    "ic": "accuracy",
}


def _family_of(entry_name: str) -> str | None:
    for family, prefixes in FAMILY_DIR_PREFIXES.items():
        if entry_name.startswith(prefixes):
            return family
    return None


def load_upstream_metrics(name: str) -> dict | None:
    """Loads reports/<name>/bench/metrics.json (a run, or a frozen baseline) and aggregates
    mean/std of the primary metric per task family. Returns {family: {"mean":.., "std":..}}
    or None if that upstream hasn't been benchmarked yet."""
    metrics_path = Path("reports") / name / "bench" / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        summary = json.load(f)

    by_family: dict[str, list[float]] = {}
    for entry in summary.get("results", []):
        if entry.get("status") != "ok":
            continue
        family = _family_of(entry.get("name", ""))
        if family is None:
            continue
        metric_name = PRIMARY_METRIC.get(family)
        if metric_name is None:
            continue
        value = entry.get("metrics", {}).get(metric_name)
        if value is not None:
            by_family.setdefault(family, []).append(value)

    return {
        family: {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for family, values in by_family.items()
    }


def build_report(raw: dict, params) -> tuple[Path, Path]:
    """reports/<run>/results.md and results_table.csv: the run against the frozen baselines
    in benchmark.baselines (model keys; each is that model's base, <model>-frozen, benchmarked
    with `make bench FROZEN=1`). A row with no results shows "—", never a made-up number."""
    from pipeline.eval.benchmark import frozen_name

    bench_cfg = raw.get("benchmark", {})
    tasks = bench_cfg.get("tasks", "asr,sid").split(",")
    rows = [params.run] + [frozen_name(key) for key in bench_cfg.get("baselines", [])]
    results = {name: load_upstream_metrics(name) for name in rows}

    lines = ["# Benchmark Results\n", f"Run: {params.run}\n", f"Tasks: {', '.join(tasks)}\n"]
    lines.append(f"Seeds: {bench_cfg.get('seeds', 'N/A')}\n\n")
    lines.append("| Upstream |" + "".join(f" {t.upper()} |" for t in tasks))
    lines.append("|---|" + "---|" * len(tasks))
    for name in rows:
        metrics = results.get(name) or {}
        cells = [
            f" {metrics[t]['mean']:.3f} ± {metrics[t]['std']:.3f} |" if t in metrics else " — |"
            for t in tasks
        ]
        lines.append(f"| {name} |" + "".join(cells))
    lines.append("\n")
    md_content = "\n".join(lines)

    out_dir = params.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = out_dir / "results.md", out_dir / "results_table.csv"
    md_path.write_text(md_content)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["upstream"]
        for t in tasks:
            header += [f"{t.upper()}_mean", f"{t.upper()}_std"]
        w.writerow(header)
        for name in rows:
            metrics = results.get(name) or {}
            row = [name]
            for t in tasks:
                row += [metrics[t]["mean"], metrics[t]["std"]] if t in metrics else ["", ""]
            w.writerow(row)
    print(f"Results written to {md_path} and {csv_path}")
    print(md_content)
    return md_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    raw, params = load_params(args.config, variants_arg(args.variant), args.tag)
    build_report(raw, params)


if __name__ == "__main__":
    main()
