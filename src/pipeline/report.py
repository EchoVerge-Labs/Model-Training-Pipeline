"""Aggregate benchmark results into a comparison table.

Reads reports/bench/<upstream>/metrics.json for each upstream in
params.yaml's benchmark.upstreams -- that file is a copy of slsb's own
results_<upstream>.json (see scripts/run_benchmark.sh), a list of
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

import yaml

# Mirrors slsb.tasks.FAMILY_DIR_PREFIXES (SLSB-benchmark's own family->dirname
# prefix convention) so a result entry's "name" (e.g. "asr_omni_sinhala") maps
# back to the task family it belongs to.
FAMILY_DIR_PREFIXES = {
    "asr": ("asr",),
    "sid": ("sid",),
    "er": ("er_",),
    "sd": ("sd_",),
    "asv": ("asv",),
}

# Mirrors select_checkpoint.py's PRIMARY_METRIC.
PRIMARY_METRIC = {"asr": "wer", "sid": "accuracy", "er": "accuracy", "asv": "eer"}


def _family_of(entry_name: str) -> str | None:
    for family, prefixes in FAMILY_DIR_PREFIXES.items():
        if entry_name.startswith(prefixes):
            return family
    return None


def load_upstream_metrics(name: str) -> dict | None:
    """Loads reports/bench/<name>/metrics.json and aggregates mean/std of the
    primary metric per task family. Returns {family: {"mean":.., "std":..}} or
    None if that upstream hasn't been benchmarked yet."""
    metrics_path = Path(f"reports/bench/{name}/metrics.json")
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


# Adaptation modes the report compares against. Labels stay stable as more runs
# (si_100h_only, ta_100h_only, ...) are added.
FROZEN_MODE = "frozen"
BASELINE_MODE = "joint_200h_mixed_batch"


def _mode_reference(upstreams: dict, mode: str) -> str | None:
    return next((n for n, c in upstreams.items() if c.get("adaptation_mode") == mode), None)


def compute_deltas(results: dict, upstreams: dict, tasks: list[str]) -> dict[str, dict[str, dict]]:
    """{upstream: {"vs_frozen"|"vs_joint_mixed": {task: signed mean difference}}}.

    Only for upstreams that have a result for both sides; nothing is filled in for
    a missing one. Differences are in the metric's own units (lower WER/EER and
    higher accuracy are better), not normalised."""
    out: dict[str, dict[str, dict]] = {}
    for key, mode in (("vs_frozen", FROZEN_MODE), ("vs_joint_mixed", BASELINE_MODE)):
        ref = _mode_reference(upstreams, mode)
        ref_metrics = results.get(ref) if ref else None
        for name in upstreams:
            metrics = results.get(name)
            if not ref_metrics or not metrics or name == ref:
                continue
            deltas = {
                t: metrics[t]["mean"] - ref_metrics[t]["mean"]
                for t in tasks
                if t in metrics and t in ref_metrics
            }
            if deltas:
                out.setdefault(name, {})[key] = deltas
    return out


def build_report(config_path: str = "params.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bench_cfg = cfg.get("benchmark", {})
    upstreams = bench_cfg.get("upstreams", {})
    tasks = bench_cfg.get("tasks", "asr,sid").split(",")

    results = {name: load_upstream_metrics(name) for name in upstreams}
    deltas = compute_deltas(results, upstreams, tasks)

    # Build markdown table
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Benchmark Results\n"]
    lines.append(f"Tasks: {', '.join(tasks)}\n")
    lines.append(f"Seeds: {bench_cfg.get('seeds', 'N/A')}\n\n")

    # Header
    header = "| Upstream |"
    separator = "|---|"
    for task in tasks:
        header += f" {task.upper()} |"
        separator += "---|"
    lines.append(header)
    lines.append(separator)

    # Rows
    for name, upstream_cfg in upstreams.items():
        display_name = upstream_cfg.get("name", name)
        row = f"| {display_name} |"
        metrics = results.get(name)
        for task in tasks:
            if metrics and task in metrics:
                m = metrics[task]["mean"]
                s = metrics[task]["std"]
                row += f" {m:.3f} ± {s:.3f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("\n")

    if deltas:
        lines.append("## Change in the primary metric (mean over languages and seeds)\n")
        lines.append(
            "Signed differences in each metric's own units (WER/EER: lower is better; "
            "accuracy: higher is better). Blank where either side has not been benchmarked.\n"
        )
        head = "| Upstream | Adaptation mode | Compared with |" + "".join(
            f" Δ {t.upper()} |" for t in tasks
        )
        lines += [head, "|---|---|---|" + "---|" * len(tasks)]
        for name, by_ref in deltas.items():
            cfg_ = upstreams[name]
            for key, label in (
                ("vs_frozen", "frozen"),
                ("vs_joint_mixed", "joint 200h mixed batches"),
            ):
                if key in by_ref:
                    cells = "".join(
                        f" {by_ref[key][t]:+.3f} |" if t in by_ref[key] else " — |" for t in tasks
                    )
                    lines.append(
                        f"| {cfg_.get('name', name)} | {cfg_.get('adaptation_mode', '—')} | {label} |"
                        + cells
                    )
        lines.append("\n")

    md_content = "\n".join(lines)
    md_path = out_dir / "results.md"
    csv_path = out_dir / "results_table.csv"

    with open(md_path, "w") as f:
        f.write(md_content)

    # Also write CSV (one column per task with mean, one with std)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header_row = ["upstream", "adaptation_mode"]
        for t in tasks:
            header_row += [f"{t.upper()}_mean", f"{t.upper()}_std"]
        for key in ("vs_frozen", "vs_joint_mixed"):
            header_row += [f"{t.upper()}_delta_{key}" for t in tasks]
        w.writerow(header_row)
        for name, upstream_cfg in upstreams.items():
            display_name = upstream_cfg.get("name", name)
            row_data = [display_name, upstream_cfg.get("adaptation_mode", "")]
            metrics = results.get(name)
            for task in tasks:
                if metrics and task in metrics:
                    row_data += [metrics[task]["mean"], metrics[task]["std"]]
                else:
                    row_data += ["", ""]
            for key in ("vs_frozen", "vs_joint_mixed"):
                row_data += [deltas.get(name, {}).get(key, {}).get(t, "") for t in tasks]
            w.writerow(row_data)

    print(f"Results written to {md_path} and {csv_path}")
    print(md_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    build_report(args.config)


if __name__ == "__main__":
    main()
