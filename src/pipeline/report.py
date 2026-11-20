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

from pipeline.common.config import MODEL_CONFIG_DIR, REPO_ROOT, load_raw

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

# The one number reported per family (SLSB's primary metric; lower is better for wer, eer, der).
PRIMARY_METRIC = {
    "asr": "wer",
    "sid": "accuracy",
    "er": "accuracy",
    "sd": "der",
    "asv": "eer",
    "ic": "accuracy",
}


def _family_of(entry_name: str) -> str | None:
    for family, prefixes in FAMILY_DIR_PREFIXES.items():
        if entry_name.startswith(prefixes):
            return family
    return None


UNKNOWN_PROTOCOL = "unknown"


def load_upstream_summary(name: str) -> dict | None:
    """reports/bench/<name>/metrics.json as {"protocol": ..., "metrics": {family: {mean, std}}},
    or None if that upstream hasn't been benchmarked. Results written before slsb v0.3.1 do not
    record their protocol; it is then "unknown", never guessed."""
    metrics_path = Path(f"reports/bench/{name}/metrics.json")
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        summary = json.load(f)
    return {
        "protocol": summary.get("protocol") or UNKNOWN_PROTOCOL,
        "metrics": _aggregate(summary),
    }


def load_upstream_metrics(name: str) -> dict | None:
    """Mean/std of the primary metric per task family: {family: {"mean":.., "std":..}}, or None
    if that upstream hasn't been benchmarked yet."""
    summary = load_upstream_summary(name)
    return None if summary is None else summary["metrics"]


def _aggregate(summary: dict) -> dict:

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


def load_report_config(config_path: str = "params.yaml", all_models: bool = False) -> dict:
    """params.yaml's config; with all_models, every model's benchmark.upstreams too, so
    one table compares all of them (each model's rows come from its own overlay)."""
    cfg = load_raw(config_path)
    if all_models:
        upstreams = dict(cfg.get("benchmark", {}).get("upstreams", {}))
        for overlay in sorted((REPO_ROOT / MODEL_CONFIG_DIR).glob("*.yaml")):
            extra = load_raw(config_path, overlay=str(overlay)).get("benchmark", {})
            upstreams.update(extra.get("upstreams", {}))
        cfg.setdefault("benchmark", {})["upstreams"] = upstreams
    return cfg


def build_report(
    config_path: str = "params.yaml", all_models: bool = False, protocol: str | None = None
):
    """Write reports/results.md and results_table.csv.

    Only upstreams that have results are rows (the others are listed under the table, never shown
    as blanks). Every row carries the SLSB protocol its numbers were produced under, because
    scores from different protocol versions are not comparable; `protocol` keeps only that one.
    """
    cfg = load_report_config(config_path, all_models)

    bench_cfg = cfg.get("benchmark", {})
    upstreams = bench_cfg.get("upstreams", {})
    tasks = bench_cfg.get("tasks", "asr,sid").split(",")

    summaries = {name: load_upstream_summary(name) for name in upstreams}
    not_benchmarked = [n for n, summary in summaries.items() if summary is None]
    rows = {n: summary for n, summary in summaries.items() if summary is not None}
    skipped_protocol = []
    if protocol:
        skipped_protocol = [n for n, summary in rows.items() if summary["protocol"] != protocol]
        rows = {n: summary for n, summary in rows.items() if summary["protocol"] == protocol}
    protocols = sorted({summary["protocol"] for summary in rows.values()})

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Benchmark Results\n"]
    lines.append(f"Tasks: {', '.join(tasks)}\n")
    lines.append(f"Seeds: {bench_cfg.get('seeds', 'N/A')}\n")
    if len(protocols) > 1:
        lines.append(
            f"> **Warning:** these rows come from {len(protocols)} different SLSB protocol versions "
            f"({', '.join(protocols)}). Scores from different protocol versions are not "
            f"comparable; compare rows with the same protocol, or pass --protocol.\n"
        )
    lines.append("")

    header = "| Upstream | Protocol |" + "".join(f" {t.upper()} |" for t in tasks)
    lines += [header, "|---|---|" + "---|" * len(tasks)]
    for name, summary in rows.items():
        row = f"| {upstreams[name].get('name', name)} | {summary['protocol']} |"
        for task in tasks:
            m = summary["metrics"].get(task)
            row += f" {m['mean']:.3f} ± {m['std']:.3f} |" if m else " — |"
        lines.append(row)
    lines.append("")
    if not_benchmarked:
        lines.append("Not benchmarked yet: " + ", ".join(not_benchmarked) + "\n")
    if skipped_protocol:
        lines.append(f"Left out (not protocol {protocol}): " + ", ".join(skipped_protocol) + "\n")

    md_content = "\n".join(lines)
    md_path = out_dir / "results.md"
    csv_path = out_dir / "results_table.csv"
    md_path.write_text(md_content)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header_row = ["upstream", "protocol"]
        for t in tasks:
            header_row += [f"{t.upper()}_mean", f"{t.upper()}_std"]
        w.writerow(header_row)
        for name, summary in rows.items():
            row_data = [upstreams[name].get("name", name), summary["protocol"]]
            for task in tasks:
                m = summary["metrics"].get(task)
                row_data += [m["mean"], m["std"]] if m else ["", ""]
            w.writerow(row_data)

    print(f"Results written to {md_path} and {csv_path}")
    print(md_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="include every model's benchmark upstreams (configs/models/*.yaml)",
    )
    parser.add_argument(
        "--protocol", default=None, help="only rows produced under this SLSB protocol, e.g. v0.5"
    )
    args = parser.parse_args()
    build_report(args.config, args.all_models, args.protocol)


if __name__ == "__main__":
    main()
