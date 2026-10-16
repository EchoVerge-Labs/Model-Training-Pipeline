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


def load_upstream_metrics(name: str, language: str | None = None) -> dict | None:
    """Loads reports/bench/<name>/metrics.json and aggregates mean/std of the
    primary metric per task family. Returns {family: {"mean":.., "std":..}} or
    None if that upstream hasn't been benchmarked yet.

    With `language`, only that language's entries count -- plus the "multilingual"
    ones (sid), which have no per-language score and are shown under both."""
    metrics_path = Path(f"reports/bench/{name}/metrics.json")
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        summary = json.load(f)

    by_family: dict[str, list[float]] = {}
    for entry in summary.get("results", []):
        if entry.get("status") != "ok":
            continue
        if language and entry.get("language") not in (language, "multilingual"):
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


def _read_json(path: str | Path | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path and Path(path).is_file() else None


def _row_metadata(row: dict) -> tuple[str, str]:
    """(selected training hours, checkpoint) for an ablation row, "—" where unknown.
    Single-language rows are looked up under reports/<experiment>/; the joint and
    frozen rows name their files (`stats`, `checkpoint_report`) or have none."""
    exp = row.get("experiment")
    stats = _read_json(row.get("stats") or (f"reports/{exp}/selection_stats.json" if exp else ""))
    ckpt = _read_json(
        row.get("checkpoint_report") or (f"reports/{exp}/checkpoint_selection.json" if exp else "")
    )
    hours = f"{sum(stats['hours_by_language'].values()):.1f}" if stats else "—"
    return hours, ckpt["best_checkpoint"] if ckpt else "—"


def ablation_table(cfg: dict) -> tuple[list[str], list[list[str]]] | None:
    """Rows comparing the frozen backbone, the joint model and the single-language models.

    One line per (model, benchmark language). A single-language model is shown on its own
    language only, unless benchmark.cross_language is on, which adds the other language.
    Anything not benchmarked yet is "—". Returns (header, rows) or None if not configured."""
    bench = cfg.get("benchmark", {})
    rows_cfg = bench.get("ablation_rows")
    if not rows_cfg:
        return None
    tasks = bench.get("tasks", "asr,sid").split(",")
    languages = bench.get("ablation_languages", ["sinhala", "tamil"])
    cross = bench.get("cross_language", True)
    backbone = (cfg.get("experiments") or {}).get("backbone_tag", "—")

    header = [
        "Backbone",
        "Adaptation",
        "Train language",
        "Selected hours",
        "Checkpoint",
        "Benchmark language",
        *[t.upper() for t in tasks],
    ]
    rows = []
    for row in rows_cfg:
        hours, checkpoint = _row_metadata(row)
        train_lang = row.get("train_language", "—")
        own = train_lang if train_lang in languages else None
        bench_langs = [own] + [lg for lg in languages if lg != own and cross] if own else languages
        for lang in bench_langs:
            metrics = load_upstream_metrics(row["upstream"], lang)
            cells = [
                f"{metrics[t]['mean']:.3f} ± {metrics[t]['std']:.3f}"
                if metrics and t in metrics
                else "—"
                for t in tasks
            ]
            rows.append(
                [
                    row.get("backbone", backbone),
                    row["mode"],
                    train_lang,
                    hours,
                    checkpoint,
                    lang + (" (own)" if lang == own else ""),
                    *cells,
                ]
            )
    return header, rows


def write_ablation(cfg: dict, out_dir: Path) -> None:
    table = ablation_table(cfg)
    if table is None:
        return
    header, rows = table
    lines = ["# Joint vs single-language CoPT\n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines.append(
        "\nSID has no per-language score (multilingual), so it repeats under both languages; "
        "ER exists for Tamil only.\n"
    )
    (out_dir / "results_ablation.md").write_text("\n".join(lines))
    with open(out_dir / "results_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Ablation table written to {out_dir / 'results_ablation.md'}")


def build_report(config_path: str = "params.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bench_cfg = cfg.get("benchmark", {})
    upstreams = bench_cfg.get("upstreams", {})
    tasks = bench_cfg.get("tasks", "asr,sid").split(",")

    results = {name: load_upstream_metrics(name) for name in upstreams}

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

    md_content = "\n".join(lines)
    md_path = out_dir / "results.md"
    csv_path = out_dir / "results_table.csv"

    with open(md_path, "w") as f:
        f.write(md_content)

    # Also write CSV (one column per task with mean, one with std)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header_row = ["upstream"]
        for t in tasks:
            header_row += [f"{t.upper()}_mean", f"{t.upper()}_std"]
        w.writerow(header_row)
        for name, upstream_cfg in upstreams.items():
            display_name = upstream_cfg.get("name", name)
            row_data = [display_name]
            metrics = results.get(name)
            for task in tasks:
                if metrics and task in metrics:
                    row_data += [metrics[task]["mean"], metrics[task]["std"]]
                else:
                    row_data += ["", ""]
            w.writerow(row_data)

    write_ablation(cfg, out_dir)
    print(f"Results written to {md_path} and {csv_path}")
    print(md_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    build_report(args.config)


if __name__ == "__main__":
    main()
