"""Run the SLSB benchmark for one run (`make bench`).

Reads benchmark.* from the params file and calls scripts/run_benchmark.sh. By default it
benchmarks this run's selected checkpoint, models/selected-<run>, and writes
reports/bench/<run>/metrics.json, so benchmarking two experiments (or two models)
never overwrites each other. `--upstream` benchmarks any other checkpoint directory or
Hugging Face id instead; its results go to reports/bench/<name>.

  python -m pipeline.eval.benchmark --config params-wavlm.yaml --experiment si-100h
  python -m pipeline.eval.benchmark --config params-wavlm.yaml \
      --upstream models/wavlm-large-si-ta-200h/checkpoint-9000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline.common.schema import load_raw_params


def bench_name(upstream: str) -> str:
    """Folder name under reports/bench/ for an arbitrary upstream (path or HF id)."""
    parts = [p for p in Path(upstream).parts if p not in ("models", ".", "..", "/")]
    return "-".join(parts) if parts else upstream.replace("/", "__")


def require_complete_pretraining(run: str, allow_incomplete: bool = False) -> None:
    """Refuse to benchmark a run whose pre-training has not finished, so a half-trained
    checkpoint is never scored and logged as a result. reports/pretrain_metrics-<run>.json is
    written with completed=true only when training reaches max_updates. --allow-incomplete (or
    an explicit --upstream, handled by the caller) overrides this."""
    if allow_incomplete:
        return
    path = Path(f"reports/pretrain_metrics-{run}.json")
    if not path.is_file():
        raise SystemExit(
            f"refusing to benchmark {run}: {path} not found -- pre-training has not finished "
            f"(run `make train` to completion first, or pass --allow-incomplete)."
        )
    metrics = json.loads(path.read_text())
    if not metrics.get("completed"):
        raise SystemExit(
            f"refusing to benchmark {run}: pre-training is incomplete "
            f"({path}: final_step={metrics.get('final_step')}, completed=false) -- finish "
            f"`make train` first, or pass --allow-incomplete."
        )


def failed_tasks(metrics_path: Path) -> list[str]:
    """(task/language/seed) entries slsb did not finish. slsb exits 0 even when individual
    tasks error, so without this a partial result would be copied and logged as if complete."""
    summary = json.loads(metrics_path.read_text())
    return [
        f"{e.get('task')}/{e.get('language')}/seed{e.get('seed')}"
        for e in summary.get("results", [])
        if e.get("status") != "ok"
    ]


def configured_name(bench: dict, upstream: str) -> str | None:
    """The `benchmark.upstreams` key whose `path` is this upstream, if any.

    report.py looks a result up by that key (reports/bench/<key>/metrics.json), so a
    configured baseline has to land there or it renders as "—" even after it was run.
    """
    for key, cfg in (bench.get("upstreams") or {}).items():
        if isinstance(cfg, dict) and cfg.get("path") == upstream:
            return key
    return None


def resolve_target(
    raw: dict, upstream: str | None = None, name: str | None = None
) -> tuple[str, str]:
    """(upstream to benchmark, folder name under reports/bench/).

    No --upstream means this run's own selected checkpoint, named after the run. For any
    other upstream, prefer the benchmark.upstreams key that points at it, so baselines land
    where report.py reads them; fall back to a name derived from the upstream itself.
    """
    bench = raw.get("benchmark", {})
    if upstream is None:
        return f"models/selected-{raw['run_name']}", name or raw["run_name"]
    return upstream, name or configured_name(bench, upstream) or bench_name(upstream)


def build_command(raw: dict, upstream: str | None = None, name: str | None = None) -> list[str]:
    bench = raw.get("benchmark", {})
    upstream, name = resolve_target(raw, upstream, name)
    cmd = [
        "bash",
        "scripts/run_benchmark.sh",
        upstream,
        str(bench["tasks"]),
        str(bench["seeds"]),
        f"reports/bench/{name}",
        bench.get("mlflow_uri", ""),
        bench.get("slsb_data_dir", "../SLSB-benchmark/data"),
    ]
    if bench.get("mlflow_uri"):  # name and tag the MLflow runs after the run
        cmd += [name, f"model={raw['model_key']},experiment={raw['experiment']}"]
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="params-mhubert147.yaml")
    parser.add_argument("--experiment", default=None, help="run name; overrides `experiment:`")
    parser.add_argument("--upstream", default=None, help="benchmark this checkpoint instead")
    parser.add_argument(
        "--name", default=None, help="reports/bench/<name> (default: from run/upstream)"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="benchmark the run's own checkpoint even if its pre-training did not finish",
    )
    args = parser.parse_args(argv)
    raw = load_raw_params(args.config, args.experiment)

    # Only the run's OWN selected checkpoint is gated on finished pre-training; an explicit
    # --upstream (a baseline HF id, or some other checkpoint) is the user's deliberate choice.
    if args.upstream is None:
        require_complete_pretraining(raw["run_name"], args.allow_incomplete)

    cmd = build_command(raw, args.upstream, args.name)
    _, name = resolve_target(raw, args.upstream, args.name)
    print("+", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    # slsb exits 0 even when some tasks error -- don't let a partial benchmark pass as complete.
    metrics_path = Path(f"reports/bench/{name}/metrics.json")
    if metrics_path.is_file():
        failed = failed_tasks(metrics_path)
        if failed:
            print(
                f"ERROR: SLSB left {len(failed)} task(s) unfinished for {name}: "
                f"{', '.join(failed)} (see {metrics_path}) -- the benchmark is partial.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
