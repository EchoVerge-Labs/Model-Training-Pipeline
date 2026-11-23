"""Run the SLSB benchmark for one run (`make bench`).

Reads benchmark.* from the params file and calls scripts/run_benchmark.sh. By default it
benchmarks this run's selected checkpoint, models/selected-<run>, and writes
reports/bench/<run>/metrics.json, so benchmarking two experiments (or two models)
never overwrites each other. `--upstream` benchmarks any other checkpoint directory or
Hugging Face id instead; its results go to reports/bench/<name>.

  python -m pipeline.benchmark --config params-wavlm.yaml --experiment si-100h
  python -m pipeline.benchmark --config params-wavlm.yaml \
      --upstream models/wavlm-large-si-ta-200h/checkpoint-9000
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline.schema import load_raw_params


def bench_name(upstream: str) -> str:
    """Folder name under reports/bench/ for an arbitrary upstream (path or HF id)."""
    parts = [p for p in Path(upstream).parts if p not in ("models", ".", "..", "/")]
    return "-".join(parts) if parts else upstream.replace("/", "__")


def build_command(raw: dict, upstream: str | None = None, name: str | None = None) -> list[str]:
    bench = raw.get("benchmark", {})
    run = raw["run_name"]
    if upstream is None:
        upstream = f"models/selected-{run}"
        name = name or run
    name = name or bench_name(upstream)
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
    args = parser.parse_args(argv)
    raw = load_raw_params(args.config, args.experiment)
    cmd = build_command(raw, args.upstream, args.name)
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
