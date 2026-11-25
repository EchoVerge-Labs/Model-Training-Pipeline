"""Run the SLSB benchmark for one run and log it to the SLSB leaderboard (`make bench`).

This only drives the SLSB-benchmark repository -- its own `slsb run` and its own
`scripts/log_results.py` -- and adds no benchmark logic of its own:

1. `slsb run` over `benchmark.tasks` x `benchmark.seeds`, WITHOUT an MLflow URI: live logging
   would make one MLflow run per task and seed, which are not leaderboard rows.
2. slsb's results_<upstream>.json is copied to reports/bench/<folder>/metrics.json, the fixed
   path report.py and dvc.yaml read.
3. With DAGSHUB_USER / DAGSHUB_TOKEN set, log_results.py logs the folder ONCE, as one MLflow run
   per model in the `slsb-<protocol>` experiment of SLSB-benchmark.mlflow (the leaderboard),
   with kind, family, base model, checkpoint step and hours of continued pre-training. Without
   the credentials the results stay local and the exact logging command is printed.

  python -m pipeline.eval.benchmark --config params-wavlm.yaml               # this run's model
  python -m pipeline.eval.benchmark --config params-wavlm.yaml --frozen      # its base model
  python -m pipeline.eval.benchmark --config params-wavlm.yaml \\
      --upstream models/wavlm-large-si-ta-200h/checkpoint-9000 --name wavlm-large-copt200h-c9000

By default it benchmarks this run's selected checkpoint (models/selected-<run>) under the run's
model name (e.g. wavlm-large-copt200h, the name of its ssl-pretraining run too), into
reports/bench/<run>/. `--upstream` benchmarks any other checkpoint or Hugging Face id; it is
logged only with an explicit `--name`, so an ad-hoc run can never overwrite a model's row.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.common.schema import load_raw_params
from pipeline.eval.slsb import (
    SlsbNotFound,
    log_results_script,
    require_version,
    slsb_command,
    slsb_python,
)
from pipeline.models.registry import MODEL_REGISTRY, get_model_spec

LEADERBOARD_URI = "https://dagshub.com/EchoVerge-LABS/SLSB-benchmark.mlflow"


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


def selected_checkpoint_step(run: str) -> str:
    """The step of the checkpoint `make ckpt` chose for this run, e.g. "9000"; "" if none."""
    report = Path(f"reports/checkpoint_selection-{run}.json")
    if not report.is_file():
        return ""
    return json.loads(report.read_text())["best_checkpoint"].removeprefix("checkpoint-")


@dataclass(frozen=True)
class Plan:
    """One benchmark: what is run, where its results go, and its leaderboard row."""

    upstream: str  # a checkpoint directory, or a Hugging Face id
    out_dir: Path
    name: str | None  # the leaderboard row (None: an ad-hoc upstream, results stay local)
    kind: str  # "adapted" or "frozen"
    family: str
    base_model: str
    checkpoint: str  # training step of the checkpoint ("" for a frozen model)
    pretrain_hours: str  # "" for a frozen model
    tasks: str
    seeds: str
    data_dir: str
    slsb_params: Path
    slsb_version: str
    leaderboard_uri: str


def make_plan(raw: dict, args: argparse.Namespace) -> Plan:
    bench = raw["benchmark"]
    spec = get_model_spec(raw["model_key"])
    base_model = raw["pretrain"]["base_model"]
    if args.frozen:
        upstream, folder = resolve_target(raw, base_model, args.out)
        name, kind, checkpoint, hours = args.name or spec.slsb_name, "frozen", "", ""
    elif args.upstream:
        upstream, folder = resolve_target(raw, args.upstream, args.out)
        is_base = not Path(args.upstream).exists()  # an HF id, not a checkpoint directory
        kind = "frozen" if is_base else "adapted"
        if is_base:  # another model's base (e.g. utter-project/mHuBERT-147): its own family
            spec = next((s for s in MODEL_REGISTRY.values() if s.base_model == args.upstream), spec)
            base_model = args.upstream
        step = re.search(r"checkpoint-(\d+)", args.upstream)
        checkpoint = args.checkpoint or (step.group(1) if step and not is_base else "")
        hours = "" if is_base else raw["pretrain_hours"]
        name = args.name  # never inherit the run's own row
    else:
        upstream, folder = resolve_target(raw, None, args.out)
        name, kind = args.name or raw["model_name"], "adapted"
        checkpoint = args.checkpoint or selected_checkpoint_step(raw["run_name"])
        hours = raw["pretrain_hours"]
    data_dir = bench.get("slsb_data_dir", "../SLSB-benchmark/data")
    return Plan(
        upstream=upstream,
        out_dir=Path("reports") / "bench" / folder,
        name=name,
        kind=kind,
        family=spec.slsb_family,
        base_model=base_model,
        checkpoint=checkpoint,
        pretrain_hours=hours,
        tasks=str(bench["tasks"]),
        seeds=str(bench["seeds"]),
        data_dir=data_dir,
        # slsb reads its head hyperparameters from the params.yaml next to its data; this
        # repo's own ./params*.yaml are the pipeline's
        slsb_params=Path(data_dir).parent / "params.yaml",
        slsb_version=str(bench["slsb_version"]),
        leaderboard_uri=args.leaderboard_uri or LEADERBOARD_URI,
    )


def slsb_run_command(plan: Plan, slsb: list[str]) -> list[str]:
    """`slsb run` without --mlflow-uri: the results are logged once, afterwards."""
    return [
        *slsb,
        "run",
        "--upstream",
        plan.upstream,
        "--tasks",
        plan.tasks,
        "--seeds",
        plan.seeds,
        "--out",
        str(plan.out_dir),
        "--data-dir",
        plan.data_dir,
        "--params",
        str(plan.slsb_params),
    ]


def log_results_command(
    plan: Plan,
    python: str,
    script: str,
    note: str = "",
    replace: bool = False,
    carry_over: bool = False,
) -> list[str]:
    cmd = [
        python,
        script,
        str(plan.out_dir),
        "--name",
        plan.name,
        "--kind",
        plan.kind,
        "--family",
        plan.family,
        "--base-model",
        plan.base_model,
        "--mlflow-uri",
        plan.leaderboard_uri,
    ]
    if plan.checkpoint:
        cmd += ["--checkpoint", plan.checkpoint]
    if plan.pretrain_hours:
        cmd += ["--pretrain-hours", plan.pretrain_hours]
    if note:
        cmd += ["--note", note]
    if replace:
        cmd.append("--replace")
    if carry_over:
        cmd.append("--carry-over")
    return cmd


def copy_metrics(plan: Plan) -> Path:
    """<out>/results_<upstream-with-/-as-__>.json -> <out>/metrics.json"""
    results = plan.out_dir / f"results_{plan.upstream.replace('/', '__')}.json"
    if not results.is_file():
        raise SystemExit(f"ERROR: slsb did not write the expected results file {results}")
    target = plan.out_dir / "metrics.json"
    shutil.copyfile(results, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="params-mhubert147.yaml")
    parser.add_argument("--experiment", default=None, help="run name; overrides `experiment:`")
    parser.add_argument("--frozen", action="store_true", help="benchmark the run's base model")
    parser.add_argument("--upstream", default=None, help="benchmark this checkpoint / HF id")
    parser.add_argument(
        "--name", default=None, help="leaderboard row (default: the run's model name)"
    )
    parser.add_argument("--out", default=None, help="reports/bench/<out> (default: from the run)")
    parser.add_argument("--checkpoint", default=None, help="training step of the checkpoint")
    parser.add_argument("--note", default="", help="free text shown on the leaderboard run")
    parser.add_argument("--leaderboard-uri", default=None, help="MLflow URI of the leaderboard")
    parser.add_argument("--no-log", action="store_true", help="keep the results local")
    parser.add_argument(
        "--replace", action="store_true", help="replace the model's leaderboard row"
    )
    parser.add_argument(
        "--carry-over",
        action="store_true",
        help="copy tasks this protocol did not change from the previous protocol's row",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="benchmark the run's own checkpoint even if its pre-training did not finish",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args(argv)
    if args.frozen and args.upstream:
        parser.error("--frozen benchmarks the base model; don't also give --upstream")
    raw = load_raw_params(args.config, args.experiment)
    plan = make_plan(raw, args)

    try:
        slsb = slsb_command()
        script = str(log_results_script())
        if not args.dry_run:
            print(f"slsb {require_version(plan.slsb_version, slsb)} (= benchmark.slsb_version)")
    except SlsbNotFound as e:
        raise SystemExit(f"ERROR: {e}") from e

    run_cmd = slsb_run_command(plan, slsb)
    # log_results.py needs mlflow: slsb's own environment if it has one, else this one
    python = slsb_python() or sys.executable
    log_cmd = (
        log_results_command(plan, python, script, args.note, args.replace, args.carry_over)
        if plan.name
        else None
    )
    print(f"[{plan.name or plan.out_dir.name}] ({plan.kind}) tasks={plan.tasks} seeds={plan.seeds}")
    print("+ " + " ".join(run_cmd), flush=True)
    if args.dry_run:
        if log_cmd:
            print("+ " + " ".join(log_cmd))
        return 0

    # Only the run's OWN selected checkpoint is gated on finished pre-training; --frozen and
    # an explicit --upstream are the user's deliberate choice.
    if not (args.frozen or args.upstream):
        require_complete_pretraining(raw["run_name"], args.allow_incomplete)
        if not Path(plan.upstream).is_dir():
            raise SystemExit(f"ERROR: {plan.upstream} does not exist -- run `make ckpt` first")

    rc = subprocess.call(run_cmd)
    if rc != 0:
        return rc
    metrics_path = copy_metrics(plan)
    print(f"Copied results -> {metrics_path}")

    # slsb exits 0 even when some tasks error -- a partial result must not become a row
    failed = failed_tasks(metrics_path)
    if failed:
        print(
            f"ERROR: SLSB left {len(failed)} task(s) unfinished for {plan.out_dir.name}: "
            f"{', '.join(failed)} (see {metrics_path}) -- the benchmark is partial and was NOT "
            f"logged to the leaderboard.",
            file=sys.stderr,
        )
        return 1

    if log_cmd is None:
        print("NOTE: an ad-hoc --upstream without --name is not logged to the leaderboard.")
        return 0
    if args.no_log:
        print("Not logged (--no-log). To log it:\n  " + " ".join(log_cmd))
        return 0
    if not (os.environ.get("DAGSHUB_USER") and os.environ.get("DAGSHUB_TOKEN")):
        print(
            "WARNING: DAGSHUB_USER / DAGSHUB_TOKEN not set, so the results were NOT logged to "
            "the leaderboard. To log them later:\n  " + " ".join(log_cmd)
        )
        return 0
    return subprocess.call(log_cmd)


if __name__ == "__main__":
    sys.exit(main())
