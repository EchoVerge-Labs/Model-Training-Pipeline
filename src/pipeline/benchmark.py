"""Benchmark a pre-training run against SLSB.

This only drives the SLSB-benchmark repository -- its own `slsb run` and its own
`scripts/log_results.py` -- and adds no benchmark logic of its own.

  # one of the configured models: its selected checkpoint (make ckpt MODEL=wavlm first)
  python -m pipeline.benchmark --model wavlm
  python -m pipeline.benchmark --model wavlm --frozen        # its pre-trained base: the baseline row

  # any other run, e.g. an experiment checkpoint
  python -m pipeline.benchmark --upstream models/xlsr300m-si-100h/checkpoint-6500 \\
      --name xlsr300m-copt-si-100h --family xlsr --base-model facebook/wav2vec2-xls-r-300m \\
      --checkpoint 6500

  --dry-run prints the commands and runs nothing; --no-log keeps the results local.

What follows from using SLSB's own protocol:
  * every task family in `benchmark.tasks`, 3 seeds, on the data in `benchmark.slsb_data_dir`;
  * only against the slsb version `benchmark.slsb_version` names (scores from different protocol
    versions are not comparable);
  * results are logged AFTER the run, as ONE MLflow run per model in the experiment
    slsb-<protocol> -- the leaderboard -- not one run per task and seed.
slsb writes <out>/results_<upstream>.json; it is copied to <out>/metrics.json, the fixed path
dvc.yaml and report.py read. Logging needs DAGSHUB_USER and DAGSHUB_TOKEN (see .env.example).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.common.checkpoints import require_pretraining_complete
from pipeline.common.config import add_config_args, load_params, load_raw
from pipeline.slsb import (
    SlsbNotFound,
    log_results_script,
    require_version,
    slsb_command,
    slsb_python,
)

LEADERBOARD_URI = "https://dagshub.com/EchoVerge-LABS/SLSB-benchmark.mlflow"


@dataclass(frozen=True)
class Plan:
    """Everything one benchmark run needs."""

    upstream: str  # a checkpoint directory, or a Hugging Face id for a frozen baseline
    out_dir: Path
    name: str  # the run's name in the leaderboard
    kind: str  # "adapted" or "frozen"
    family: str
    base_model: str
    checkpoint: str  # training step of the checkpoint ("" for a frozen model)
    pretrain_hours: str
    tasks: str
    seeds: str
    data_dir: str
    slsb_params: Path
    slsb_version: str
    leaderboard_uri: str


def selected_checkpoint_step(reports_dir: Path) -> str:
    """The step of the checkpoint `make ckpt` chose, e.g. "9000"; "" if there is no report."""
    report = reports_dir / "checkpoint_selection.json"
    if not report.is_file():
        return ""
    return json.loads(report.read_text())["best_checkpoint"].removeprefix("checkpoint-")


def _hours(value) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def make_plan(raw: dict, reports_dir: str, args: argparse.Namespace) -> Plan:
    """The run's settings: from the model's config, with every field overridable on the command line."""
    bench = raw["benchmark"]
    model = raw.get("model", {})
    frozen = args.frozen
    if args.upstream and not args.name:
        # an ad-hoc run must not inherit another model's name: that is how rows get overwritten
        raise SystemExit("ERROR: --upstream needs --name (and --family) for the leaderboard row")
    kind = "frozen" if frozen else "adapted"
    base_model = args.base_model or raw["pretrain"]["base_model"]
    upstream = args.upstream or (
        base_model if frozen else raw.get("checkpoint_selection", {}).get("output_dir")
    )
    name = args.name or (model.get("frozen_name") if frozen else model.get("run_name"))
    family = args.family or model.get("family")
    missing = [
        flag
        for flag, v in (("--upstream", upstream), ("--name", name), ("--family", family))
        if not v
    ]
    if missing:
        raise SystemExit(f"ERROR: no value for {', '.join(missing)}: give --model or those flags")
    checkpoint = ""
    if not frozen:
        checkpoint = args.checkpoint or selected_checkpoint_step(Path(reports_dir))
    data_dir = bench["slsb_data_dir"]
    return Plan(
        upstream=upstream,
        out_dir=Path(args.out) if args.out else Path("reports") / "bench" / name,
        name=name,
        kind=kind,
        family=family,
        base_model=base_model,
        checkpoint=checkpoint,
        pretrain_hours=""
        if frozen
        else _hours(args.pretrain_hours or raw["select"]["target_hours"]),
        tasks=str(bench["tasks"]),
        seeds=str(bench["seeds"]),
        data_dir=data_dir,
        # slsb reads its head hyperparameters from the params.yaml that sits next to its data
        slsb_params=Path(data_dir).parent / "params.yaml",
        slsb_version=str(bench["slsb_version"]),
        leaderboard_uri=args.leaderboard_uri or LEADERBOARD_URI,
    )


def slsb_run_command(plan: Plan, slsb: list[str]) -> list[str]:
    """`slsb run` without --mlflow-uri: live logging makes one MLflow run per task and seed, which are
    not leaderboard rows. The results are logged once, afterwards, by log_results_command."""
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


def log_results_command(plan: Plan, python: str, replace: bool = False) -> list[str]:
    cmd = [
        python,
        str(log_results_script()),
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
    if replace:
        cmd.append("--replace")
    return cmd


def copy_metrics(plan: Plan) -> Path:
    """<out>/results_<upstream-with-/-as-__>.json -> <out>/metrics.json"""
    results = plan.out_dir / f"results_{plan.upstream.replace('/', '__')}.json"
    if not results.is_file():
        raise SystemExit(f"ERROR: slsb did not write the expected results file {results}")
    target = plan.out_dir / "metrics.json"
    shutil.copyfile(results, target)
    return target


def failed_tasks(metrics_path: Path) -> list[str]:
    """Names of any (task, language, seed) entries slsb did not finish. slsb exits 0 even when
    individual tasks error, so a partial result would otherwise be copied and logged as if complete."""
    summary = json.loads(metrics_path.read_text())
    return [
        f"{e.get('task')}/{e.get('language')}/seed{e.get('seed')}"
        for e in summary.get("results", [])
        if e.get("status") != "ok"
    ]


def verify_selected_provenance(plan: Plan) -> None:
    """A configured adapted run benchmarks whatever is in models/selected*. Check that dir was
    produced for THIS model (make ckpt writes selection_provenance.json), so a stale selected
    dir holding another model is not logged to the leaderboard under this model's name."""
    prov_path = Path(plan.upstream) / "selection_provenance.json"
    if not prov_path.is_file():
        raise SystemExit(
            f"ERROR: {plan.upstream} has no selection_provenance.json -- it may be stale or from "
            f"another model. Re-run `make ckpt` for this model before benchmarking."
        )
    prov = json.loads(prov_path.read_text())
    if prov.get("base_model") != plan.base_model:
        raise SystemExit(
            f"ERROR: {plan.upstream} was selected for base_model {prov.get('base_model')!r}, but "
            f"this run expects {plan.base_model!r}. Re-run `make ckpt` for this model, or clear the "
            f"stale selected dir."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_config_args(parser)
    parser.add_argument(
        "--frozen", action="store_true", help="benchmark the pre-trained base model"
    )
    parser.add_argument(
        "--upstream", help="checkpoint directory or HF id (default: from the model config)"
    )
    parser.add_argument("--name", help="the run's name in the leaderboard")
    parser.add_argument("--family", help="xlsr, wav2vec2, wavlm, hubert, mhubert, ...")
    parser.add_argument("--base-model", help="the pre-trained model it started from (HF id)")
    parser.add_argument("--checkpoint", help="training step of the checkpoint")
    parser.add_argument("--pretrain-hours", help="hours of continued pre-training")
    parser.add_argument("--out", help="results directory (default: reports/bench/<name>)")
    parser.add_argument("--leaderboard-uri", help="MLflow tracking URI of the leaderboard")
    parser.add_argument("--no-log", action="store_true", help="do not log to the leaderboard")
    parser.add_argument(
        "--replace", action="store_true", help="replace the model's leaderboard run"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="benchmark even though the model's pre-training has not reached max_updates",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args(argv)

    raw = load_raw(args.config, args.model, args.overlay)
    params = load_params(args.config, args.model, args.overlay)
    reports_dir = params.model.reports_dir
    plan = make_plan(raw, reports_dir, args)

    try:
        slsb = slsb_command()
        if not args.dry_run:
            version = require_version(plan.slsb_version, slsb)
            print(f"slsb {version} (matches benchmark.slsb_version)")
    except SlsbNotFound as e:
        raise SystemExit(f"ERROR: {e}") from e

    run_cmd = slsb_run_command(plan, slsb)
    python = (
        slsb_python() or sys.executable
    )  # log_results.py needs mlflow, in slsb's own environment
    log_cmd = log_results_command(plan, python, args.replace)
    print(f"[{plan.name}] ({plan.kind}) tasks={plan.tasks} seeds={plan.seeds}")
    print("  " + " ".join(run_cmd))
    if args.dry_run:
        print("  " + " ".join(log_cmd))
        return 0
    if plan.kind == "adapted" and not Path(plan.upstream).is_dir():
        raise SystemExit(f"ERROR: {plan.upstream} does not exist -- run `make ckpt` first")

    # An adapted model from the config is benchmarked only once its pre-training has finished,
    # and only when the selected dir provably belongs to this model (an explicit --upstream is
    # an ad-hoc checkpoint of some other run, so neither is checked).
    if plan.kind == "adapted" and not args.upstream:
        if not args.allow_incomplete:
            require_pretraining_complete(reports_dir, params.pretrain.max_updates)
        verify_selected_provenance(plan)

    subprocess.run(run_cmd, check=True)
    metrics_path = copy_metrics(plan)
    print(f"Copied results -> {metrics_path}")

    # slsb exits 0 even when some tasks error; a partial result must not become a leaderboard row
    failed = failed_tasks(metrics_path)
    if failed:
        raise SystemExit(
            f"ERROR: {len(failed)} benchmark task(s) did not complete: {', '.join(failed)}.\n"
            f"Results are in {plan.out_dir}, but they were NOT logged (the row would be partial). "
            f"Fix the cause and re-run, or log manually once complete."
        )

    if args.no_log:
        return 0
    if not (os.environ.get("DAGSHUB_USER") and os.environ.get("DAGSHUB_TOKEN")):
        print(
            "WARNING: DAGSHUB_USER / DAGSHUB_TOKEN not set, so the results were NOT logged to the "
            "leaderboard. To log them later:\n  " + " ".join(log_cmd)
        )
        return 0
    subprocess.run(log_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
