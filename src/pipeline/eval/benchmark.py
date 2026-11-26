"""Run the SLSB benchmark for one run and log it to the SLSB leaderboard (`make bench`).

This only drives the SLSB-benchmark repository -- its own `slsb run` and its own
`scripts/log_results.py` -- and adds no benchmark logic of its own:

1. `slsb run` over `benchmark.tasks` x `benchmark.seeds`, WITHOUT an MLflow URI: live logging
   would make one MLflow run per task and seed, which are not leaderboard rows.
2. slsb's results_<upstream>.json is copied to <bench dir>/metrics.json, the fixed path
   report.py and dvc.yaml read.
3. With DAGSHUB_USER / DAGSHUB_TOKEN set, log_results.py logs the folder ONCE, as one MLflow run
   per model in the `slsb-<protocol>` experiment of SLSB-benchmark.mlflow (the leaderboard),
   with kind, family, base model, checkpoint step and hours of continued pre-training. A run's
   row then gets the same name-slot tags as its ssl-pretraining run (method, languages, hours,
   cnn, passes, batching, data, tag). Without the credentials the results stay local and the
   exact logging command is printed.

  python -m pipeline.eval.benchmark --config params-wavlm.yaml                 # the run
  python -m pipeline.eval.benchmark --config params-wavlm.yaml --variant configs/variants/cnn.yaml
  python -m pipeline.eval.benchmark --config params-wavlm.yaml --frozen        # its base model
  python -m pipeline.eval.benchmark --config params-wavlm.yaml \\
      --upstream models/wavlm-large-copt-si-ta-200h/checkpoint-9000 --name <row name>

By default it benchmarks the run's selected checkpoint (models/<run>/selected) under the run's
name into reports/<run>/bench/. --frozen benchmarks the model's base as its frozen row (e.g.
wavlm-large-frozen) into reports/<that name>/bench/. --upstream benchmarks any other checkpoint or
Hugging Face id; it is logged only with an explicit --name, so it can never overwrite a row.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.common.naming import hours_str
from pipeline.common.schema import Params, add_run_args, load_params, variants_arg
from pipeline.eval.slsb import (
    SlsbNotFound,
    log_results_script,
    require_version,
    slsb_command,
    slsb_python,
)
from pipeline.models.registry import MODEL_REGISTRY, get_model_spec

LEADERBOARD_URI = "https://dagshub.com/EchoVerge-LABS/SLSB-benchmark.mlflow"
# The task families a full leaderboard row covers (slsb's own families).
ALL_FAMILIES = ("asr", "sid", "er", "sd", "asv", "ic")


def bench_name(upstream: str) -> str:
    """Folder name under reports/ for an arbitrary upstream (path or HF id)."""
    parts = [p for p in Path(upstream).parts if p not in ("models", ".", "..", "/")]
    return "-".join(parts) if parts else upstream.replace("/", "__")


def frozen_name(model_key: str) -> str:
    """A model's frozen baseline: its base model, un-adapted, named <model>-frozen (e.g.
    wavlm-large-frozen) on the leaderboard and under reports/. It has no training run."""
    return f"{get_model_spec(model_key).slsb_name}-frozen"


def require_complete_pretraining(params: Params, allow_incomplete: bool = False) -> None:
    """Refuse to benchmark a run whose pre-training has not finished, so a half-trained
    checkpoint is never scored and logged as a result. reports/<run>/pretrain_metrics.json is
    written with completed=true only when training reaches max_updates. --allow-incomplete (or
    --frozen / an explicit --upstream, handled by the caller) overrides this."""
    if allow_incomplete:
        return
    path = params.reports_dir / "pretrain_metrics.json"
    if not path.is_file():
        raise SystemExit(
            f"refusing to benchmark {params.run}: {path} not found -- pre-training has not "
            f"finished (run `make train` to completion first, or pass --allow-incomplete)."
        )
    metrics = json.loads(path.read_text())
    if not metrics.get("completed"):
        raise SystemExit(
            f"refusing to benchmark {params.run}: pre-training is incomplete "
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


def selected_checkpoint_step(params: Params) -> str:
    """The step of the checkpoint `make ckpt` chose for this run, e.g. "9000"; "" if none."""
    report = params.reports_dir / "checkpoint_selection.json"
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
    tags: dict = field(default_factory=dict)  # extra tags for the row (a run's name slots)

    @property
    def partial_note(self) -> str:
        """A row that does not cover every task family says so in its note."""
        missing = [f for f in ALL_FAMILIES if f not in self.tasks.split(",")]
        if not missing:
            return ""
        return (
            f"Partial benchmark: task families {self.tasks} only "
            f"(not {','.join(missing)}); not comparable with full rows on those tasks."
        )


def make_plan(raw: dict, params: Params, args: argparse.Namespace) -> Plan:
    from pipeline.common.tracking import run_tags

    bench = raw["benchmark"]
    spec = get_model_spec(params.model_key)
    base_model = params.pretrain.base_model
    tags: dict = {}
    if args.frozen:
        upstream, name = base_model, args.name or frozen_name(params.model_key)
        out_dir = Path("reports") / (args.out or name) / "bench"
        kind, checkpoint, hours = "frozen", "", ""
    elif args.upstream:
        upstream, name = args.upstream, args.name  # never inherit the run's own row
        out_dir = Path("reports") / (args.out or name or bench_name(upstream)) / "bench"
        is_base = not Path(upstream).exists()  # an HF id, not a checkpoint directory
        kind = "frozen" if is_base else "adapted"
        if is_base:  # another model's base (e.g. utter-project/mHuBERT-147): its own family
            spec = next((s for s in MODEL_REGISTRY.values() if s.base_model == upstream), spec)
            base_model = upstream
        step = re.search(r"checkpoint-(\d+)", upstream)
        checkpoint = args.checkpoint or (step.group(1) if step and not is_base else "")
        hours = "" if is_base else hours_str(params.hours)
    else:
        upstream, name = str(params.selected_dir), args.name or params.run
        out_dir = Path("reports") / (args.out or params.run) / "bench"
        kind = "adapted"
        checkpoint = args.checkpoint or selected_checkpoint_step(params)
        hours = hours_str(params.hours)
        tags = {k: v for k, v in run_tags(params).items() if k != "mlflow.note.content"}
    data_dir = bench.get("slsb_data_dir", "../SLSB-benchmark/data")
    return Plan(
        upstream=upstream,
        out_dir=out_dir,
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
        tags=tags,
    )


def tag_leaderboard_row(plan: Plan, metrics_path: Path) -> None:
    """Give the run's leaderboard row the same name-slot tags as its ssl-pretraining run,
    so both records can be filtered the same way. Best-effort: the row is already logged."""
    if not plan.tags:
        return
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        protocol = json.loads(metrics_path.read_text()).get("protocol")
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.environ.get("DAGSHUB_USER", ""))
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ.get("DAGSHUB_TOKEN", ""))
        mlflow.set_tracking_uri(plan.leaderboard_uri)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(f"slsb-{protocol}")
        rows = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.`slsb.model` = '{plan.name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not rows:
            raise LookupError(f"no row named {plan.name} in slsb-{protocol}")
        for key, value in plan.tags.items():
            client.set_tag(rows[0].info.run_id, key, value)
        print(f"Tagged leaderboard row {plan.name} ({rows[0].info.run_id}) with the run's slots")
    except Exception as e:  # noqa: BLE001 -- the scores are logged; tags can be re-added
        warnings.warn(f"could not tag the leaderboard row {plan.name}: {e}")


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
    add_run_args(parser)
    parser.add_argument("--frozen", action="store_true", help="benchmark the run's base model")
    parser.add_argument("--upstream", default=None, help="benchmark this checkpoint / HF id")
    parser.add_argument("--name", default=None, help="leaderboard row (default: the run's name)")
    parser.add_argument("--out", default=None, help="reports/<out>/bench (default: the row name)")
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
    raw, params = load_params(args.config, variants_arg(args.variant), args.tag)
    plan = make_plan(raw, params, args)

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
    note = " ".join(x for x in (args.note, plan.partial_note) if x)
    log_cmd = (
        log_results_command(plan, python, script, note, args.replace, args.carry_over)
        if plan.name
        else None
    )
    print(f"[{plan.name or plan.out_dir.parent.name}] ({plan.kind}) -> {plan.out_dir}")
    print(f"  tasks={plan.tasks} seeds={plan.seeds}")
    print("+ " + shlex.join(run_cmd), flush=True)
    if args.dry_run:
        if log_cmd:
            print("+ " + shlex.join(log_cmd))
        if plan.tags:
            print(f"  then tag the row with {plan.tags}")
        return 0

    # Only the run's OWN selected checkpoint is gated on its folder and on finished
    # pre-training; --frozen and an explicit --upstream are the user's deliberate choice.
    if not (args.frozen or args.upstream):
        from pipeline.common.tracking import claim_run_folder

        claim_run_folder(params, write=False)
        require_complete_pretraining(params, args.allow_incomplete)
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
            f"ERROR: SLSB left {len(failed)} task(s) unfinished for {plan.out_dir}: "
            f"{', '.join(failed)} (see {metrics_path}) -- the benchmark is partial and was NOT "
            f"logged to the leaderboard.",
            file=sys.stderr,
        )
        return 1

    if log_cmd is None:
        print("NOTE: an ad-hoc --upstream without --name is not logged to the leaderboard.")
        return 0
    if args.no_log:
        print("Not logged (--no-log). To log it:\n  " + shlex.join(log_cmd))
        return 0
    if not (os.environ.get("DAGSHUB_USER") and os.environ.get("DAGSHUB_TOKEN")):
        print(
            "WARNING: DAGSHUB_USER / DAGSHUB_TOKEN not set, so the results were NOT logged to "
            "the leaderboard. To log them later:\n  " + shlex.join(log_cmd)
        )
        return 0
    rc = subprocess.call(log_cmd)
    if rc == 0:
        tag_leaderboard_row(plan, metrics_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
