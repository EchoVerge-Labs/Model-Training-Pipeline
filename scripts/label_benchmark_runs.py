"""Give slsb's MLflow runs a readable name and tags.

slsb (pinned at v0.1.0) names every run "<upstream>_<task>_<language>". For a
local upstream that is "models/selected_asr_tamil": identical for every seed and
for every model that was ever copied to models/selected. This renames the runs of
one benchmark to

    <prefix>_<task>_<language>_s<seed>        e.g. xlsr_200h_CoPT_norm_asr_tamil_s0

and tags them with what was benchmarked. Nothing is deleted -- metrics, params,
timestamps and run ids are untouched, and the name slsb gave the run is kept in
the `original_run_name` tag.

scripts/run_benchmark.sh calls this after slsb; to label older runs by hand:

    python scripts/label_benchmark_runs.py --mlflow-uri <uri> \
        --upstream models/selected --commit 6e1f31f \
        --prefix xlsr_200h_CoPT_rawtrain_normeval --tags checkpoint=9000 --dry-run
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.entities import RunTag
from mlflow.tracking import MlflowClient

EXPERIMENT = "SLSB-benchmark"  # slsb's DEFAULT_EXPERIMENT


def parse_tags(value: str) -> dict[str, str]:
    """ "model=xlsr300m,checkpoint=9000" -> {"model": "xlsr300m", "checkpoint": "9000"}"""
    tags = {}
    for item in filter(None, (part.strip() for part in value.split(","))):
        key, sep, val = item.partition("=")
        if not sep or not key:
            raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
        tags[key.strip()] = val.strip()
    return tags


def local_upstream_tags(upstream: str, selection_report: Path | None) -> dict[str, str]:
    """Tags read off the benchmarked model itself, when the upstream is a local directory."""
    tags = {}
    preprocessor = Path(upstream) / "preprocessor_config.json"
    if preprocessor.is_file():
        # What slsb's feature extractor did to the audio for this benchmark.
        tags["eval_normalize"] = str(json.loads(preprocessor.read_text())["do_normalize"]).lower()
    if selection_report and selection_report.is_file():
        best = json.loads(selection_report.read_text())["best_checkpoint"]
        tags["checkpoint"] = best.removeprefix("checkpoint-")
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--upstream", required=True, help="the --upstream slsb was run with")
    parser.add_argument("--prefix", required=True, help="run name prefix, e.g. xlsr_200h_CoPT_norm")
    parser.add_argument("--tags", type=parse_tags, default={}, help="key=value,key=value")
    parser.add_argument("--note", default=None, help="description shown on the run page")
    parser.add_argument(
        "--since-ms", type=int, default=0, help="only runs started at or after this"
    )
    parser.add_argument("--commit", default="", help="only runs whose git_commit starts with this")
    parser.add_argument(
        "--selection-report",
        type=Path,
        default=None,
        help="checkpoint_selection.json -- tags the runs with the checkpoint that was selected",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ["DAGSHUB_TOKEN"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()

    tags = {**local_upstream_tags(args.upstream, args.selection_report), **args.tags}
    if args.note:
        tags["mlflow.note.content"] = args.note

    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit(f"no MLflow experiment named {EXPERIMENT!r} at {args.mlflow_uri}")
    # Upstreams contain "/", which the filter grammar takes verbatim inside quotes.
    query = f"params.upstream = '{args.upstream}' and attributes.start_time >= {args.since_ms}"
    runs, page_token = [], None
    while True:
        page = client.search_runs(
            [experiment.experiment_id], filter_string=query, max_results=200, page_token=page_token
        )
        runs.extend(page)
        page_token = page.token
        if not page_token:
            break
    runs = [r for r in runs if r.data.params.get("git_commit", "").startswith(args.commit)]
    runs.sort(key=lambda r: r.info.start_time)

    seen, changed = set(), 0
    for run in runs:
        p = run.data.params
        name = f"{args.prefix}_{p['task']}_{p['language']}_s{p['seed']}"
        if name in seen:
            raise SystemExit(
                f"two runs would both be named {name!r} -- narrow the match with --since-ms/--commit"
            )
        seen.add(name)

        new_tags = dict(tags)
        # The name slsb gave the run, kept across re-labelling.
        if "original_run_name" not in run.data.tags:
            new_tags["original_run_name"] = run.info.run_name
        new_tags = {k: v for k, v in new_tags.items() if run.data.tags.get(k) != v}
        if run.info.run_name == name and not new_tags:
            continue
        changed += 1
        print(f"{run.info.run_name:<42} -> {name}")
        if args.dry_run:
            continue
        if new_tags:
            client.log_batch(run.info.run_id, tags=[RunTag(k, v) for k, v in new_tags.items()])
        if run.info.run_name != name:
            client.update_run(run.info.run_id, name=name)

    verb = "would be labelled" if args.dry_run else "labelled"
    print(f"{changed} run(s) {verb}, {len(runs) - changed} already up to date")


if __name__ == "__main__":
    main()
