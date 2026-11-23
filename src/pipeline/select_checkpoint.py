"""Select the best pre-training checkpoint using a fast proxy evaluation.

Serves all five models: the params file (--config) says which model and where its
checkpoints are, and the report suffix defaults to `-<model_key>`.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from pipeline.checkpoints import has_backbone
from pipeline.schema import Params, load_raw_params

# Checkpoints are ranked on the DEV score slsb records in each result entry's `details` (the
# score its probe head was selected on), never on the test metric: choosing a checkpoint by
# test score would make the test numbers reported for it optimistic. Maps proxy task ->
# (details key, higher is better). Emotion recognition is absent on purpose: its 5-fold runs
# record no dev score, so it cannot be a proxy task.
DEV_SCORE = {
    "asr": ("dev_cer", False),
    "sid": ("dev_accuracy", True),
    "asv": ("dev_eer", False),
    "sd": ("dev_der", False),
}


def find_milestone_checkpoints(model_dir: str, milestones: list[int]) -> list[Path]:
    """Milestone checkpoint directories with a loadable backbone, oldest first.

    Only milestones are proxy-evaluated: each eval is a full downstream
    fine-tune, so scoring every rolling checkpoint is too expensive. Checkpoints
    are written atomically (see train_masked_prediction.py), so a checkpoint-<step> directory
    that has a backbone is a finished one, for the contrastive models too, whose
    checkpoints have no masked-prediction head.
    """
    wanted = sorted(milestones)
    found = {step: Path(model_dir) / f"checkpoint-{step}" for step in wanted}
    missing = [f"checkpoint-{step}" for step, path in found.items() if not has_backbone(path)]
    if missing:
        print(f"WARNING: milestone checkpoints missing or incomplete, skipping: {missing}")
    return [path for path in found.values() if has_backbone(path)]


def evaluate_checkpoint(
    ckpt_path: Path, task: str, seed: int, data_dir: str | None = None
) -> float:
    """Run a fast proxy evaluation using slsb.

    slsb writes its results to <out>/results_<upstream-with-/-as-__>.json, a
    list of per-(task,language,seed) result entries -- not a flat
    {task: score} dict. See scripts/run_benchmark.sh for the same convention
    used by the dvc.yaml `benchmark` stage.
    """
    # slsb's output is only needed to read the score: a private directory per call (so
    # concurrent selections never collide) that is removed afterwards.
    with tempfile.TemporaryDirectory(prefix=f"ckpt_eval_{ckpt_path.name}_") as tmp:
        return _evaluate(ckpt_path, task, seed, data_dir, Path(tmp))


def _evaluate(ckpt_path: Path, task: str, seed: int, data_dir: str | None, out_dir: Path) -> float:
    upstream = str(ckpt_path)
    cmd = [
        "slsb",
        "run",
        "--upstream",
        upstream,
        "--tasks",
        task,
        "--seeds",
        str(seed),
        "--out",
        str(out_dir),
    ]
    if data_dir:
        # slsb reads its probe hyperparams from ./params.yaml by default, which here
        # is the pipeline's params file -- use the one next to the benchmark data.
        slsb_params = Path(data_dir).parent / "params.yaml"
        cmd.extend(["--data-dir", data_dir, "--params", str(slsb_params)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"WARNING: Eval failed for {ckpt_path.name}:\n{result.stderr[-2000:]}")
        return float("inf")

    safe_upstream = upstream.replace("/", "__")
    results_path = out_dir / f"results_{safe_upstream}.json"
    if not results_path.exists():
        print(f"WARNING: expected results file not found: {results_path}")
        return float("inf")

    with open(results_path) as f:
        summary = json.load(f)

    ok_entries = [r for r in summary.get("results", []) if r.get("status") == "ok"]
    if not ok_entries:
        print(f"WARNING: no successful runs for {ckpt_path.name} (task={task})")
        return float("inf")

    if task not in DEV_SCORE:
        print(f"WARNING: no dev score known for proxy task '{task}'")
        return float("inf")
    key, higher_is_better = DEV_SCORE[task]
    values = [e["details"][key] for e in ok_entries if key in e.get("details", {})]
    if not values:
        print(f"WARNING: no dev score ({key}) in the results for {ckpt_path.name} (task={task})")
        return float("inf")

    mean_value = sum(values) / len(values)
    # lower is better throughout: a higher-is-better score (accuracy) is negated
    return -mean_value if higher_is_better else mean_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params-mhubert147.yaml")
    parser.add_argument("--experiment", default=None, help="run name; overrides `experiment:`")
    parser.add_argument(
        "--report-suffix",
        default=None,
        help="Suffix for the report file; default '-<run name>' e.g. "
        "checkpoint_selection-wavlm-large-si-ta-200h.json",
    )
    args = parser.parse_args()

    params = Params.from_yaml(args.config, args.experiment)
    cfg = params.pretrain
    raw = load_raw_params(args.config, args.experiment)
    sel_cfg = raw.get("checkpoint_selection", {})

    proxy_task = sel_cfg.get("proxy_task", "asr")
    proxy_seed = sel_cfg.get("proxy_seed", 0)
    proxy_data = sel_cfg.get("proxy_data_dir")
    output_dir = Path(sel_cfg.get("output_dir", f"models/selected-{params.run}"))

    if proxy_task not in DEV_SCORE:
        raise SystemExit(
            f"ERROR: checkpoint_selection.proxy_task={proxy_task!r} has no dev score to rank on; "
            f"use one of {sorted(DEV_SCORE)} (emotion recognition records no dev score)."
        )

    checkpoints = find_milestone_checkpoints(cfg.output_dir, cfg.milestone_checkpoints)
    if not checkpoints:
        print("No checkpoints found — using the final model directory directly")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Copy/symlink the main model dir
        for f in Path(cfg.output_dir).iterdir():
            if f.name.startswith("checkpoint-"):
                continue
            dest = output_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest) if f.is_file() else shutil.copytree(f, dest)
        return

    print(f"Evaluating {len(checkpoints)} checkpoints on proxy task: {proxy_task}")
    results = {}
    for ckpt in checkpoints:
        score = evaluate_checkpoint(ckpt, proxy_task, proxy_seed, proxy_data)
        results[ckpt.name] = score
        print(f"  {ckpt.name}: {proxy_task} score = {score:.4f}")

    # Pick best -- but never "select" a checkpoint when every eval failed
    if all(score == float("inf") for score in results.values()):
        raise SystemExit(
            "ERROR: every proxy eval failed -- not selecting a checkpoint. "
            "Check checkpoint_selection.proxy_data_dir and the slsb errors above."
        )
    best_name = min(results, key=results.get)
    best_path = Path(cfg.output_dir) / best_name
    print(f"\nBest checkpoint: {best_name} (score: {results[best_name]:.4f})")

    # Copy to selected
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(best_path, output_dir)

    # Write selection report
    report = {
        "proxy_task": proxy_task,
        "proxy_seed": proxy_seed,
        "ranked_on": DEV_SCORE[proxy_task][0],  # a dev score, never the test metric
        "best_checkpoint": best_name,
        "best_score": results[best_name],
        "all_scores": results,
    }
    suffix = args.report_suffix if args.report_suffix is not None else f"-{params.run}"
    report_path = Path(f"reports/checkpoint_selection{suffix}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
