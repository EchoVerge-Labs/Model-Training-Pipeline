"""Select the best pre-training checkpoint using a fast proxy evaluation.

Each milestone checkpoint is benchmarked with slsb on one task family, and the one with the
best **dev** score is kept. SLSB's protocol selects on dev and scores the test set once; the
test scores of a checkpoint chosen by its test score would be optimistic. slsb records the dev
score of every head it trains (`details`), so selection never has to look at test.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from pipeline.common.checkpoints import (
    TRAINING_STATE_FILE,
    is_complete_checkpoint,
    require_pretraining_complete,
)
from pipeline.common.config import add_config_args, load_raw, params_from_args
from pipeline.slsb import slsb_command

# default scratch root for a checkpoint's proxy evaluation; main() uses a per-model subdir
EVAL_DIR = Path("/tmp/ckpt_eval")

# The dev score slsb records for each task family (results entry -> details[key]) and whether
# a higher value is better. Emotion recognition is not here: its 5-fold runs record no dev score.
DEV_SCORE = {
    "asr": ("dev_cer", False),
    "sid": ("dev_accuracy", True),
    "asv": ("dev_eer", False),
    "sd": ("dev_der", False),
}


def find_milestone_checkpoints(model_dir: str, milestones: list[int]) -> list[Path]:
    """Complete milestone checkpoint directories in the model output, oldest first.

    Only milestones are proxy-evaluated: each eval is a full downstream
    fine-tune, so scoring every rolling checkpoint is too expensive. A milestone
    directory that exists but is incomplete (a crashed save) is skipped.
    """
    model_path = Path(model_dir)
    wanted = [model_path / f"checkpoint-{step}" for step in sorted(milestones)]
    missing = [c.name for c in wanted if not is_complete_checkpoint(c)]
    if missing:
        print(f"WARNING: milestone checkpoints missing or incomplete, skipping: {missing}")
    return [c for c in wanted if is_complete_checkpoint(c)]


def evaluate_checkpoint(
    ckpt_path: Path,
    task: str,
    seed: int,
    eval_root: Path | None = None,
    data_dir: str | None = None,
) -> float:
    """Run a fast proxy evaluation using slsb; returns the mean DEV score, lower is better.

    slsb writes its results to <out>/results_<upstream-with-/-as-__>.json, a list of
    per-(task,language,seed) entries. Each entry's `details` holds the dev score the head was
    selected on; that, never the test metric, is what ranks the checkpoints.
    """
    if task not in DEV_SCORE:
        raise SystemExit(
            f"checkpoint_selection.proxy_task={task!r} has no dev score to select on; "
            f"choose one of {sorted(DEV_SCORE)}"
        )
    key, higher_is_better = DEV_SCORE[task]
    upstream = str(ckpt_path)
    out_dir = (eval_root or EVAL_DIR) / ckpt_path.name
    cmd = [
        *slsb_command(),
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
    values = [e["details"][key] for e in ok_entries if key in e.get("details", {})]
    if not values:
        print(f"WARNING: no dev score ({key}) in the results for {ckpt_path.name} (task={task})")
        return float("inf")

    mean_value = sum(values) / len(values)
    # lower is better throughout: accuracy is negated
    return -mean_value if higher_is_better else mean_value


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="select even though pre-training has not reached max_updates",
    )
    args = parser.parse_args()

    params = params_from_args(args)
    cfg = params.pretrain
    if not args.allow_incomplete:
        require_pretraining_complete(params.model.reports_dir, cfg.max_updates)
    raw = load_raw(args.config, args.model, args.overlay)
    sel_cfg = raw.get("checkpoint_selection", {})

    proxy_task = sel_cfg.get("proxy_task", "asr")
    proxy_seed = sel_cfg.get("proxy_seed", 0)
    proxy_data = sel_cfg.get("proxy_data_dir")
    output_dir = Path(sel_cfg.get("output_dir", "models/selected"))

    # per-model proxy-eval scratch dir, so concurrent selections of different models
    # do not collide on a shared /tmp/ckpt_eval/checkpoint-<step>
    eval_root = EVAL_DIR / Path(cfg.output_dir).name

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
        _write_provenance(output_dir, cfg, best_checkpoint="(final model dir)")
        return

    print(
        f"Evaluating {len(checkpoints)} checkpoints on proxy task: {proxy_task} (selecting on dev)"
    )
    results = {}
    for ckpt in checkpoints:
        score = evaluate_checkpoint(ckpt, proxy_task, proxy_seed, eval_root, proxy_data)
        results[ckpt.name] = score
        print(f"  {ckpt.name}: {proxy_task} dev {DEV_SCORE[proxy_task][0]} = {score:.4f}")

    # Pick best -- but never "select" a checkpoint when every eval failed
    if all(score == float("inf") for score in results.values()):
        raise SystemExit(
            "ERROR: every proxy eval failed -- not selecting a checkpoint. "
            "Check checkpoint_selection.proxy_data_dir and the slsb errors above."
        )
    best_name = min(results, key=results.get)
    best_path = Path(cfg.output_dir) / best_name
    print(f"\nBest checkpoint: {best_name} (score: {results[best_name]:.4f})")

    # Copy to selected -- without training_state.pt (optimizer state, ~2x model size):
    # the selected copy is only read by the benchmark/SLSB, which never resumes from it.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(best_path, output_dir, ignore=shutil.ignore_patterns(TRAINING_STATE_FILE))
    _write_provenance(output_dir, cfg, best_checkpoint=best_name)

    # Write selection report
    report = {
        "proxy_task": proxy_task,
        "selection_split": "dev",
        "selection_metric": DEV_SCORE[proxy_task][0],
        "proxy_seed": proxy_seed,
        "best_checkpoint": best_name,
        "best_score": results[best_name],
        "all_scores": results,
        "source_output_dir": str(cfg.output_dir),
        "base_model": cfg.base_model,
    }
    reports_dir = Path(params.model.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "checkpoint_selection.json", "w") as f:
        json.dump(report, f, indent=2)


def _write_provenance(output_dir: Path, cfg, best_checkpoint: str) -> None:
    """Record which pre-training run this selected dir came from, so the benchmark can
    verify it is the right model (guards against a stale models/selected* holding another
    model, which would otherwise be logged to the leaderboard under the wrong name)."""
    prov = {
        "source_output_dir": str(cfg.output_dir),
        "base_model": cfg.base_model,
        "best_checkpoint": best_checkpoint,
    }
    (output_dir / "selection_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
