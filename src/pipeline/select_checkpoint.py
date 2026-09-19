"""Select the best pre-training checkpoint using a fast proxy evaluation."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.schema import Params

# Mirrors slsb's own per-family primary metric (see SLSB-benchmark's
# configs/tasks/*.yaml / scripts/run_all.py PRIMARY_METRIC). Lower is better
# for all of these once normalised in evaluate_checkpoint() below.
PRIMARY_METRIC = {"asr": "wer", "sid": "accuracy", "er": "accuracy", "asv": "eer"}


def find_milestone_checkpoints(model_dir: str) -> list[Path]:
    """Find checkpoint directories in the model output."""
    model_path = Path(model_dir)
    checkpoints = sorted(model_path.glob("checkpoint-*"))
    return checkpoints


def evaluate_checkpoint(ckpt_path: Path, task: str, seed: int, data_dir: str = None) -> float:
    """Run a fast proxy evaluation using slsb.

    slsb writes its results to <out>/results_<upstream-with-/-as-__>.json, a
    list of per-(task,language,seed) result entries -- not a flat
    {task: score} dict. See scripts/run_benchmark.sh for the same convention
    used by the dvc.yaml `benchmark` stage.
    """
    upstream = str(ckpt_path)
    out_dir = Path(f"/tmp/ckpt_eval/{ckpt_path.name}")
    cmd = [
        "slsb", "run",
        "--upstream", upstream,
        "--tasks", task,
        "--seeds", str(seed),
        "--out", str(out_dir),
    ]
    if data_dir:
        cmd.extend(["--data-dir", data_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: Eval failed for {ckpt_path.name}: {result.stderr[:200]}")
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

    metric_name = PRIMARY_METRIC.get(task)
    if metric_name is None:
        print(f"WARNING: no known primary metric for proxy task '{task}'")
        return float("inf")

    values = [e["metrics"][metric_name] for e in ok_entries if metric_name in e.get("metrics", {})]
    if not values:
        print(f"WARNING: metric '{metric_name}' missing from {ckpt_path.name} results")
        return float("inf")

    mean_value = sum(values) / len(values)

    # For ASR/EER: lower is better already. For accuracy: higher is better, so negate.
    if metric_name == "accuracy":
        return -mean_value
    return mean_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    cfg = params.pretrain
    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    sel_cfg = raw.get("checkpoint_selection", {})

    proxy_task = sel_cfg.get("proxy_task", "asr")
    proxy_seed = sel_cfg.get("proxy_seed", 0)
    proxy_data = sel_cfg.get("proxy_data_dir")
    output_dir = Path(sel_cfg.get("output_dir", "models/selected"))

    checkpoints = find_milestone_checkpoints(cfg.output_dir)
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

    # Pick best
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
        "best_checkpoint": best_name,
        "best_score": results[best_name],
        "all_scores": results,
    }
    with open("reports/checkpoint_selection.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
