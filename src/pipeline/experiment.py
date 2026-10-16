"""Single-language ablation runs: where each run's files live, and the guards that keep
the runs apart from each other and from the joint Sinhala+Tamil run.

A run is the joint selection restricted to ONE language (select.py runs its normal
algorithm unchanged and keeps that language), trained with the unchanged `pretrain`
recipe. Only locations differ, so `resolve()` rewrites paths and nothing else.

Backbone-agnostic: this file is identical on every branch.
"""

import argparse
import dataclasses
import gzip
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pipeline.schema import Params

PROVENANCE_FILE = "experiment.json"

# Locations of the joint run. A single-language run must never write under any of these.
# (Shars live in data/shars_single/, not data/shars/<name>/: DVC rejects an output
# nested inside another stage's output directory.)
JOINT_DIRS = ("data/shars", "models/selected")


@dataclass(frozen=True)
class ExperimentPaths:
    name: str  # e.g. "si-100h"
    language: str  # catalog spelling, e.g. "sinhala"
    target_hours: float
    backbone_tag: str  # e.g. "xlsr300m"
    manifest_dir: Path
    shar_dir: Path
    model_dir: Path
    selected_dir: Path
    reports_dir: Path
    bench_dir: Path
    mlflow_run_name: str
    # Backbones that train on k-means pseudo-labels (WavLM): this run's label file, assigned
    # on its own shards. None where the backbone has no labels.
    labels_path: Path | None = None

    @property
    def train_manifest(self) -> Path:
        return self.manifest_dir / "train.jsonl"

    @property
    def dev_manifest(self) -> Path:
        return self.manifest_dir / "dev.jsonl"

    @property
    def selection_report(self) -> Path:
        return self.reports_dir / "selection_report.md"

    @property
    def selection_stats(self) -> Path:
        return self.reports_dir / "selection_stats.json"

    @property
    def unresolved_clips(self) -> Path:
        return self.reports_dir / "unresolved_clips.csv"

    @property
    def materialise_report(self) -> Path:
        return self.reports_dir / "materialise.json"

    @property
    def checkpoint_selection(self) -> Path:
        return self.reports_dir / "checkpoint_selection.json"

    @property
    def proxy_scratch(self) -> Path:
        """Where select_checkpoint's slsb proxy evals write (not shared between runs)."""
        return Path("/tmp/ckpt_eval") / f"{self.backbone_tag}-{self.name}"


def _norm(path: str | Path) -> str:
    return str(Path(path)).rstrip("/")


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def resolve(params: Params, name: str) -> tuple[Params, ExperimentPaths]:
    """Params for run `name` (only locations rewritten) and its derived paths.

    Raises if the run is unknown, or if any of its locations could collide with the
    joint run or with another run.
    """
    if params.experiments is None or name not in params.experiments.runs:
        known = sorted(params.experiments.runs) if params.experiments else []
        raise ValueError(f"unknown experiment {name!r}; params.yaml defines {known}")
    cfg = params.experiments
    run = cfg.runs[name]
    tag = cfg.backbone_tag
    exp = ExperimentPaths(
        name=name,
        language=run.language,
        target_hours=run.target_hours,
        backbone_tag=tag,
        manifest_dir=Path("data/manifests") / name,
        shar_dir=Path("data/shars_single") / name,
        model_dir=Path("models") / f"{tag}-{name}",
        selected_dir=Path("models/selected_single") / f"{tag}-{name}",
        reports_dir=Path("reports") / name,
        bench_dir=Path("reports/bench") / f"{tag}-{name}",
        mlflow_run_name=f"{tag}-copt-{name}",
    )
    _check_isolated(params, cfg.runs.keys(), tag)

    pretrain_update = {"output_dir": str(exp.model_dir)}
    update = {"shard": params.shard.model_copy(update={"output_dir": str(exp.shar_dir)})}
    cluster = getattr(params, "cluster", None)
    if cluster is not None:
        labels = Path("data/labels") / name / Path(cluster.labels_output).name
        exp = dataclasses.replace(exp, labels_path=labels)
        update["cluster"] = cluster.model_copy(update={"labels_output": str(labels)})
        pretrain_update["labels_path"] = str(labels)
    update["pretrain"] = params.pretrain.model_copy(update=pretrain_update)
    return params.model_copy(update=update), exp


def _check_isolated(params: Params, names, tag: str) -> None:
    """Every location of every run is disjoint from the joint run and from every other run."""
    joint = [_norm(params.pretrain.output_dir), *JOINT_DIRS]
    joint += ["data/manifests/train.jsonl", "data/manifests/dev.jsonl"]
    owner: dict[str, str] = {}
    for n in names:
        for what, loc in (
            ("shar_dir", f"data/shars_single/{n}"),
            ("model_dir", f"models/{tag}-{n}"),
            ("selected_dir", f"models/selected_single/{tag}-{n}"),
            ("bench_dir", f"reports/bench/{tag}-{n}"),
        ):
            for j in joint:
                if _overlaps(_norm(loc), j):
                    raise ValueError(f"experiment {n}: {what} {loc} overlaps the joint run's {j}")
            for other_loc, other in owner.items():
                if _overlaps(_norm(loc), other_loc) and other != n:
                    raise ValueError(f"experiments {n} and {other} share {loc} ({what})")
            owner[_norm(loc)] = n


def load(config: str, experiment: str | None) -> tuple[Params, ExperimentPaths | None]:
    """Params (rewritten for `experiment`, if given) and the run's paths (None = joint)."""
    params = Params.from_yaml(config)
    if experiment is None:
        return params, None
    return resolve(params, experiment)


# ── language guards ─────────────────────────────────────────────────────


def assert_manifest_language(manifest: str | Path, language: str) -> int:
    """Raise if any clip in the manifest is not `language`. Returns the clip count."""
    n, foreign = 0, defaultdict(int)
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            lang = json.loads(line).get("language", "unknown")
            n += 1
            if lang != language:
                foreign[lang] += 1
    if foreign:
        raise ValueError(
            f"{manifest}: expected only {language!r} clips but found {dict(foreign)} "
            f"out of {n} -- refusing to continue"
        )
    return n


def assert_shar_language(shar_dir: str | Path, language: str) -> int:
    """Raise if any cut in the shards (or no cut at all) is not `language`."""
    n, foreign = 0, defaultdict(int)
    for cuts_file in sorted(Path(shar_dir).glob("cuts.*.jsonl.gz")):
        with gzip.open(cuts_file, "rt", encoding="utf-8") as f:
            for line in f:
                lang = json.loads(line).get("custom", {}).get("language", "unknown")
                n += 1
                if lang != language:
                    foreign[lang] += 1
    if n == 0:
        raise ValueError(f"{shar_dir}: no cuts found -- run the shard step for this experiment")
    if foreign:
        raise ValueError(
            f"{shar_dir}: expected only {language!r} cuts but found {dict(foreign)} out of {n}"
        )
    return n


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── resume safety ───────────────────────────────────────────────────────


def check_resume(exp: ExperimentPaths, params: Params, resume_from, write: bool = True) -> None:
    """Refuse to train `exp` on top of anything that is not its own.

    - the output directory is stamped with experiment.json (name, language, base model,
      training-manifest hash); a stamp that differs, or checkpoints with no stamp at all,
      stop the run;
    - an explicit --resume-from must be a checkpoint inside this run's own directory.
    """
    out = Path(params.pretrain.output_dir)
    if out.resolve() != exp.model_dir.resolve():
        raise ValueError(f"{exp.name}: output_dir {out} is not this run's {exp.model_dir}")

    if resume_from is not None:
        parent = Path(resume_from).resolve().parent
        if parent != out.resolve():
            raise ValueError(
                f"{exp.name}: --resume-from {resume_from} is not inside {out} -- a run can only "
                f"resume its own checkpoints (never another experiment's or the joint run's)"
            )

    stamp = {
        "experiment": exp.name,
        "language": exp.language,
        "base_model": params.pretrain.base_model,
        "train_manifest_sha256": file_sha256(exp.train_manifest),
    }
    stamp_path = out / PROVENANCE_FILE
    if stamp_path.exists():
        found = json.loads(stamp_path.read_text())
        diff = {k: (found.get(k), v) for k, v in stamp.items() if found.get(k) != v}
        if diff:
            raise ValueError(
                f"{out} belongs to a different run or dataset (stored vs expected): {diff}. "
                f"Use a fresh output_dir or delete it deliberately."
            )
    elif any(out.glob("checkpoint-*")):
        raise ValueError(
            f"{out} holds checkpoints but no {PROVENANCE_FILE}, so they cannot be proven to "
            f"belong to {exp.name}; refusing to auto-resume"
        )
    elif write:
        out.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")


# ── summary ─────────────────────────────────────────────────────────────


def summary_text(train, exp: ExperimentPaths, params: Params) -> str:
    """The pre-training summary of what was selected. `train`: segments or manifest rows
    exposing .language/.genre/.duration_seconds."""
    hours = sum(s.duration_seconds for s in train) / 3600
    by_genre: dict[str, float] = defaultdict(float)
    foreign = 0
    for s in train:
        by_genre[s.genre] += s.duration_seconds / 3600
        foreign += s.language != exp.language
    cfg = params.pretrain
    passes = cfg.max_updates * cfg.target_batch_seconds / 3600 / hours if hours else float("nan")
    lines = [
        f"Language: {exp.language.capitalize()}",
        f"Target hours: {exp.target_hours:g}",
        f"Selected hours: {hours:.2f}",
        f"Number of clips: {len(train)}",
        f"Clips of any other language: {foreign}",
        f"Passes over the audio at max_updates={cfg.max_updates}: {passes:.1f}",
        "Genre breakdown:",
    ]
    lines += [f"  {g}: {h:.2f}h" for g, h in sorted(by_genre.items(), key=lambda x: -x[1])]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate the single-language experiment config")
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    params = Params.from_yaml(args.config)
    if params.experiments is None:
        raise SystemExit("params.yaml has no `experiments` section")
    for name in params.experiments.runs:
        resolved, exp = resolve(params, name)
        locations = {"output_dir", "labels_path"}
        before, after = params.pretrain.model_dump(), resolved.pretrain.model_dump()
        changed = {k for k in before if before[k] != after[k]} - locations
        assert not changed, f"resolve() changed training hyperparameters: {changed}"
        print(f"{name}: {exp.language} {exp.target_hours:g}h")
        for label, value in (
            ("manifests", exp.manifest_dir),
            ("shards", exp.shar_dir),
            ("model", exp.model_dir),
            ("selected", exp.selected_dir),
            ("reports", exp.reports_dir),
            ("benchmark", exp.bench_dir),
        ):
            print(f"  {label:10s}{value}")
    print("OK: all experiments are isolated from the joint run and from each other")


if __name__ == "__main__":
    main()
