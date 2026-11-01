# Model Training Pipeline — Continued Pre-training of XLS-R 300m on Sinhala/Tamil

DVC-orchestrated pipeline for continued self-supervised wav2vec2 pre-training, automatic
checkpoint selection, and benchmarking. Produces a Sinhala/Tamil-specialised speech
encoder from `facebook/wav2vec2-xls-r-300m`.

## Pipeline DAG

```
snapshot_catalog → index_drive → select → materialise → shard → pretrain → select_checkpoint → benchmark → report
```

Each stage is a `dvc.yaml` target; `dvc repro` runs whatever is stale given
`params.yaml` and each stage's declared deps. The `benchmark` stage depends
on `pretrain`'s and `select_checkpoint`'s outputs, so `dvc repro` chains
training straight into evaluation automatically.

## Prerequisites

- NVIDIA DGX Spark (GB10, ARM64) with the NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.11-py3`)
- Google Drive with the pre-processed audio, mounted locally with `rclone mount` (default `~/google-drive`; set `select.drive_root` in `params.yaml` to the `Pre Processed Data` folder). Clips are laid out `<Lang>/<genre>/<source-dir>/<hash>_NNN.wav`
- A DagsHub account (DVC remote + MLflow tracking) — see `.env.example`
- A Google OAuth client-secret JSON (Desktop-app type) for an account that can read the "Sinhala X Tamil Voice Dataset" sheet. The first `make snapshot` opens a browser sign-in and caches the token at `~/.config/gspread/authorized_user.json`
- [SLSB-benchmark](https://github.com/EchoVerge-Labs/SLSB-benchmark) v0.1.0 installed (`pip install -e ".[benchmark]"`, or already in the Docker image)

## Quick start

```bash
# 1. Install (inside the NGC container, or a matching ARM64 CUDA environment)
pip install -e ".[data,benchmark,dev]"
cp .env.example .env   # fill in DAGSHUB_TOKEN, GOOGLE_SHEET_ID, GOOGLE_CLIENT_SECRET_JSON, HF_TOKEN

# 2. Measure real throughput BEFORE scoping max_updates / timelines.
#    This loads the real 300M model and does real GPU compute -- treat it
#    like any other GPU job (see "Running training" below).
make phase0

# 3. Run the pipeline stage by stage (or `make all` / `dvc repro` for the whole DAG)
make snapshot   # freeze the Google Sheet catalog to data/catalog/catalog_latest.csv
                # (or: python -m pipeline.snapshot_catalog --local-csv <sinhala.csv> <tamil.csv>)
make index      # list every wav on the Drive mount (path + size) -> data/catalog/drive_index.tsv
make select     # dedupe, pin each clip to its Drive file, select train/holdout -> data/manifests/
make pull       # copy the selected clips from the mount -> data/raw/ (same <Lang>/<genre>/... layout)
make shard      # pack into Lhotse Shar tarballs -> data/shars/
make train      # continued pre-training -> models/xlsr300m-si-ta-200h-norm/
make ckpt       # proxy-eval milestone checkpoints, copy the best -> models/selected/
make bench      # slsb run on the selected checkpoint -> reports/bench/xlsr300m_adapted_norm/
make report     # aggregate into reports/results.md + results_table.csv
```

### Why `make index` exists

The sheet's `name_in_drive` (`<hash>_NNN.wav`) is **not unique** on the Drive: the same
name recurs across different source videos with different audio. `select` therefore pins
each catalog row to a real file by `(language, genre, name, size)` against the index,
records that path (not the bare name) in the manifests, and gives every clip a unique id.
Rows repeated in the sheet are collapsed, and rows that don't resolve to exactly one file
are set aside in `reports/unresolved_clips.csv` rather than guessed.

### Running training

`make train` / `make phase0` launch real, hours-to-days-long GPU jobs on
shared hardware. Run them in `tmux` (not `nohup`), and don't launch either
without checking first if you're running this pipeline as part of a larger
session. For two-node training, run `scripts/nccl_bandwidth_test.sh` on both
nodes first to confirm the ConnectX-7 link is actually being used (see
`configs/nccl-env.sh` — `NCCL_SOCKET_IFNAME` in particular needs to match
your actual interface name).

### Resuming after a crash

Checkpoints (model + optimizer state + step) are written every
`pretrain.save_every_updates`, atomically, so a crash mid-save can't leave a checkpoint
that looks valid. **Relaunching the same command resumes automatically** from the newest
complete checkpoint in `pretrain.output_dir` (falling back to the previous one if the
newest is unreadable), and the loss-curve CSV keeps its history. To choose explicitly:

```bash
bash scripts/launch_pretrain.sh --resume-from models/xlsr300m-si-ta-200h-norm/checkpoint-3000
bash scripts/launch_pretrain.sh --no-resume     # ignore existing checkpoints, start over
```

Relaunching a run that already reached `max_updates` does nothing.

### Collapse detection

Codebook perplexity is checked once per `pretrain.eval_every_updates` updates. Training
stops when it stays below `monitoring.perplexity_floor` for `monitoring.consecutive_alerts`
readings in a row (defaults: 10.0 and 5, i.e. 2,500 updates at the default interval).

### Baseline comparisons

`dvc.yaml`'s `benchmark` stage only benchmarks the adapted checkpoint
(`models/selected`) — `params.yaml`'s `benchmark.upstreams` lists three
baselines too (`xlsr300m_vanilla`, `mhubert147`, `wavlm_large`) for
`report.py`'s comparison table, but they aren't wired into the DVC DAG since
they don't depend on this repo's training output. Run each once manually
before `make report`:

```bash
bash scripts/run_benchmark.sh facebook/wav2vec2-xls-r-300m asr,sid,er,asv 0,1,2 reports/bench/xlsr300m_vanilla
bash scripts/run_benchmark.sh utter-project/mHuBERT-147   asr,sid,er,asv 0,1,2 reports/bench/mhubert147
bash scripts/run_benchmark.sh microsoft/wavlm-large        asr,sid,er,asv 0,1,2 reports/bench/wavlm_large
```

`report.py` renders `—` for any upstream it can't find a `metrics.json` for,
rather than fabricating a score.

## Single-Language 100h CoPT Experiments

**Purpose.** An ablation of the joint model (100h Sinhala + 100h Tamil, 200h). Train one
model on Sinhala only and one on Tamil only — each from the *base* model, never one after
the other — and compare with the joint model to see whether joint CoPT gives positive
cross-lingual transfer, no interaction, or negative interference. Only the data changes;
the whole `pretrain` recipe (LR, schedule, batch, masking, quantizer, precision, seeds,
checkpointing, collapse monitoring) is the same.

```
base model ─┬─ Sinhala only (si-100h) → Sinhala-adapted checkpoint
            └─ Tamil only   (ta-100h) → Tamil-adapted checkpoint
```

**How the data is chosen.** `select --experiment <run>` filters the joint run's committed manifests
(`data/manifests/{train,dev}.jsonl`) to one language. So Sinhala-only is *exactly* the Sinhala
half of the joint training set (same clips and ids) and Tamil-only the Tamil half (the two
partition it), and the holdout is the same channels. It needs no catalog snapshot, Drive
index or Google sign-in, so it also works when the live sheet has changed since the joint run.
Realised hours are what the joint run realised (Sinhala 94.5h, Tamil 99.2h of the nominal
100h: Sinhala is catalog-limited), reported in `reports/<run>/selection_stats.json`. `select`
prints the summary (language, target/selected hours, clips, other-language clips = 0,
genre breakdown, passes over the audio) and fails if a clip of the other language is in
the manifest; `shard` and `train` re-check the manifest/shards.

**Training duration — same updates.** The joint run is controlled by `pretrain.max_updates`
(13,500 updates x 1,600s of audio = 6,000h seen ≈ 30 passes over 200h), not by epochs.
The single-language runs keep `max_updates` and everything tied to it (warmup, LR decay,
milestones 4500/9000/13500) unchanged, so each model gets the *same optimisation* as the
joint one. Consequence: each sees its ~95-99h about 60-63 times, i.e. ~2x the
per-language audio exposure of the joint model (which saw each language ~30 times). If you
instead want equal per-language exposure, set `pretrain.max_updates: 6750`, `warmup_updates: 500`
and `milestone_checkpoints: [2250, 4500, 6750]` for the single-language runs only (that needs
its own params file / branch — `pretrain` is shared by all runs here) and say so when
comparing.

**Equal-exposure variants (`si-100h-p30`, `ta-100h-p30`).** The standard runs double each language's
exposure (63 passes vs the joint model's 31). These variants train on the *same clips* for
**6,750 updates** (half), so each language gets about the passes over its own audio that the joint
run gave it: 31.8 passes of the 94.5h Sinhala and 30.3 of the 99.2h Tamil, against 31.0 in the
joint run (13,500 updates x 1,600 s / 193.6h). Only the schedule length differs: warmup 500 (same 7.4%
share), the tri-stage LR scales with `max_updates`, milestones 2250 / 4500 / 6750 (same 1/3, 2/3, 3/3
positions) and a checkpoint every 250 updates (it must divide every milestone). This is set per
run in `params.yaml` under `pretrain_overrides`; the schema accepts only `max_updates`,
`warmup_updates`, `milestone_checkpoints` and `save_every_updates`, so nothing else can drift.
They are independent runs with their own manifests, shards, models and reports (`xlsr300m-si-100h-p30`
etc.; the manifests are byte-identical to the standard run's, which a test checks).
Run them like any other: `make run-si-100h-p30`, `make train-ta-100h-p30`. About 40 hours each.
`results_ablation.md` lists both, so the comparison reads: frozen / joint 200h / single (equal
compute: same updates) / single (equal exposure: same passes). Reading it: if the equal-exposure
model matches the 13,500-update one, the extra passes were not helping; if the joint model beats
the equal-exposure single-language models on a language, adding the other language helped it.

**Configuration** (`params.yaml`):

```yaml
experiments:
  backbone_tag: xlsr300m
  runs:
    si-100h: {language: sinhala, target_hours: 100.0}
    ta-100h: {language: tamil, target_hours: 100.0}
```

`target_hours` must equal `select.target_hours` x that language's `language_mix` share
(validated by `schema.py`). `python -m pipeline.experiment` validates the runs and prints
their paths. All path logic lives in `src/pipeline/experiment.py`.

**Commands** (`<run>` = `si-100h` or `ta-100h`):

```bash
make run-si-100h        # select → pull → shard → train → ckpt → bench, via DVC, one stage at a time
make run-ta-100h

# or stage by stage
make select-<run>   pull-<run>   shard-<run>
make train-<run>                 # GPU job: use tmux; resumes automatically
make ckpt-<run>     bench-<run>
make report                      # also writes reports/results_ablation.{md,csv}
```

DVC stages: `select_single@<run>` → `materialise_single@<run>` → `shard_single@<run>` →
`pretrain_single@<run>` → `select_checkpoint_single@<run>` → `benchmark_single@<run>`
(`dvc repro -s <stage>@<run>`), sharing only `snapshot_catalog`/`index_drive` with the joint
pipeline. Changing a run's language or hours (or any shared section) invalidates only the
stages that depend on it.

**Where things go** (the joint run's paths are never written):

| | joint 200h | `si-100h` (same for `ta-100h`) |
|---|---|---|
| manifests | `data/manifests/{train,dev}.jsonl` | `data/manifests/si-100h/` |
| shards | `data/shars/` | `data/shars_single/si-100h/` |
| pre-trained model + checkpoints | `models/xlsr300m-si-ta-200h-norm/` | `models/xlsr300m-si-100h/` |
| selected checkpoint | `models/selected/` | `models/selected_single/xlsr300m-si-100h/` |
| selection / training reports | `reports/*.json,csv,md` | `reports/si-100h/` |
| benchmark | `reports/bench/xlsr300m_adapted_norm/` | `reports/bench/xlsr300m-si-100h/` |
| MLflow run name | `xlsr300m-si-ta-1600.0s` | `xlsr300m-copt-si-100h` |

(Shards and selected checkpoints are not nested under `data/shars/` / `models/selected/`
because DVC forbids an output inside another stage's output directory.)

**Resuming, and not resuming the wrong run.** Same as the joint run: relaunch the same
command (`make train-si-100h`) and it continues from the newest complete checkpoint in that
run's own directory; or `bash scripts/launch_pretrain.sh --experiment si-100h --resume-from
models/xlsr300m-si-100h/checkpoint-3000`, `--no-resume` to start over. Each run's directory
is stamped with `experiment.json` (run, language, base model, hash of the training
manifest). A run refuses to start if the directory carries another run's stamp, holds
checkpoints with no stamp, or the training data changed; `--resume-from` must point inside the
run's own directory, so Sinhala can never resume from a Tamil or joint-200h checkpoint.
Checkpoint selection (`select_checkpoint`, same proxy ASR eval on the same milestones) looks
only at the run's own checkpoints and writes `reports/<run>/checkpoint_selection.json`.

**Benchmarking.** SLSB has no language option: it runs task *families* (`asr,sid,er,asv`,
chosen by `data/<family>*` folders) and each result entry carries its `language`. So each
adapted checkpoint is benchmarked once with the same tasks/seeds as the joint model, into
its own `reports/bench/<name>/`, and `report.py` splits the entries by language. In
`reports/results_ablation.md` the model's own language is marked `(own)` and is the primary
result; with `benchmark.cross_language: true` the other language is shown too (a free
cross-transfer / forgetting check). SID is multilingual (one score shown under both
languages) and ER exists for Tamil only. Missing results render as `—`. Frozen-backbone and
joint results are never overwritten. To benchmark manually:
`bash scripts/run_benchmark.sh models/selected_single/xlsr300m-si-100h asr,sid,er,asv 0,1,2 reports/bench/xlsr300m-si-100h`.

**Reproducing.** Nothing but the `experiments` block and the shared sections drives the
runs: from a clean checkout with the catalog snapshot and Drive index in place,
`make run-si-100h run-ta-100h` redoes everything (selection is deterministic for a given
catalog snapshot and `select.seed`).

## Configuration

`params.yaml` is the single source of truth; every stage reads its own
section from it, and `src/pipeline/schema.py` validates the `select`,
`shard`, and `pretrain` sections with Pydantic before any stage runs (`make
test` / CI run this validation too, so a bad config is caught before it
wastes GPU hours).

## Known gaps

- **Resume restores the model, optimizer and step, not the exact data position.** The
  data loader is re-seeded with the resume step, so a resumed run re-sees some clips —
  fine for self-supervised pre-training, but not a bit-for-bit continuation.
  Multi-node resume needs the checkpoint directory on every node (a mismatch is detected
  and stops all ranks rather than letting their optimizers diverge).
- **`scripts/run_benchmark.sh`** exists to bridge slsb's real output filename
  (`results_<upstream-with-/-as-__>.json`) to the fixed `metrics.json` path
  `dvc.yaml` and `report.py` expect — see that script's header comment.
