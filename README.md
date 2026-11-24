# Model Training Pipeline — Continued Pre-training on Sinhala/Tamil

Continued self-supervised pre-training (CoPT) of five speech encoders on Sinhala/Tamil,
each **keeping its own native SSL objective**, followed by automatic checkpoint selection
and SLSB benchmarking. One codebase, one entry point, one config per model — pick the model
with `MODEL=<key>`.

| `MODEL` key  | Base checkpoint                     | Objective family              | Run prefix       |
|--------------|-------------------------------------|-------------------------------|------------------|
| `mhubert147` | `utter-project/mHuBERT-147`         | masked prediction (k-means)   | `mhubert147`     |
| `hubert`     | `facebook/hubert-large-ll60k`       | masked prediction (k-means)   | `hubert-large`   |
| `wavlm`      | `microsoft/wavlm-large`             | masked prediction + utterance mixing | `wavlm-large` |
| `xlsr`       | `facebook/wav2vec2-xls-r-300m`      | wav2vec 2.0 contrastive       | `xlsr300m`       |
| `wav2vec2`   | `facebook/wav2vec2-large-lv60`      | wav2vec 2.0 contrastive       | `wav2vec2-mono`  |

`mhubert147` is the default when `MODEL` is omitted.

## Pipeline DAG

```
snapshot → index → select → pull → shard ─┬─(HuBERT family)→ fit-kmeans → labels ─┐
                                          └─(contrastive)──────────────────────────┤
                                                                                   ▼
                                                            train → ckpt → bench → report
```

The data stages (`snapshot … shard`) are **shared by every model** and run once; everything
after is per-model (`MODEL=`). Only the HuBERT family needs `fit-kmeans` + `labels`.

## Two objective families

[`pipeline.train`](src/pipeline/train.py) dispatches to the trainer for the model's objective:

- **Masked prediction** (`mhubert147`, `hubert`, `wavlm`) — predict a k-means cluster id for
  every masked frame (cross-entropy, `ignore_index=-100`). The cluster ids are made **once,
  offline**: `fit-kmeans` runs the *original* checkpoint over a sample of the training audio and
  clusters one transformer layer's features; `labels` assigns every frame its nearest centroid.
  Training then continues that same model against those ids — not distillation, no second model
  at train time. WavLM additionally mixes a second utterance into each one
  (`pretrain.utterance_mix_prob`); the labels stay those of the clean primary.
- **Contrastive** (`xlsr`, `wav2vec2`) — the wav2vec 2.0 objective (`Wav2Vec2ForPreTraining`):
  contrastive loss over jointly-learned quantised targets, with the diversity loss and Gumbel
  temperature schedule.

Both families: reset optimizer/schedule, a low peak LR (CoPT, not from scratch), a trainable
quantizer/head, tri-stage LR (warmup → hold → exp decay), gradient accumulation to a target
seconds-of-audio batch, atomic checkpoints with auto-resume, a training-health detector
(below), and MLflow logging.

## Prerequisites

- An NVIDIA GPU with a recent PyTorch (on DGX Spark / GB10 ARM64, the NGC container
  `nvcr.io/nvidia/pytorch:25.11-py3`). Everything runs on GPU — there is no CPU path.
- **For the data stages only** (not needed to train from existing shards):
  - The pre-processed audio reachable as a local Google Drive mount (e.g. `rclone mount` at
    `~/google-drive/Pre Processed Data`), laid out `<Lang>/<genre>/<source-dir>/<hash>_NNN.wav`;
    point `select.drive_root` / `drive_index_path` / `catalog_path` in `params-data.yaml` at it.
  - A Google OAuth client-secret JSON (Desktop-app type) for an account that can read the
    "Sinhala X Tamil Voice Dataset" sheet. The first `make snapshot` opens a browser sign-in and
    caches the token at `~/.config/gspread/authorized_user.json`.
- A DagsHub account for the DVC remote + MLflow tracking — see `.env.example`.
- The `slsb` CLI, for `make ckpt` / `make bench` (see Setup).

## Setup

```bash
cd ~/EchoVerge-LABS/Model-Training-Pipeline
python -m venv .venv && source .venv/bin/activate
make setup                        # pip install -e ".[data,dev]"
pip install -e ../SLSB-benchmark   # the `slsb` CLI, for make ckpt / bench (see note)
cp .env.example .env              # then fill DAGSHUB_TOKEN, GOOGLE_SHEET_ID,
                                  #           GOOGLE_CLIENT_SECRET_JSON, HF_TOKEN
```

> **slsb:** install it from the sibling **`../SLSB-benchmark`** checkout (v0.5), which is what
> `make ckpt`/`make bench` use for both the CLI and its benchmark data (`benchmark.slsb_data_dir`).
> Do **not** use the `[benchmark]` extra in `pyproject.toml` — it still pins slsb v0.1.0, which
> predates the `details` dev-score fields `select_checkpoint` ranks on.

## Configuration

Config is split so the five models share one data definition:

- **`params-data.yaml`** — data selection + sharding, **shared by all models** (they train on
  the same shards). Read by the data stages (`index`, `select`, `materialise`, `shard`).
- **`params-<model>.yaml`** — one per model: base checkpoint, objective settings, cluster/label
  config, checkpoint selection, and benchmark upstreams. `Params.from_yaml` merges the shared
  `select`/`shard` sections in automatically.

`src/pipeline/common/schema.py` validates every section with Pydantic before any stage runs
(`make test` / CI run this too), so a bad config — a milestone past `max_updates`, warmup ≥
`max_updates`, an inconsistent shard window — is caught before it wastes GPU hours.

### Runs and experiment names

A **run** is `<model>-<experiment>`, e.g. `wavlm-large-si-ta-200h`. The experiment name comes
from `experiment:` in the model's params file, overridable with `EXPERIMENT=<name>` (make) or
`--experiment <name>` (CLI). The run name replaces `{run}` throughout the params file and names
**every output**, so two experiments on one model — or the same experiment on two models — never
share a path:

| Output              | Path                                          |
|---------------------|-----------------------------------------------|
| Checkpoints         | `models/<run>/`                               |
| Selected checkpoint | `models/selected-<run>/`                      |
| Loss curves/metrics | `reports/pretrain_curves-<run>.csv`, `pretrain_metrics-<run>.json` |
| k-means stats       | `reports/kmeans_stats-<run>.json`             |
| Selection report    | `reports/checkpoint_selection-<run>.json`     |
| Benchmark           | `reports/bench/<run>/metrics.json`            |
| Results table       | `reports/results-<run>.md`                    |
| Training log        | `logs/<run>.log`                              |
| MLflow run          | named `<run>`                                 |

> **`EXPERIMENT` only labels the outputs — it does not change what is trained.** The data,
> language mix and number of updates come from `params-data.yaml` (`select.target_hours`,
> `select.language_mix`) and the model's `pretrain.max_updates`/`milestone_checkpoints`. For a
> genuinely different run (e.g. 100 h single-language), edit those values and give it a matching
> `EXPERIMENT` name so the folders say what they contain.

### Why `make index` exists

The sheet's `name_in_drive` (`<hash>_NNN.wav`) is **not** unique on the Drive — the same name
recurs across different source videos with different audio. `select` therefore pins each catalog
row to a real file by `(language, genre, name, size)` against the index, records that path (not
the bare name) in the manifests, and gives every clip a unique id. Repeated sheet rows are
collapsed; rows that don't resolve to exactly one file are set aside in
`reports/unresolved_clips.csv` rather than guessed.

## Running — every new terminal

`make train` / `make phase0` launch **real, hours-to-days-long GPU jobs on shared hardware.**
Run them in **`tmux`** (not `nohup`), and before launching either, check you're not already
running this pipeline as part of a larger session on the same GPU.

**Always pass `MODEL=`.** Every per-model target defaults to `MODEL=mhubert147`, so a `make
train` that loses its `MODEL=` trains mHuBERT-147 — silently, and for days. `make run-name
MODEL=<key> EXPERIMENT=<name>` prints the run name; if it isn't the model you meant, nothing
below will be either.

```bash
cd ~/EchoVerge-LABS/Model-Training-Pipeline
git branch --show-current        # confirm you are on the branch you mean to run
source .venv/bin/activate        # the Makefile calls plain `python`
set -a; source .env; set +a      # DagsHub login for MLflow + NNODES/NODE_RANK/MASTER_* (defaults NNODES=1)
slsb --help | head -1            # slsb must resolve in THIS venv (installed from ../SLSB-benchmark)
```

## End-to-end workflow

```bash
# --- data: shared across all models, run ONCE (needs the Drive mount) ---
make snapshot                        # freeze the Google Sheet catalog -> data/catalog/catalog_latest.csv
make index                           # list every wav on the mount (relpath + size) -> drive_index.tsv
make select                          # dedupe, pin each clip to its file, split train/holdout -> data/manifests/
make pull                            # copy the selected clips off the mount -> data/raw/
make shard                           # pack into Lhotse Shar tarballs -> data/shars/

# --- per model: example WavLM (a HuBERT-family, masked-prediction model) ---
# Pick the model with MODEL=, and name this run with EXPERIMENT= (omit it for the
# params file's default). Every command below takes the SAME MODEL/EXPERIMENT pair.
# `export`, not a plain assignment: tmux starts a fresh shell, and an unexported $M
# would expand to nothing there -- `make train` alone falls back to MODEL=mhubert147
# and trains the wrong model for days without complaining.
export M="MODEL=wavlm EXPERIMENT=si-ta-200h"

make phase0 MODEL=wavlm              # throughput + peak memory + wall-clock estimate (per model)
make fit-kmeans $M                   # HuBERT family only: k-means on the base model's features
make labels     $M                   # HuBERT family only: cluster id per frame -> data/labels/

# name the tmux session after the run, so several runs are never confused
RUN=$(make run-name $M)              # -> wavlm-large-si-ta-200h
tmux new -s "$RUN" "make train $M"   # the command is expanded HERE, by this shell
#   13,500 updates; checkpoints every 1,500; milestones 4500/9000/13500
#   output: models/wavlm-large-si-ta-200h/ ; log: logs/wavlm-large-si-ta-200h.log
#   detach: Ctrl-b d   |   re-attach: tmux attach -t wavlm-large-si-ta-200h
#   crashed after a checkpoint? re-run the same command -> auto-resumes from the newest one.

# confirm you launched what you meant to, before walking away
head -3 logs/$RUN.log                # names the model and its params file

# watch progress (another terminal; do NOT trust pretrain_metrics-*.json until it finishes)
tail -f reports/pretrain_curves-wavlm-large-si-ta-200h.csv   # live: step, loss, accuracy/perplexity
#   also the DagsHub MLflow run named "wavlm-large-si-ta-200h"

make ckpt   $M                       # proxy-eval the milestones, copy the best (ranked on dev score)
#   -> models/selected-wavlm-large-si-ta-200h/ + reports/checkpoint_selection-<run>.json
make bench  $M                       # SLSB benchmark of the selected checkpoint
#   -> reports/bench/wavlm-large-si-ta-200h/metrics.json (+ MLflow runs, if configured)
make report $M                       # aggregate -> reports/results-<run>.md + results_table-<run>.csv
```

The contrastive models (`xlsr`, `wav2vec2`) are identical **minus** `fit-kmeans`/`labels`:

```bash
export M="MODEL=xlsr EXPERIMENT=si-ta-200h-norm"
make phase0 MODEL=xlsr
tmux new -s "$(make run-name $M)" "make train $M"   # -> xlsr300m-si-ta-200h-norm
make ckpt $M && make bench $M && make report $M     # after it finishes
```

`make all MODEL=<key> EXPERIMENT=<name>` chains
`shard → [fit-kmeans → labels →] train → ckpt → bench → report` (assumes `data/raw` is already
selected and materialised). Still run it inside `tmux`.

### Several experiments on one model

`EXPERIMENT=` is how you keep runs of the *same* model apart — e.g. a frozen-CNN run and a
CNN-unfreeze run of HuBERT. Each gets its own checkpoints, reports, benchmark folder, log and
MLflow run, so nothing is overwritten and you can run them side by side on different servers:

```bash
make train MODEL=hubert EXPERIMENT=si-ta-200h       # -> models/hubert-large-si-ta-200h/
make train MODEL=hubert EXPERIMENT=cnn-unfreeze     # -> models/hubert-large-cnn-unfreeze/
```

Use the same `MODEL`/`EXPERIMENT` pair for every stage of that run (`make run-name` prints the
resulting name, which is also the tmux session name to use).

**k-means and labels are per *model*, not per experiment** — `cluster.kmeans_output` and
`labels_path` have no `{run}` in them, so all experiments of a model share one set of cluster
targets (`models/kmeans/<model>-L<layer>-k<k>.joblib`, `data/labels/<model>_train_labels.jsonl.gz`).
That is correct while `EXPERIMENT` only relabels outputs and the training data is unchanged —
you run `make fit-kmeans`/`make labels` **once per model**, not once per experiment. If you ever
change the *data* for an experiment, give that run its own `cluster.*` paths too, or it will
train against the other run's targets.

## Training knobs

- **Memory / OOM** — run `make phase0 MODEL=<key>` first; it prints peak GPU memory and a
  wall-clock estimate for the configured `max_updates`. If training OOMs, lower
  `pretrain.per_device_max_seconds` in the model's params file (e.g. 200 → 150; gradient
  accumulation keeps the effective batch the same). `PHASE0_ARGS=--no-freeze-cnn` measures the
  trainable-CNN case.
- **CNN feature encoder** — frozen by default (`pretrain.freeze_feature_encoder: true`,
  transformer-only CoPT). Set it `false` for full CoPT where the CNN trains too (more memory;
  check `make phase0` first). Configurable for all five models.

## Resuming after a crash

Checkpoints (model + optimizer state + step) are written every `pretrain.save_every_updates`,
**atomically** — everything lands in a `.tmp` dir that is renamed into place, so a crash
mid-save can never leave a checkpoint that looks valid. Relaunching the same `make train`
command **resumes automatically** from the newest complete checkpoint in `pretrain.output_dir`
(falling back to the previous one if the newest is unreadable), and the loss-curve CSV keeps its
history. Relaunching a run that already reached `max_updates` does nothing.

To choose explicitly:

```bash
bash scripts/launch_pretrain.sh --model wavlm \
    --resume-from models/wavlm-large-si-ta-200h/checkpoint-9000   # a specific checkpoint
bash scripts/launch_pretrain.sh --model wavlm --no-resume          # ignore checkpoints, start over
```

Resume restores the model, optimizer and step, **not** the exact data position: the loader is
re-seeded with the resume step, so a resumed run re-sees some clips — fine for SSL pre-training,
but not a bit-for-bit continuation.

## Training-health detection

The trainer stops a run that is clearly going nowhere, counted once per
`pretrain.eval_every_updates` updates:

- **Contrastive** (`xlsr`, `wav2vec2`) — codebook collapse: training stops when codebook
  perplexity stays below `monitoring.perplexity_floor` for `monitoring.consecutive_alerts`
  readings in a row (defaults 10.0 and 5 → 2,500 updates at the default interval).
- **Masked prediction** (`mhubert147`, `hubert`, `wavlm`) — a stall: masked-frame accuracy at
  the random-guess floor, or prediction perplexity near 1 (one cluster predicted for
  everything), for the configured number of consecutive checks.

## Checkpoint selection

`make ckpt` evaluates the milestone checkpoints with a fast SLSB proxy task
(`checkpoint_selection.proxy_task`) and keeps the best, **ranked on the dev score** slsb records
in each result's `details` (dev CER for ASR, dev accuracy for SID, …) — never the test metric,
so the test set never influences selection. The winner is copied to `models/selected-<run>/`.

**Completion gate.** `make ckpt` and `make bench` refuse to run until the run's pre-training has
**finished** (`reports/pretrain_metrics-<run>.json` with `completed: true`), so a half-trained
checkpoint is never selected or scored. Override with `ALLOW_INCOMPLETE=1` (e.g. a smoke test);
benchmarking an explicit `UPSTREAM=` baseline skips the gate. `make bench` also fails if `slsb`
left any task unfinished (it exits 0 even on a partial run), so a partial benchmark is never
recorded as complete.

## Baseline comparisons

`make report` renders one row per upstream in the model's `benchmark.upstreams` (its adapted run
plus the un-adapted baselines), showing `—` for any upstream it can't find a `metrics.json` for
rather than fabricating a score. `make bench` only benchmarks the adapted checkpoint, so run each
baseline once first with `UPSTREAM=`:

```bash
make bench MODEL=wavlm UPSTREAM=microsoft/wavlm-large           # the un-adapted base
make bench MODEL=wavlm UPSTREAM=utter-project/mHuBERT-147       # any cross-model baseline
# ... then:
make report MODEL=wavlm
```

Baselines are shared across models and only need running once each. `report.py` looks a result
up by its **`benchmark.upstreams` key**, so benchmarking a configured baseline by its path
automatically writes to that key's folder (`UPSTREAM=microsoft/wavlm-large` →
`reports/bench/wavlm_large/`) and the table picks it up. Any other upstream gets a folder derived
from its path (`…/checkpoint-9000` → `reports/bench/<run>-checkpoint-9000/`); add `NAME=<folder>`
to override.

## Multi-node (DDP)

Training uses `torchrun` via [`scripts/launch_pretrain.sh`](scripts/launch_pretrain.sh).
Gradients are normalised once per update by the masked-position count across all micro-batches
and ranks; ranks verify they agree on the resume step before starting. One shared `.env` supplies
defaults; shell-provided values win, so per node (inside `tmux` on each):

```bash
# first, confirm the inter-node link is actually used (ConnectX-7); see configs/nccl-env.sh,
# NCCL_SOCKET_IFNAME must match your real interface name:
MASTER_ADDR=10.8.100.40 NODE_RANK=0 bash scripts/nccl_bandwidth_test.sh   # on node 0
MASTER_ADDR=10.8.100.40 NODE_RANK=1 bash scripts/nccl_bandwidth_test.sh   # on node 1

# then train:
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.8.100.40 make train MODEL=wavlm       # node 0 (rendezvous host)
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.8.100.40 make train MODEL=wavlm       # node 1
```

Copy the checkpoint directory to every node before resuming a multi-node run (each rank reads its
own `output_dir`; a mismatch is detected and stops all ranks rather than letting their optimizers
diverge).

## DVC

`dvc.yaml` describes the data stages and the default **mHuBERT-147** run end to end (its `vars`
come from `params-mhubert147.yaml`); `dvc repro` runs whatever is stale. The other four models
are driven by `make`, not DVC — there is no `dvc-<model>.yaml`. `scripts/run_benchmark.sh` bridges
slsb's real output filename (`results_<upstream-with-/-as-__>.json`) to the fixed `metrics.json`
path `dvc.yaml` and `report.py` expect. `dvc.lock` may be stale after config renames — **don't
`dvc repro` blindly**, as it can retrain for hours.

## Testing

```bash
make test     # pytest (GPU-only: builds models on CUDA, no CPU fallbacks)
make lint     # ruff check + format --check
```

## Layout

Code is grouped by concern; a model differs from another only by its registry spec and its
`params-<model>.yaml`, so there is no per-model trainer — the engine is per objective *family*.

```
params-data.yaml            shared data selection + sharding
params-<model>.yaml         one per model: base, objective, cluster, selection, benchmark
src/pipeline/
  train.py                  CLI entry point; dispatches on objective family
  common/                   infra used everywhere (no model/family logic)
    schema.py  device.py  checkpoints.py  callbacks.py  datamodule.py
  models/                   one file per model (its ModelSpec) + the registry
    registry.py  spec.py  mhubert147.py  hubert.py  wavlm.py  xlsr.py  wav2vec2.py
  training/                 the engine: one trainer per objective family
    masked_prediction.py       HuBERT / mHuBERT-147 / WavLM
    contrastive.py             XLS-R / wav2vec2
    masked_model.py            backbone + linear cluster-id head
  pecopt/                   full_copt (default) + PE-CoPT method templates
  data/                     snapshot_catalog / drive_index / select / materialise / shard
  labels/                   fit_kmeans / assign_cluster_labels / layer_features (k-means targets)
  eval/                     select_checkpoint / benchmark / report
scripts/
  launch_pretrain.sh        torchrun wrapper (single- and multi-node)
  phase0_throughput.py      throughput / memory probe
  run_benchmark.sh          slsb wrapper used by make bench
  nccl_bandwidth_test.sh    inter-node link check before multi-node training
```
