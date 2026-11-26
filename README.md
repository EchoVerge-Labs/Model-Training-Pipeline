# Model Training Pipeline — Continued Pre-training on Sinhala/Tamil

Continued self-supervised pre-training (CoPT) of five speech encoders on Sinhala/Tamil,
each **keeping its own native SSL objective**, followed by automatic checkpoint selection
and SLSB benchmarking. One codebase, one entry point, one config per model — pick the model
with `MODEL=<key>`.

| `MODEL` key  | Base checkpoint                     | Objective family              | Name (runs / frozen baseline) |
|--------------|-------------------------------------|-------------------------------|-------------------------------|
| `mhubert147` | `utter-project/mHuBERT-147`         | masked prediction (k-means)   | `mhubert147-…` / `mhubert147-frozen` |
| `hubert`     | `facebook/hubert-large-ll60k`       | masked prediction (k-means)   | `hubert-large-…` / `hubert-large-frozen` |
| `wavlm`      | `microsoft/wavlm-large`             | masked prediction + utterance mixing | `wavlm-large-…` / `wavlm-large-frozen` |
| `xlsr`       | `facebook/wav2vec2-xls-r-300m`      | wav2vec 2.0 contrastive       | `xlsr300m-…` / `xlsr300m-frozen` |
| `wav2vec2`   | `facebook/wav2vec2-large-lv60`      | wav2vec 2.0 contrastive       | `wav2vec2-large-lv60-…` / `wav2vec2-large-lv60-frozen` |

`MODEL` has no default: every per-model target stops with an error when it is missing. A run's
full name is derived from its config, e.g. `wavlm-large-copt-si-ta-200h` (see [Run names](#run-names)).

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
(below), and MLflow logging to DagsHub (see [Where results go](#where-results-go-dagshub)).
Both can also build every optimizer update from **one language** (language-homogeneous
updates, `VARIANT=configs/variants/homog.yaml`, runs named `…-homog`): all of an update's
gradient-accumulation micro-batches, on every rank, come from the same language, chosen by a
seeded schedule that a resumed run continues exactly.

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

> **slsb:** install it from the sibling **`../SLSB-benchmark`** checkout at **v0.5.0**, which is
> what `make ckpt`/`make bench` use for the CLI, its benchmark data (`benchmark.slsb_data_dir`)
> and its leaderboard logger (`scripts/log_results.py`). `make bench` refuses any other slsb
> version (`benchmark.slsb_version`); the `[benchmark]` extra in `pyproject.toml` pins the same tag.

## Configuration

Config is split so the five models share one data definition:

- **`params-data.yaml`** — data selection + sharding, **shared by all models** (they train on
  the same shards). Read by the data stages (`index`, `select`, `pull`, `shard`).
- **`params-<model>.yaml`** — one per model: base checkpoint, objective settings, cluster
  config, checkpoint selection, and benchmark settings (`benchmark.baselines`: the frozen
  baselines in the results table). `Params.from_yaml` merges the shared `select`/`shard`
  sections in automatically. No file holds an output path: every path is derived from the
  run's name or its data id.
- **`configs/variants/*.yaml`** — overlays holding only the keys a variant changes, deep-merged
  over the params file with `VARIANT=` (see [Variants and tags](#variants-and-tags)).

`src/pipeline/common/schema.py` validates every section with Pydantic before any stage runs
(`make test` / CI run this too), so a bad config — a milestone past `max_updates`, warmup ≥
`max_updates`, an inconsistent shard window — is caught before it wastes GPU hours.

### Run names

Every run has **one name, derived from its config** (`src/pipeline/common/naming.py`) — never
typed by hand:

```
<model>-<method>-si-ta-<hours>h[-cnn][-p<N>][-homog][-<tag>]
```

| Slot | Values | From |
|---|---|---|
| model | `wavlm-large`, `hubert-large`, `mhubert147`, `xlsr300m`, `wav2vec2-large-lv60` | `MODEL` |
| method | `copt`, `adapter`, `lora`, `gclora` | `pretrain.pecopt.method` |
| languages | `si-ta` | `select.language_mix` |
| hours | `200h`, `1h`, `1000h`, … | `select.target_hours` |
| `cnn` | the CNN feature encoder trains | `pretrain.freeze_feature_encoder: false` |
| `p<N>` | passes over the data (whole passes), only when not the standard 30 | `max_updates` × `target_batch_seconds` ÷ hours |
| `homog` | language-homogeneous updates | `pretrain.language_batching.enabled` |
| tag | free text, e.g. `smoke` | `TAG=` (or `tag:` in an overlay) |

e.g. `wavlm-large-copt-si-ta-200h`, `hubert-large-copt-si-ta-200h-cnn`,
`xlsr300m-copt-si-ta-200h-homog`, `hubert-large-lora-si-ta-200h-cnn`,
`wavlm-large-copt-si-ta-1h-smoke`. `make name MODEL=… VARIANT=… TAG=…` prints it. A model's
un-adapted base is its **frozen baseline**, `<model>-frozen` (e.g. `wavlm-large-frozen`).

The name is the run's folders, its tmux session, its DagsHub training run and its SLSB
leaderboard row:

| Output | Path |
|---|---|
| Checkpoints, final model, `run.json` | `models/<name>/` |
| Selected checkpoint | `models/<name>/selected/` |
| Merged config, curves, metrics, selection report | `reports/<name>/` (`config.yaml`, `pretrain_curves.csv`, `pretrain_metrics.json`, `checkpoint_selection.json`; homog: `pretrain_language_curves.csv`) |
| Benchmark | `reports/<name>/bench/` (SLSB's files + `metrics.json`) |
| Results table | `reports/<name>/results.md`, `results_table.csv` |
| Logs | `logs/<name>/train.log`, `ckpt.log`, `bench.log` |
| DagsHub | ssl-pretraining run `<name>` and leaderboard row `<name>` |

Data shared between runs is named by its **data id**, `si-ta-<hours>h[-<tag>]`:
`data/manifests/<data id>/`, `data/shars/<data id>/`, `reports/data/<data id>/` (selection and
pull reports), `models/kmeans/<model>-L<layer>-k<k>-<data id>.joblib` and
`data/labels/<model>-L<layer>-k<k>-<data id>.jsonl.gz`.

**One name, one config.** `models/<name>/run.json` records the name's slots and a
fingerprint of the config it stands for; `make train`, `ckpt` and `bench` stop if the folder
belongs to a different config, or holds checkpoints without a `run.json`. `make select`
refuses to overwrite a selection made with other settings (`data/manifests/<data id>/selection.json`),
and `make fit-kmeans` an existing k-means model (`FORCE=1` refits on purpose).

### Why `make index` exists

The sheet's `name_in_drive` (`<hash>_NNN.wav`) is **not** unique on the Drive — the same name
recurs across different source videos with different audio. `select` therefore pins each catalog
row to a real file by `(language, genre, name, size)` against the index, records that path (not
the bare name) in the manifests, and gives every clip a unique id. Repeated sheet rows are
collapsed; rows that don't resolve to exactly one file are set aside in
`reports/data/<data id>/unresolved_clips.csv` rather than guessed.

## Running — every new terminal

`make train` / `make phase0` launch **real, hours-to-days-long GPU jobs on shared hardware.**
Run them in **`tmux`** (not `nohup`), and before launching either, check you're not already
running this pipeline as part of a larger session on the same GPU.

**Always pass `MODEL=`.** There is no default: a per-model target without `MODEL=` (`name`,
`phase0`, `fit-kmeans`, `labels`, `train`, `ckpt`, `bench`, `report`, `all`) stops with an
error instead of guessing. `make name
MODEL=<key> VARIANT=<overlays> TAG=<tag>` prints the run name; if it isn't the run you meant,
nothing below will be either.

```bash
cd ~/EchoVerge-LABS/Model-Training-Pipeline
git branch --show-current        # confirm you are on the branch you mean to run
source .venv/bin/activate        # make train needs `torchrun` from the venv
set -a; source .env; set +a      # DagsHub login for MLflow + NNODES/NODE_RANK/MASTER_* (defaults NNODES=1)
slsb --help | head -1            # slsb must resolve in THIS venv (installed from ../SLSB-benchmark)
```

## End-to-end workflow

```bash
# --- data: shared across all models, run ONCE (needs the Drive mount) ---
make snapshot                        # freeze the Google Sheet catalog -> data/catalog/catalog_latest.csv
make index                           # list every wav on the mount (relpath + size) -> drive_index.tsv
make select                          # dedupe, pin each clip to its file, split train/holdout -> data/manifests/si-ta-200h/
make pull                            # copy the selected clips off the mount -> data/raw/
make shard                           # pack into Lhotse Shar tarballs -> data/shars/si-ta-200h/

# --- per run: example WavLM (a HuBERT-family, masked-prediction model) ---
# Every command below takes the SAME MODEL/VARIANT/TAG. `export`, not a plain assignment:
# tmux starts a fresh shell, and an unexported $M would expand to nothing there --
# `make train` alone then stops with "MODEL is not set".
export M="MODEL=wavlm"               # e.g. "MODEL=wavlm VARIANT=configs/variants/cnn.yaml"

make phase0 MODEL=wavlm              # throughput + peak memory + wall-clock estimate (per model)
make fit-kmeans $M                   # HuBERT family only, once per model and data:
make labels     $M                   #   -> models/kmeans/wavlm-large-L18-k500-si-ta-200h.joblib + data/labels/

# name the tmux session after the run, so several runs are never confused
RUN=$(make -s name $M)               # -> wavlm-large-copt-si-ta-200h
tmux new -s "$RUN" "make train $M"   # the command is expanded HERE, by this shell
#   13,500 updates; checkpoints every 1,500; milestones 4500/9000/13500
#   output: models/$RUN/ ; log: logs/$RUN/train.log
#   detach: Ctrl-b d   |   re-attach: tmux attach -t $RUN
#   crashed after a checkpoint? re-run the same command -> auto-resumes from the newest one.

# confirm you launched what you meant to, before walking away
head -3 logs/$RUN/train.log          # names the run, its data and its model

# watch progress (another terminal; do NOT trust pretrain_metrics.json until it finishes)
tail -f reports/$RUN/pretrain_curves.csv   # live: step, loss, accuracy/perplexity
#   also the DagsHub ssl-pretraining run named $RUN

make ckpt   $M                       # proxy-eval the milestones, copy the best (ranked on dev score)
#   -> models/$RUN/selected/ + reports/$RUN/checkpoint_selection.json
make bench  $M                       # SLSB benchmark of the selected checkpoint
#   -> reports/$RUN/bench/ + leaderboard row $RUN on SLSB-benchmark
make report $M                       # the run vs its frozen baselines -> reports/$RUN/results.md
```

The contrastive models (`xlsr`, `wav2vec2`) are identical **minus** `fit-kmeans`/`labels`:

```bash
export M="MODEL=xlsr"
make phase0 MODEL=xlsr
tmux new -s "$(make -s name $M)" "make train $M"   # -> xlsr300m-copt-si-ta-200h
make ckpt $M && make bench $M && make report $M    # after it finishes
```

`make all MODEL=<key> VARIANT=… TAG=…` chains
`shard → [fit-kmeans → labels →] train → ckpt → bench → report` (assumes `data/raw` is already
selected and materialised). Still run it inside `tmux`.

### Variants and tags

A **variant** is an overlay in `configs/variants/` holding only what it changes; pass one or
more with `VARIANT=` (comma-separated), and the run's name follows:

| Variant | Changes | Name |
|---|---|---|
| `cnn.yaml` | the CNN feature encoder trains (full CoPT) | `…-cnn` |
| `homog.yaml` | language-homogeneous updates | `…-homog` |
| `adapter.yaml`, `lora.yaml`, `gclora.yaml` | the PE-CoPT method (templates, not implemented yet) | method slot |
| `smoke-1h.yaml` | 1 h selection, 30 passes (68 updates), ER-only benchmark, the 200 h k-means; tag `smoke` | `…-si-ta-1h-smoke` |

```bash
make train MODEL=hubert                                          # -> hubert-large-copt-si-ta-200h
make train MODEL=hubert VARIANT=configs/variants/cnn.yaml        # -> hubert-large-copt-si-ta-200h-cnn
make train MODEL=xlsr VARIANT=configs/variants/cnn.yaml,configs/variants/homog.yaml
                                                                 # -> xlsr300m-copt-si-ta-200h-cnn-homog
```

A variant may change any setting, but one the name does not show (a learning rate, a mask
probability, …) needs a **tag** — `TAG=lr1e-4` — or the stage refuses to start, so two
different runs can never share a name. A variant that changes the *selection* (like
`smoke-1h`) also gives the data its own id (`si-ta-1h-smoke`), so it never overwrites the
200 h manifests or shards. Use the same `MODEL`/`VARIANT`/`TAG` for every stage of the run.

**k-means and labels are per model and data**, not per run: every run of a model on
`si-ta-200h` shares `models/kmeans/<model>-L<layer>-k<k>-si-ta-200h.joblib` and its labels, so
`make fit-kmeans`/`make labels` run once per model and data. A run on other data labels its own
shards; `cluster.kmeans_data` can point it at another selection's k-means (the smoke variant
reuses the 200 h one).

## Training knobs

- **Memory / OOM** — run `make phase0 MODEL=<key>` first; it prints peak GPU memory and a
  wall-clock estimate for the configured `max_updates`. If training OOMs, lower
  `pretrain.per_device_max_seconds` in the model's params file (e.g. 200 → 150; gradient
  accumulation keeps the effective batch the same). `PHASE0_ARGS=--no-freeze-cnn` measures the
  trainable-CNN case.
- **CNN feature encoder** — frozen by default (`pretrain.freeze_feature_encoder: true`,
  transformer-only CoPT). `VARIANT=configs/variants/cnn.yaml` trains it too (full CoPT, the run
  is named `…-cnn`; more memory, so check `make phase0` with `PHASE0_ARGS=--no-freeze-cnn`
  first). Configurable for all five models.

## Resuming after a crash

Checkpoints (model + optimizer state + step) are written every `pretrain.save_every_updates`,
**atomically** — everything lands in a `.tmp` dir that is renamed into place, so a crash
mid-save can never leave a checkpoint that looks valid. Relaunching the same `make train`
command **resumes automatically** from the newest complete checkpoint in `models/<name>/`
(falling back to the previous one if the newest is unreadable), the loss-curve CSV keeps its
history, and the DagsHub run continues. Relaunching a run that already reached `max_updates`
does nothing. A run can only resume its own folder (`run.json`): training it longer or with
other settings is a different run, with its own name — resume it from the old checkpoint with
`--resume-from`. Steps logged between the last checkpoint and the crash are trained again,
so on DagsHub they appear twice in the run's metric history (MLflow keeps every point it was
sent); `reports/<name>/pretrain_curves.csv` keeps only the re-run values.

To choose explicitly:

```bash
bash scripts/launch_pretrain.sh --model wavlm \
    --resume-from models/wavlm-large-copt-si-ta-200h/checkpoint-9000   # a specific checkpoint
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
so the test set never influences selection. The winner is copied to `models/<name>/selected/`,
and the scores go to `reports/<name>/checkpoint_selection.json` and to the run on DagsHub.

**Completion gate.** `make ckpt` and `make bench` refuse to run until the run's pre-training has
**finished** (`reports/<name>/pretrain_metrics.json` with `completed: true`), so a half-trained
checkpoint is never selected or scored. Override with `ALLOW_INCOMPLETE=1` (e.g. a smoke test);
`FROZEN=1` and an explicit `UPSTREAM=` skip the gate. `make bench` also fails if `slsb`
left any task unfinished (it exits 0 even on a partial run), so a partial benchmark is never
recorded as complete.

## Baseline comparisons

`make report` writes `reports/<name>/results.md` (and `results_table.csv`): one row for the run
and one per frozen baseline in the model's `benchmark.baselines` (model keys), showing `—` for
any row it has no `metrics.json` for rather than fabricating a score. A frozen baseline is a
model's un-adapted base, named `<model>-frozen`; benchmark each once with `FROZEN=1`:

```bash
make bench MODEL=wavlm FROZEN=1        # microsoft/wavlm-large -> reports/wavlm-large-frozen/bench/
make bench MODEL=xlsr FROZEN=1         #                       -> reports/xlsr300m-frozen/bench/
make report MODEL=wavlm
```

Each also logs its leaderboard row (`wavlm-large-frozen`), and has no training run. Any other
checkpoint or Hugging Face id can be benchmarked with `UPSTREAM=` (results in
`reports/<OUT or its path>/bench/`); it is logged to the leaderboard only with a row name,
`NAME=`, so an ad-hoc checkpoint can never overwrite a run's row.

## Where results go (DagsHub)

Credentials: `DAGSHUB_USER` / `DAGSHUB_TOKEN` in `.env`. A run has the same name on both:
its training run in Model-Training-Pipeline and its row on the SLSB leaderboard. Checkpoints
stay on the training server, and the files in `reports/<name>/` stay there until someone
commits them.

**Training → `Model-Training-Pipeline.mlflow`, experiment `ssl-pretraining`** (`mlflow:` in the
params file), one run per training run, named `<name>`:

- tags `kind=adapted`, `family`, `base_model`, `slsb.model=<name>`, the name's slots (`model`,
  `method`, `languages`, `hours`, `cnn`, `passes`, `batching`, `data`, `tag`) and a note;
- params: the training settings, `output_dir`, `params_commit`, `trained` (start to end),
  `selected_checkpoint` once `make ckpt` ran, and `language_batching_*` for a homog run;
- metrics: the `train/*` curves every `eval_every_updates`, `lang/<language>/*` for a homog
  run, and `checkpoint_selection/<checkpoint>` (each milestone's dev score) from `make ckpt`;
- files: `params.yaml` (the merged config), `pretrain_curves.csv`, `pretrain_metrics.json`,
  `pretrain_language_curves.csv` (homog), `kmeans_stats.json` (HuBERT family), and
  `checkpoint_selection.json` from `make ckpt`.

A resumed run continues the same MLflow run (its id is in `models/<name>/mlflow_run_id.txt`). A
run ends **FINISHED** only when it reached `max_updates`; stopped early (stall / collapse
detector, kill signal) it ends **KILLED**, crashed **FAILED**. Frozen baselines have no
training run.

**Benchmark → `SLSB-benchmark.mlflow`, experiment `slsb-<protocol>`** (the leaderboard), written
by `make bench` itself as **one row per run**: `slsb run` (without live MLflow logging) writes
`reports/<name>/bench/`, and SLSB's own `scripts/log_results.py` logs it as the row `<name>`
(or `<model>-frozen` with `FROZEN=1`): every task's metric (mean over seeds) and its `_std`,
with `kind`, `family`, `base_model`, `checkpoint`, `pretrain_hours`, protocol and slsb version,
and `per_seed.csv` plus the results folder as files. A run's row then gets the same name-slot
tags as its training run. A full row covers all six task families (`benchmark.tasks`); a
variant may benchmark fewer (the smoke variant: ER only), and the row's note then says which.
A model that already has a row is left alone unless `REPLACE=1`; after a protocol release that
changed only some tasks, re-run those with `CARRY_OVER=1`. Without the credentials (or with
`NO_LOG=1`) the results stay local and the logging command is printed. `DRY_RUN=1` prints the
commands only. `benchmark.slsb_version` pins the protocol (`make bench` refuses another slsb).

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

Copy the run folder (`models/<name>/`, including its `run.json`) to every node before resuming a
multi-node run (each rank reads its own folder; a mismatch is detected and stops all ranks rather
than letting their optimizers diverge).

## DVC

`dvc.yaml` describes the data stages and the default **mHuBERT-147** run
(`mhubert147-copt-si-ta-200h`, data `si-ta-200h`) end to end (its `vars` come from
`params-mhubert147.yaml`); `dvc repro` runs whatever is stale. The other four models
are driven by `make`, not DVC — there is no `dvc-<model>.yaml`. `pipeline.eval.benchmark` bridges
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
configs/variants/           overlays: cnn, homog, adapter / lora / gclora, smoke-1h
src/pipeline/
  train.py                  CLI entry point; dispatches on objective family
  common/                   infra used everywhere (no model/family logic)
    schema.py  device.py  checkpoints.py  callbacks.py  datamodule.py
    naming.py                 run names and data ids
    tracking.py               the run's folder (run.json) and its DagsHub record (ssl-pretraining)
  models/                   one file per model (its ModelSpec) + the registry
    registry.py  spec.py  mhubert147.py  hubert.py  wavlm.py  xlsr.py  wav2vec2.py
  training/                 the engine: one trainer per objective family
    masked_prediction.py       HuBERT / mHuBERT-147 / WavLM
    contrastive.py             XLS-R / wav2vec2
    masked_model.py            backbone + linear cluster-id head
    language_batching.py       language-homogeneous updates (homog)
  pecopt/                   full_copt (default) + PE-CoPT method templates
  data/                     snapshot_catalog / drive_index / select / materialise / shard
  labels/                   fit_kmeans / assign_cluster_labels / layer_features (k-means targets)
  eval/                     select_checkpoint / benchmark (+ slsb.py) / report
scripts/
  launch_pretrain.sh        torchrun wrapper (single- and multi-node)
  phase0_throughput.py      throughput / memory probe
  nccl_bandwidth_test.sh    inter-node link check before multi-node training
```
