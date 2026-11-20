# Model Training Pipeline — Continued Pre-training of Speech Encoders on Sinhala/Tamil

DVC-orchestrated pipeline for continued self-supervised pre-training of speech encoders,
checkpoint selection, and benchmarking against
[SLSB](https://github.com/EchoVerge-Labs/SLSB-benchmark). It produces Sinhala/Tamil-specialised
speech encoders from several pretrained models, all from this one branch:

| `MODEL=` / `--model` | Base model | Trainer | Objective |
|---|---|---|---|
| `xlsr` (default) | `facebook/wav2vec2-xls-r-300m` | `xlsr` | wav2vec2 contrastive loss + Gumbel quantizer |
| `wav2vec2` | `facebook/wav2vec2-large-lv60` (English) | `xlsr` | same as XLS-R |
| `wavlm` | `microsoft/wavlm-large` | `wavlm` | masked prediction of k-means ids + utterance mixing |
| `hubert` | `facebook/hubert-large-ll60k` | `hubert` | masked prediction of k-means ids |
| `mhubert` | `utter-project/mHuBERT-147` | `hubert` | masked prediction of k-means ids |

`params.yaml` holds everything the models share (data selection, sharding, the training
recipe) and, as it always has, the XLS-R run. Every other model has
`configs/models/<model>.yaml` with only the keys it changes; `--model <model>` merges it
over `params.yaml`. A fair comparison across models is therefore true by construction: batch,
schedule, masking, learning rate and checkpoint schedule are the shared ones.

The training **method** is full continued pre-training (`pretrain.method: copt`) for now;
adapters, LoRA and GC-LoRA will be further methods of the same pipeline, not branches.

## Contents

[Layout](#layout) · [Pipeline DAG](#pipeline-dag) · [Prerequisites](#prerequisites) ·
[Quick start](#quick-start) · [What each model writes](#what-each-model-writes) ·
[Masked-prediction models](#the-masked-prediction-models-wavlm-hubert-mhubert) ·
[Running training](#running-training) · [Checkpoint selection](#checkpoint-selection) ·
[Benchmarking](#benchmarking-against-slsb) · [Configuration](#configuration) ·
[Tests and CI](#tests-and-ci) · [Troubleshooting](#troubleshooting) · [Known gaps](#known-gaps)

## Layout

```
src/pipeline/
  train.py                CLI: runs the trainer of --model
  models/
    registry.py           model name -> trainer
    xlsr/trainer.py       wav2vec2 contrastive objective (XLS-R, wav2vec2)
    wavlm/                trainer.py + model.py (WavLMForMaskedPrediction)
    hubert/               trainer.py + model.py (HubertForMaskedPrediction; HuBERT, mHuBERT)
  common/                 shared by every trainer: LR schedule, checkpoints/resume,
                          distributed helpers, config loading (+ model overlays)
  labels/                 k-means pseudo-labels for the masked-prediction models
  datamodule.py           Lhotse Shar loader (labels, normalisation, utterance mixing)
  callbacks.py            MLflow logger, codebook-collapse and masked-accuracy-stall detectors
  select.py shard.py ...  data stages, model-independent
  select_checkpoint.py    proxy-evaluates milestone checkpoints, keeps the best
  benchmark.py slsb.py    drive SLSB-benchmark (see "Benchmarking against SLSB")
  report.py               aggregates reports/bench/*/metrics.json into a table
configs/models/           one overlay per model
docs/branch-ledger.md     where every change of the old per-model branches went
```

Each trainer owns its training loop, because the objectives differ. Models whose code is
the same share a trainer and differ only by config: wav2vec2 runs the `xlsr` trainer,
mHuBERT-147 the `hubert` trainer.

## Pipeline DAG

```
snapshot_catalog → index_drive → select → materialise → shard ─┬→ pretrain → select_checkpoint → benchmark → report     (XLS-R)
                                                               ├→ copt@wav2vec2 → ckpt@wav2vec2 → bench@wav2vec2
                                                               └→ fit_kmeans@M → assign_cluster_labels@M → copt@M → ckpt@M → bench@M   (M = wavlm, hubert, mhubert)
```

Each stage is a `dvc.yaml` target and `dvc repro` runs whatever is stale given `params.yaml`
and each stage's declared deps. The XLS-R stages keep their original names. The other
models' stages are `foreach` stages named `<stage>@<model>`, e.g.
`dvc repro copt@wavlm ckpt@wavlm bench@wavlm`. Because every stage depends on the outputs of
the one before, `dvc repro bench@wavlm` chains labelling, training, checkpoint selection and
benchmarking by itself.

| Stage | `make` target | What it does |
|---|---|---|
| `snapshot_catalog` | `snapshot` | freeze the Google Sheet catalog to `data/catalog/catalog_latest.csv` |
| `index_drive` | `index` | list every wav on the Drive mount (path + size) |
| `select` | `select` | dedupe, pin clips to Drive files, choose train/holdout manifests |
| `materialise` | `pull` | copy the selected clips from the mount to `data/raw/` |
| `shard` | `shard` | pack into Lhotse Shar tarballs in `data/shars/` |
| `fit_kmeans@M` | `fit-kmeans` | k-means on a layer of the base model (wavlm, hubert, mhubert) |
| `assign_cluster_labels@M` | `labels` | a cluster id for every training frame |
| `pretrain` / `copt@M` | `train` | continued pre-training |
| `select_checkpoint` / `ckpt@M` | `ckpt` | proxy-evaluate the milestone checkpoints, keep the best |
| `benchmark` / `bench@M` | `bench` | `slsb run` on the selected checkpoint, then log to the leaderboard |
| `report` | `report` | aggregate every model's results into a table |

## Prerequisites

- NVIDIA DGX Spark (GB10, ARM64) with the NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.11-py3`)
- Google Drive with the pre-processed audio, mounted locally with `rclone mount` (default `~/google-drive`; set `select.drive_root` in `params.yaml` to the `Pre Processed Data` folder). Clips are laid out `<Lang>/<genre>/<source-dir>/<hash>_NNN.wav`
- A DagsHub account (DVC remote + MLflow tracking) — see `.env.example`
- A Google OAuth client-secret JSON (Desktop-app type) for an account that can read the "Sinhala X Tamil Voice Dataset" sheet. The first `make snapshot` opens a browser sign-in and caches the token at `~/.config/gspread/authorized_user.json`
- For benchmarking: a checkout of [SLSB-benchmark](https://github.com/EchoVerge-Labs/SLSB-benchmark) next to this repo (see [Benchmarking](#benchmarking-against-slsb))

`.env` (copy `.env.example`) holds the credentials. `make` and `scripts/launch_pretrain.sh`
both read it.

| Variable | Used for |
|---|---|
| `DAGSHUB_USER`, `DAGSHUB_TOKEN` | DVC remote, MLflow logging of training runs, logging benchmark results to the leaderboard |
| `GOOGLE_SHEET_ID`, `GOOGLE_CLIENT_SECRET_JSON` | `make snapshot` (optionally `GOOGLE_AUTH_USER_JSON` for the cached token) |
| `HF_TOKEN` | downloading base models |
| `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK` | multi-node training (defaults: one node, `127.0.0.1:29500`) |
| `SLSB_DIR`, `SLSB_BIN` | where to find SLSB (optional; default `../SLSB-benchmark`) |

## Quick start

```bash
# 1. Install (inside the NGC container, or a matching ARM64 CUDA environment)
pip install -e ".[data,benchmark,dev]"
cp .env.example .env   # fill in DAGSHUB_TOKEN, GOOGLE_SHEET_ID, GOOGLE_CLIENT_SECRET_JSON, HF_TOKEN

# 2. Measure real throughput BEFORE scoping max_updates / timelines.
#    This loads the real model and does real GPU compute -- treat it
#    like any other GPU job (see "Running training" below).
make phase0                  # XLS-R;  make phase0 MODEL=wavlm  for another model

# 3. The data stages are model-independent: run them once
make snapshot   # freeze the Google Sheet catalog to data/catalog/catalog_latest.csv
                # (or: python -m pipeline.snapshot_catalog --local-csv <sinhala.csv> <tamil.csv>)
make index      # list every wav on the Drive mount (path + size) -> data/catalog/drive_index.tsv
make select     # dedupe, pin each clip to its Drive file, select train/holdout -> data/manifests/
make pull       # copy the selected clips from the mount -> data/raw/ (same <Lang>/<genre>/... layout)
make shard      # pack into Lhotse Shar tarballs -> data/shars/

# 4. Train a model (MODEL= any of xlsr wav2vec2 wavlm hubert mhubert; unset = xlsr)
make fit-kmeans MODEL=wavlm   # hubert / wavlm / mhubert only: k-means on a layer of the base model
make labels     MODEL=wavlm   #   ... then a cluster id for every frame -> data/labels/
make train      MODEL=wavlm   # continued pre-training -> models/wavlm-large-si-ta-200h/   (run it in tmux)

# 5. Only once training has finished: pick a checkpoint, then benchmark it
make ckpt       MODEL=wavlm   # proxy-eval milestone checkpoints, copy the best -> models/selected_wavlm/
make bench      MODEL=wavlm   # slsb on it, then log to the leaderboard -> reports/bench/wavlm-large-copt200h/

# 6. Baseline and report (no training needed for the baseline)
make bench-frozen MODEL=wavlm # the pre-trained base model, benchmarked the same way
make report                   # aggregate every model into reports/results.md + results_table.csv
```

The whole of step 4–5 for one model is also `dvc repro bench@wavlm`.

Without `MODEL=` every command is the XLS-R run, exactly as before. `make help` lists all targets.

### Why `make index` exists

The sheet's `name_in_drive` (`<hash>_NNN.wav`) is **not unique** on the Drive: the same
name recurs across different source videos with different audio. `select` therefore pins
each catalog row to a real file by `(language, genre, name, size)` against the index,
records that path (not the bare name) in the manifests, and gives every clip a unique id.
Rows repeated in the sheet are collapsed, and rows that don't resolve to exactly one file
are set aside in `reports/unresolved_clips.csv` rather than guessed. Catalog exports that
carry no `size` column are matched by `(language, genre, name)` instead, when that is unique.

## What each model writes

Models never overwrite each other. The XLS-R run keeps its original locations.

| Model | Training output | Selected checkpoint | Reports | Leaderboard run (`run_name`) | Baseline run (`frozen_name`) |
|---|---|---|---|---|---|
| `xlsr` | `models/xlsr300m-si-ta-200h-norm/` | `models/selected/` | `reports/` | `xlsr300m-copt200h-norm` | `xlsr300m` |
| `wav2vec2` | `models/wav2vec2-si-ta-200h/` | `models/selected_wav2vec2/` | `reports/wav2vec2/` | `wav2vec2-large-lv60-copt200h` | `wav2vec2-large-lv60` |
| `wavlm` | `models/wavlm-large-si-ta-200h/` | `models/selected_wavlm/` | `reports/wavlm/` | `wavlm-large-copt200h` | `wavlm-large` |
| `hubert` | `models/hubert-large-si-ta-200h/` | `models/selected_hubert/` | `reports/hubert/` | `hubert-large-copt200h` | `hubert-large` |
| `mhubert` | `models/mhubert147-si-ta-200h/` | `models/selected_mhubert/` | `reports/mhubert/` | `mhubert147-copt200h` | `mhubert147` |

Inside a training output directory:

- `checkpoint-<step>/` — `config.json`, the weights, the feature extractor, `training_state.pt` (optimizer and step), and for the masked-prediction models `masked_prediction_head.pt`. The backbone is a plain Hugging Face model that `slsb --upstream` loads directly.
- The final model at the top level of the directory, written when training ends.
- Written atomically: a save becomes `checkpoint-<step>` only when it is complete, so a crash never leaves a half-written checkpoint that looks valid.
- Which checkpoints exist: one every `pretrain.save_every_updates` (1500), the newest `keep_last_n_checkpoints` (5) of the rolling ones, and always the `milestone_checkpoints` (4500, 9000, 13500).

`<reports>/` gets `pretrain_curves.csv` (loss curve), `pretrain_metrics.json` (final step and
whether the run `completed`) and `checkpoint_selection.json`. Benchmark results go to
`reports/bench/<run_name or frozen_name>/`.

## The masked-prediction models (wavlm, hubert, mhubert)

HuBERT, mHuBERT and WavLM learn by predicting a cluster id for every masked frame, not
wav2vec2-style contrastive learning. The model architecture and objective are unchanged;
this is not distillation. The cluster ids are made once, offline: `fit_kmeans` runs the
*original* checkpoint over ~40 h of the training set and clusters the output of one
transformer layer (`cluster.layer`: 18 of 24 for the large models, 9 of 12 for mHuBERT-147)
into 500 clusters; `assign_cluster_labels` assigns every training cut's frames to the
nearest centroid. Waveforms are normalised (`do_normalize`, read from the base model)
identically for labelling and training. `transformers` has no `*ForPreTraining` class for
these models and the checkpoint carries no prediction head, so a fresh linear head over the
cluster ids is added on top of the plain encoder; it is saved next to the backbone as
`masked_prediction_head.pt` and only needed to resume training.

**WavLM's utterance mixing** (`pretrain.utterance_mix_prob: 0.2`) is a data augmentation:
20% of utterances get a random crop (at most 50% of their length) of another utterance from
the same batch mixed in at -5..+5 dB, after normalisation. Cluster labels always come from
the clean audio. WavLM's relative-position bias uses more memory than HuBERT; if a step runs
out of memory, lower `pretrain.per_device_max_seconds` first (it only changes
micro-batching, not the effective batch).

**Padding mask.** Every trainer passes the attention mask for padded batches, whatever the
base model's feature extractor says (mHuBERT-147's says `return_attention_mask: false`).
That is how the per-model branches trained and how SLSB evaluates every upstream.

**Layerdrop** is each base model's own value: 0.1 for XLS-R, WavLM, HuBERT and mHuBERT, and
0.0 for wav2vec2-lv60 (set in `configs/models/wav2vec2.yaml`).

## Running training

`make train` / `make phase0` launch real, hours-to-days-long GPU jobs on
shared hardware. Run them in `tmux` (not `nohup`), and don't launch either
without checking first if you're running this pipeline as part of a larger
session. For two-node training, run `scripts/nccl_bandwidth_test.sh` on both
nodes first to confirm the ConnectX-7 link is actually being used (see
`configs/nccl-env.sh` — `NCCL_SOCKET_IFNAME` in particular needs to match
your actual interface name). Set `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`
in `.env` on each node.

Batch sizes are in **seconds of audio**, not examples: `target_batch_seconds` is the effective
batch, `per_device_max_seconds` the micro-batch, and gradient accumulation makes up the
difference.

Training runs are logged to MLflow on DagsHub (experiment `ssl-pretraining`), named after the
model's `run_name`.

### Resuming after a crash

Checkpoints (model + optimizer state + step) are written every
`pretrain.save_every_updates`. **Relaunching the same command resumes automatically** from the
newest complete checkpoint in `pretrain.output_dir` (falling back to the previous one if the
newest is unreadable), and the loss-curve CSV keeps its history. To choose explicitly:

```bash
MODEL=wavlm bash scripts/launch_pretrain.sh --resume-from models/wavlm-large-si-ta-200h/checkpoint-3000
MODEL=wavlm bash scripts/launch_pretrain.sh --no-resume     # ignore existing checkpoints, start over
```

Relaunching a run that already reached `max_updates` does nothing. Checkpoints written by
the earlier per-model branches resume unchanged.

### Collapse and stall detection

- **xlsr / wav2vec2:** codebook perplexity is checked once per `pretrain.eval_every_updates`
  updates. Training stops when it stays below `monitoring.perplexity_floor` for
  `monitoring.consecutive_alerts` readings in a row (defaults: 10.0 and 5, i.e. 2,500
  updates at the default interval).
- **wavlm / hubert / mhubert:** masked accuracy at the random-guess floor, or prediction
  perplexity near 1 (everything predicted as one cluster), for about a fifth of the run
  stops training.

A run stopped this way ends with `completed: false` in `pretrain_metrics.json`, so
checkpoint selection and benchmarking refuse to use it (see below).

## Checkpoint selection

`make ckpt MODEL=<model>` proxy-evaluates each milestone checkpoint with `slsb` on one task
family (`checkpoint_selection.proxy_task`, default `asr`; seed `proxy_seed`) and copies the best
one to the model's `models/selected*/` directory, writing `checkpoint_selection.json`.

- **It ranks by DEV scores.** SLSB selects on dev and scores test once, so selection uses the
  dev score slsb records for the proxy task (dev CER for ASR) and never looks at a test metric.
  Proxy tasks with a dev score: `asr`, `sid`, `asv`, `sd`.
- **Only milestones are scored** (a proxy eval is a full downstream fine-tune). Missing or
  incomplete milestones are skipped with a warning.
- **If every proxy eval fails nothing is selected**, rather than picking blindly.
- **It waits for pre-training to finish** (below).

### Pre-training must be complete

Selecting a checkpoint and benchmarking an adapted model only make sense after pre-training
ran to `pretrain.max_updates`, so both `make ckpt` and `make bench` check the model's
`<reports>/pretrain_metrics.json` first and stop with an error if it is missing, shows
`completed: false`, or ends before `max_updates`. Finish training (rerunning the same command
resumes it), or pass `--allow-incomplete` to go on anyway. The check does not apply to
`--frozen` baselines, which involve no training, or to an explicit `--upstream` checkpoint.

## Benchmarking against SLSB

Benchmarking is done by the [SLSB-benchmark](https://github.com/EchoVerge-Labs/SLSB-benchmark)
repository, not by code here: `pipeline.benchmark` only drives SLSB's own `slsb run` and its
`scripts/log_results.py`. SLSB measures what a **frozen** encoder knows: only a weighted sum
over its layers and a small standard head are trained per task, chosen on dev and scored once on
test, with 3 seeds.

### Setting up SLSB

- Check SLSB out next to this repo (`../SLSB-benchmark`, with its own `.venv`), or point
  `SLSB_DIR` / `SLSB_BIN` at it, or have `slsb` on `PATH`.
- It must be at the version `benchmark.slsb_version` names (currently `0.5.0`). The script
  refuses any other version, because scores from different protocol versions are not
  comparable. (`pip install -e ".[benchmark]"` installs the matching tag.)
- Its benchmark data must be present at `benchmark.slsb_data_dir` (default
  `../SLSB-benchmark/data`, DVC-tracked in that repo). Its head hyperparameters are read from
  the `params.yaml` next to that data.

### Commands

```bash
make bench        MODEL=wavlm   # the selected checkpoint (after make ckpt); all six task families, 3 seeds
make bench-frozen MODEL=wavlm   # the pre-trained base model: the baseline row (no training needed)

python -m pipeline.benchmark --model wavlm --dry-run   # print the commands, run nothing

# any other run, e.g. an experiment checkpoint
python -m pipeline.benchmark --upstream models/xlsr300m-si-100h/checkpoint-6500 \
    --name xlsr300m-copt-si-100h --family xlsr --base-model facebook/wav2vec2-xls-r-300m --checkpoint 6500

# a baseline for a model that has no config here
python -m pipeline.benchmark --frozen --upstream facebook/wav2vec2-base \
    --name wav2vec2-base --family wav2vec2 --base-model facebook/wav2vec2-base
```

Other flags: `--no-log` keeps the results local, `--replace` replaces the model's existing
leaderboard run, `--out DIR` sets the results directory, `--allow-incomplete` skips the
pre-training check. An ad-hoc `--upstream` needs `--name` (and `--family`) so it cannot
overwrite another model's row.

### What happens

1. `slsb run` over every family in `benchmark.tasks` (asr, sid, er, sd, asv, ic — all six, as a
   leaderboard row needs) and `benchmark.seeds` (0,1,2). It is run **without** `--mlflow-uri`:
   live logging would create one MLflow run per task and seed, which SLSB says are not comparable
   rows.
2. SLSB writes `results_<upstream>.json`; it is copied to `reports/bench/<name>/metrics.json`,
   the fixed path DVC and `report.py` read.
3. With `DAGSHUB_USER` / `DAGSHUB_TOKEN` set, SLSB's `scripts/log_results.py` logs the results
   **once, afterwards**, as **one MLflow run per model** in the `slsb-<protocol>` experiment of
   `https://dagshub.com/EchoVerge-LABS/SLSB-benchmark.mlflow`: the leaderboard. The run
   carries kind (`adapted` or `frozen`), family, base model, checkpoint step and hours of
   continued pre-training. Without the credentials the results stay local and the script prints
   the exact command that logs them later.

The pre-training run and its leaderboard row share a name (`model.run_name`, stored as the tag
`slsb.model` on the `ssl-pretraining` run), so each pre-training run can be matched to its
benchmark result.

### Where results go, and the report

`make bench` writes `reports/bench/<run_name>/` and `make bench-frozen`
`reports/bench/<frozen_name>/` (names in the table [above](#what-each-model-writes)).
`make report` tabulates every `reports/bench/*/metrics.json` that a model config lists under
`benchmark.upstreams`, with each row's SLSB protocol and a warning when protocols are mixed
(`--protocol v0.5` keeps one). Upstreams with no results are not shown, never filled in. The
leaderboard remains the reference comparison.

The DVC `benchmark` / `bench@M` stages write to the same directories, so DVC, `make` and `make report`
agree. (The XLS-R results of the earlier raw-trained run keep their old directory names.)

## Configuration

`params.yaml` is the shared base; `configs/models/<model>.yaml` overlays one model on it
(`--model`), and `--overlay <file>` merges a further file on top. `src/pipeline/schema.py`
validates the merged result with Pydantic before any stage runs, including that each
trainer has the settings of its own objective (`make test` / CI run this validation too,
so a bad config is caught before it wastes GPU hours).

| Section | What it controls |
|---|---|
| `select`, `shard` | data selection (200 h, 50/50 Sinhala/Tamil, hold-out) and Shar packing |
| `model` | trainer, tag, `run_name` / `frozen_name` (leaderboard names), `reports_dir` |
| `pretrain` | base model, schedule (`max_updates`, tri-stage LR), batch seconds, masking, checkpointing, `output_dir` |
| `cluster` | k-means layer, clusters and output paths (masked-prediction models) |
| `monitoring` | codebook-collapse thresholds |
| `checkpoint_selection` | proxy task, seed, data, output directory |
| `benchmark` | task families, seeds, `slsb_data_dir`, `slsb_version`, the rows `report` lists |
| `mlflow` | tracking URI and experiment for training runs |

`tests/test_model_configs.py` pins what a fair comparison needs: every model shares the
training recipe, no two models write to the same path, and each config selects the right
trainer, base model and cluster layer.

## Tests and CI

`make test` — tiny random models on CPU, no GPU needed. The two trainer families have their
own end-to-end tests (`tests/test_resume.py` for xlsr, `tests/test_masked_trainers.py` for
wavlm and hubert). `make lint` runs `ruff check` and `ruff format --check`. CI runs lint,
the config validation, the tests and a `dvc dag` check on every push and pull request.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pre-training has not finished` / `stopped at step N of M` | Training did not reach `max_updates`. Rerun the same train command (it resumes), or `--allow-incomplete`. |
| `slsb X is installed but benchmark.slsb_version is Y` | SLSB is at a different protocol version. Check out the matching tag, or change `benchmark.slsb_version` knowingly. |
| `slsb not found` | Install SLSB at `../SLSB-benchmark` or set `SLSB_DIR` / `SLSB_BIN`. |
| `... does not exist -- run make ckpt first` | No selected checkpoint yet. Run `make ckpt MODEL=...`. |
| `every proxy eval failed` | Check `checkpoint_selection.proxy_data_dir` and the slsb errors printed above it. |
| Results were NOT logged | `DAGSHUB_USER` / `DAGSHUB_TOKEN` not set. Run the command the script printed. |
| `no cluster labels for cut ...` | Shards changed after labelling. Rerun `make labels MODEL=...` after `make shard`. |
| Out of memory in a step | Lower `pretrain.per_device_max_seconds` (micro-batching only). |
| A model is missing from `make report` | Its results are not under a directory named in `benchmark.upstreams`, or `make report` ran before the benchmark. |

## Known gaps

- **Resume restores the model, optimizer and step, not the exact data position.** The
  data loader is re-seeded with the resume step, so a resumed run re-sees some clips —
  fine for self-supervised pre-training, but not a bit-for-bit continuation.
  Multi-node resume needs the checkpoint directory on every node (a mismatch is detected
  and stops all ranks rather than letting their optimizers diverge).
- **`scripts/run_benchmark.sh` and `scripts/label_benchmark_runs.py`** are the older per-task
  logging path (one MLflow run per task and seed, in this repo's own experiment). `pipeline.benchmark`
  replaces it for leaderboard runs; they stay for the single-language experiment results until
  those move to `pipeline.benchmark` too.
- **No benchmark/pre-training contamination check yet.** Nothing verifies that a benchmark test
  clip was never in the continued-pretraining audio (SLSB lists this too).
