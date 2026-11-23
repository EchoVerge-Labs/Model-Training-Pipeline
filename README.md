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

## Two objective families

The model's objective decides which trainer runs ([`pipeline.train`](src/pipeline/train.py)
dispatches on it):

- **Masked prediction** (`mhubert147`, `hubert`, `wavlm`) — predict a k-means cluster id for
  every masked frame (cross-entropy, `ignore_index=-100`). The cluster ids are made **once,
  offline** before training: `fit_kmeans` runs the *original* checkpoint over a sample of the
  training audio and clusters one transformer layer's features; `assign_cluster_labels` labels
  every frame with its nearest centroid. Training then continues that same model against those
  ids — not distillation, no second model at train time. WavLM additionally mixes a second
  utterance into each one (`pretrain.utterance_mix_prob`); the labels stay those of the clean
  primary.
- **Contrastive** (`xlsr`, `wav2vec2`) — the wav2vec 2.0 objective (`Wav2Vec2ForPreTraining`):
  contrastive loss over jointly-learned quantised targets, with the diversity loss and Gumbel
  temperature schedule. A codebook-collapse detector stops a dead run.

Both families: reset optimizer/schedule, a low peak LR (CoPT, not from scratch), a trainable
quantizer/head, tri-stage LR (warmup → hold → exp decay), gradient accumulation to a target
seconds-of-audio batch, atomic checkpoints with auto-resume, and MLflow logging.

## Setup

```bash
cd ~/EchoVerge-LABS/Model-Training-Pipeline
python -m venv .venv && source .venv/bin/activate
make setup                      # pip install -e ".[data,dev]"
pip install -e ../SLSB-benchmark  # the `slsb` CLI, for make ckpt / bench (see note)
cp .env.example .env            # then fill in the DagsHub / Google / HF values
```

> **slsb:** install it from the sibling **`../SLSB-benchmark`** checkout (v0.5), which is what
> `make ckpt`/`make bench` use for both the CLI and its benchmark data (`benchmark.slsb_data_dir`).
> Do **not** use the `[benchmark]` extra in `pyproject.toml` — it still pins slsb v0.1.0, which
> predates the `details` dev-score fields `select_checkpoint` ranks on.

Every `make` target activates nothing for you — **`source .venv/bin/activate` first**, and
`source .env` (or let the Makefile read it) for DagsHub/MLflow and multi-node settings.

**Prerequisites for the data stages** (not needed to train from existing shards): the
pre-processed audio must be reachable as a local Google Drive mount (e.g. `rclone mount` at
`~/google-drive/Pre Processed Data`), and `select.drive_root` / `drive_index_path` /
`catalog_path` in `params-data.yaml` must point at it. `make snapshot` needs `GOOGLE_SHEET_ID`.

## Configuration

Config is split so the five models share one data definition:

- **`params-data.yaml`** — data selection + sharding, **shared by all models** (they train on
  the same shards). Read by the data stages (`index`, `select`, `materialise`, `shard`).
- **`params-<model>.yaml`** — one per model (`params-mhubert147.yaml`, `params-hubert.yaml`,
  `params-wavlm.yaml`, `params-xlsr.yaml`, `params-wav2vec2.yaml`): the base checkpoint,
  objective settings, cluster/label config, checkpoint selection and benchmark upstreams.
  `Params.from_yaml` merges the shared `select`/`shard` sections in automatically.

### Runs and experiment names

A **run** is `<model>-<experiment>`, e.g. `wavlm-large-si-ta-200h`. The experiment name comes
from `experiment:` in the model's params file, overridable per invocation with
`EXPERIMENT=<name>` (make) or `--experiment <name>` (CLI). The run name is substituted for
`{run}` throughout the params file and names **every output**, so two experiments on one model —
or the same experiment on two models — never share a path:

> **`EXPERIMENT` only labels the outputs — it does not change what is trained.** The data,
> language mix and number of updates come from `params-data.yaml` (`select.target_hours`,
> `select.language_mix`) and the model's `pretrain.max_updates`/`milestone_checkpoints`. For a
> genuinely different run (e.g. 100 h single-language), edit those values and give it a matching
> `EXPERIMENT` name so the folders say what they contain.

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

```bash
make train MODEL=wavlm EXPERIMENT=si-100h     # -> models/wavlm-large-si-100h, etc.
make train MODEL=wavlm                         # -> models/wavlm-large-si-ta-200h (the default)
```

## End-to-end workflow

The data stages (`index` → `select` → `materialise` → `shard`) are **shared** and run once;
they read `params-data.yaml`. Everything after is per-model and takes `MODEL=` (and optionally
`EXPERIMENT=`). The HuBERT family also needs `fit-kmeans` + `labels` before training.

```bash
# 0. every new terminal
cd ~/EchoVerge-LABS/Model-Training-Pipeline
source .venv/bin/activate
set -a; source .env; set +a          # DagsHub login + NNODES/NODE_RANK/MASTER_* (defaults NNODES=1)

# --- data (shared across all models; run once) ---
make snapshot                        # pull the clip catalog from the Google Sheet
make index                           # index every wav on the Drive mount (relpath + size)
make select                          # choose the Sinhala/Tamil training subset
make pull                            # copy the selected clips off the Drive mount -> data/raw/
make shard                           # pack into Lhotse Shar tarballs -> data/shars/

# --- per model (example: WavLM) ---
make phase0 MODEL=wavlm              # optional: measure GPU throughput / memory first
make fit-kmeans MODEL=wavlm          # HuBERT family only: k-means on the base model's features
make labels   MODEL=wavlm            # HuBERT family only: cluster id per frame -> data/labels/

tmux new -s wavlm                    # so the run survives a disconnect
make train MODEL=wavlm               # continued pre-training (run INSIDE tmux)
#   13,500 updates; checkpoints every 1,500; milestones 4500/9000/13500
#   output: models/wavlm-large-si-ta-200h/ ; log: logs/wavlm-large-si-ta-200h.log
#   live progress: tail -f reports/pretrain_curves-wavlm-large-si-ta-200h.csv
#   crash/disconnect? just `make train MODEL=wavlm` again -> auto-resumes from the newest
#   complete checkpoint in the output dir.

make ckpt   MODEL=wavlm              # pick the best milestone, ranked on the dev score
#   -> models/selected-wavlm-large-si-ta-200h/ + reports/checkpoint_selection-<run>.json
make bench  MODEL=wavlm              # SLSB benchmark of the selected checkpoint
#   -> reports/bench/wavlm-large-si-ta-200h/metrics.json (+ MLflow runs, if configured)
make report MODEL=wavlm              # aggregate -> reports/results-<run>.md
```

`make all MODEL=wavlm` runs `shard → [fit-kmeans → labels →] train → ckpt → bench → report`
in order (it assumes `data/raw` is already selected and materialised).

The contrastive models (`xlsr`, `wav2vec2`) are identical minus `fit-kmeans`/`labels`:

```bash
make phase0 MODEL=xlsr
make train  MODEL=xlsr               # tmux recommended
make ckpt   MODEL=xlsr && make bench MODEL=xlsr && make report MODEL=xlsr
```

### Training knobs

- **Memory / OOM** — run `make phase0 MODEL=<key>` first. If training OOMs, lower
  `pretrain.per_device_max_seconds` in the model's params file (gradient accumulation keeps the
  effective batch the same). `make phase0 MODEL=<key> PHASE0_ARGS=--no-freeze-cnn` measures the
  trainable-CNN case.
- **CNN feature encoder** — frozen by default (`pretrain.freeze_feature_encoder: true`,
  transformer-only CoPT). Set it `false` for full CoPT where the CNN trains too (more memory;
  check `make phase0` first). Configurable for all five models.
- **Resume** — automatic: re-running `make train` resumes from the newest complete checkpoint.
  To control it directly:
  ```bash
  bash scripts/launch_pretrain.sh --model wavlm \
      --resume-from models/wavlm-large-si-ta-200h/checkpoint-9000   # a specific checkpoint
  bash scripts/launch_pretrain.sh --model wavlm --no-resume          # force a clean restart
  ```
- **Progress** — trust `reports/pretrain_curves-<run>.csv` (updated live) and the MLflow run;
  `reports/pretrain_metrics-<run>.json` is only final once the run completes.

### Benchmarking another checkpoint

`make bench` benchmarks the run's selected checkpoint by default. To benchmark any other
checkpoint or a Hugging Face id (e.g. the un-adapted baseline), into its own folder:

```bash
make bench MODEL=wavlm UPSTREAM=models/wavlm-large-si-ta-200h/checkpoint-9000
make bench MODEL=wavlm UPSTREAM=microsoft/wavlm-large
```

## Checkpoint selection

`make ckpt` evaluates the milestone checkpoints with a fast SLSB proxy task
(`checkpoint_selection.proxy_task`) and keeps the best, **ranked on the dev score** slsb records
in each result's `details` (dev CER for ASR, dev accuracy for SID, …) — never the test metric,
so the test set never influences selection. The winner is copied to `models/selected-<run>/`.

## Multi-node (DDP)

Training uses `torchrun` via [`scripts/launch_pretrain.sh`](scripts/launch_pretrain.sh).
Gradients are normalised once per update by the masked-position count across all
micro-batches and ranks; ranks verify they agree on the resume step before starting. One shared
`.env` supplies defaults; shell-provided values win, so per node:

```bash
# node 0 (rendezvous host)
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.8.100.40 make train MODEL=wavlm
# node 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.8.100.40 make train MODEL=wavlm
```

Copy the checkpoint directory to every node before resuming a multi-node run (each rank reads
its own `output_dir`).

## DVC

`dvc.yaml` describes the data stages and the default **mHuBERT-147** run end to end (its
`vars` come from `params-mhubert147.yaml`); `dvc repro` runs whatever is stale. The other four
models are driven by `make`, not DVC — there is no `dvc-<model>.yaml`. The `dvc.lock` may be
stale after config renames; **don't `dvc repro` blindly**, as it can retrain for hours.

## Testing

```bash
make test     # pytest (GPU-only: builds models on CUDA, no CPU fallbacks)
make lint     # ruff check + format --check
```

## Layout

```
params-data.yaml            shared data selection + sharding
params-<model>.yaml         one per model: base, objective, selection, benchmark
src/pipeline/
  train.py                  entry point; dispatches on objective family
  train_masked_prediction.py   HuBERT / mHuBERT-147 / WavLM trainer
  train_contrastive.py         XLS-R / wav2vec2 trainer
  masked_prediction_model.py   backbone + linear cluster-id head
  datamodule.py             Shar streaming, seconds-budgeted batches, labels, mixing
  fit_kmeans.py / assign_cluster_labels.py / layer_features.py   offline k-means targets
  select.py / materialise.py / shard.py / drive_index.py / snapshot_catalog.py   data prep
  select_checkpoint.py      proxy eval, dev-score ranking
  benchmark.py / report.py  SLSB benchmark driver + results table
  checkpoints.py            atomic save/resume helpers (shared by both trainers)
  model_registry.py / schema.py / device.py / callbacks.py
scripts/
  launch_pretrain.sh        torchrun wrapper (single- and multi-node)
  phase0_throughput.py      throughput / memory probe
  run_benchmark.sh          slsb wrapper used by make bench
```
