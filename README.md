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

## Joint 200h CoPT with Language-Homogeneous Updates

**Question:** does keeping every optimizer update single-language improve joint
Sinhala–Sri Lankan Tamil continued pre-training, compared with the original mixed-language batches?

The model is still **one shared backbone with one optimizer, trained on both languages**. The
only change is how an update's batch is built:

```
ORIGINAL                         NEW (language-homogeneous)
update 1: Sinhala + Tamil clips  update 1: Sinhala clips only
update 2: Sinhala + Tamil clips  update 2: Tamil clips only
update 3: Sinhala + Tamil clips  update 3: Tamil clips only   (balanced random order)
```

Everything else is the original run's by construction: the run is `params.yaml` plus a small
overlay, [configs/language_homogeneous.yaml](configs/language_homogeneous.yaml), which only sets
batching and output locations. Same clips (the shards in `data/shars/` are reused as is, nothing is
re-selected or re-sharded), same preprocessing, optimizer, LR schedule, masking, native wav2vec2
objective, audio-seconds per update, `max_updates`, checkpoint/eval frequency, seed, checkpoint
selection and SLSB settings. The original path is untouched: `language_batching.enabled` defaults to
`false`, and `make train` / `ckpt` / `bench` behave as before.

**Branch:** `experiment/xlsr-200h-language-homogeneous-batches` (from `main`).

```bash
make check-homogeneous     # clips == original run's? writes reports/language_batching_dataset_check.json
make train-homogeneous     # check + launch (GPU; resumes automatically). On two nodes set NNODES/NODE_RANK/MASTER_ADDR as for `make train`
make ckpt-homogeneous      # same proxy-eval selection as `make ckpt`
make bench-homogeneous     # same SLSB tasks/seeds/data as `make bench` (asr,sid,er,asv; Sinhala and Tamil)
python -m pipeline.report  # frozen vs joint mixed vs joint language-homogeneous, with deltas
```

| Output | Path |
|---|---|
| model, checkpoints | `models/xlsr300m-si-ta-200h-homogeneous/` |
| selected checkpoint | `models/selected-homogeneous/` |
| curves, per-language curves, metrics | `reports/language_homogeneous/` (`pretrain_curves.csv`, `pretrain_language_curves.csv`, `pretrain_metrics.json`, `checkpoint_selection.json`) |
| benchmark | `reports/bench/xlsr300m_si_ta_200h_homogeneous/` |
| MLflow run | `xlsr300m-copt-si-ta-200h-homogeneous-batches` |

`pipeline.variant.assert_isolated` refuses to start if any of these would point at the original
run's paths.

### How it works

- **Streams.** One stream per language reads the *same* shar directory and keeps only cuts whose
  `language` matches (set from the manifest by `shard.py`). Each stream packs batches with the
  same `per_device_max_seconds` budget as the mixed one, so a homogeneous micro-batch is the same
  size as a mixed one. Streaming is preserved; nothing is loaded into RAM or duplicated on disk.
  (A stream skips over the other language's tar entries, so it reads about twice the bytes it uses.)
- **Schedule.** `balanced_random` (default): updates are grouped in blocks of `window` (20), each
  block holds exactly 10 Sinhala + 10 Tamil updates in a seeded random order — so the split is
  exactly 50/50 over any whole number of blocks and a run of one language is at most 20 updates.
  `alternate` is strict round-robin. Both are a pure function of `(seed, update index)`.
- **DDP.** Every rank computes the same language for update *N* from `(seed, N)` — no
  communication needed — and each update starts with a tiny all-reduce (min/max of the language
  index) that stops *all* ranks with `MixedLanguageError` if any rank disagrees.
- **Gradient accumulation.** The language is chosen once per optimizer update and all
  `grad_accum_steps` micro-batches (4 on two nodes, 8 on one) are pulled from that language's
  stream. Each micro-batch's `languages` tag is checked against the update's language, and the
  accumulation window is checked again before `optimizer.step()`; any violation stops training.
- **Resume.** The language for step *N* depends only on the global step and the seed, so a resumed
  run continues the exact sequence an uninterrupted run would have; the per-language update counts
  are recomputed from the step, and audio-hours totals are stored in `training_state.pt`. As in the
  original run, the data loader position is *not* restored (see Known gaps).
- **Same-clips guard.** At launch, `data/manifests/train.jsonl` is compared by clip id, per language,
  with `git:main:data/manifests/train.jsonl`, and per-language hours in the shars are compared with
  the manifest. Any difference stops the run before any GPU work.

The selected data is ~94.5 h Sinhala and ~99.2 h Tamil (not exactly 100/100), so with 50/50
updates the *update* and audio-seconds-per-update exposure is balanced, while an epoch of the
smaller language comes round slightly more often.

### Verifying no mixed update occurred

`reports/language_homogeneous/pretrain_language_curves.csv` (and MLflow `lang/*`) log, every
`eval_every_updates`: current language, per-language update counts / share % / audio hours / mean
loss, and `mixed_microbatch_violations` / `mixed_accumulation_violations`. Both must be 0 — and
since a violation raises, a finished run with `completed: true` in `pretrain_metrics.json` had none.
`lang/rank_disagreements` is the DDP counterpart.

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
