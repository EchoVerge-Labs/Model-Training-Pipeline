# Model Training Pipeline — Continued Pre-training of XLS-R 300m on Sinhala/Tamil

DVC-orchestrated pipeline for continued self-supervised wav2vec2 pre-training, automatic
checkpoint selection, and benchmarking. Produces a Sinhala/Tamil-specialised speech
encoder from `facebook/wav2vec2-xls-r-300m`.

## Pipeline DAG

```
snapshot_catalog → select → materialise → shard → pretrain → select_checkpoint → benchmark → report
```

Each stage is a `dvc.yaml` target; `dvc repro` runs whatever is stale given
`params.yaml` and each stage's declared deps. The `benchmark` stage depends
on `pretrain`'s and `select_checkpoint`'s outputs, so `dvc repro` chains
training straight into evaluation automatically.

## Prerequisites

- NVIDIA DGX Spark (GB10, ARM64) with the NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.11-py3`)
- Google Drive with pre-processed audio, mounted via `rclone` (remote name `gdrive`, folder `Pre Processed Data/{Sinhala,Tamil}`)
- A DagsHub account (DVC remote + MLflow tracking) — see `.env.example`
- A Google service-account JSON with read access to the "Sinhala X Tamil Voice Dataset" sheet
- [SLSB-benchmark](https://github.com/EchoVerge-Labs/SLSB-benchmark) v0.1.0 installed (`pip install -e ".[benchmark]"`, or already in the Docker image)

## Quick start

```bash
# 1. Install (inside the NGC container, or a matching ARM64 CUDA environment)
pip install -e ".[data,benchmark,dev]"
cp .env.example .env   # fill in DAGSHUB_TOKEN, GOOGLE_SHEET_ID, GOOGLE_SERVICE_ACCOUNT_JSON, HF_TOKEN

# 2. Measure real throughput BEFORE scoping max_updates / timelines.
#    This loads the real 300M model and does real GPU compute -- treat it
#    like any other GPU job (see "Running training" below).
make phase0

# 3. Run the pipeline stage by stage (or `make all` / `dvc repro` for the whole DAG)
make snapshot   # freeze the Google Sheet catalog to data/catalog/catalog_latest.csv
make select     # deterministic 200h train/holdout selection -> data/manifests/
make pull       # rclone the selected files from Drive -> data/raw/
make shard      # pack into Lhotse Shar tarballs -> data/shars/
make train      # continued pre-training -> models/xlsr300m-si-ta-200h/
make ckpt       # proxy-eval milestone checkpoints, copy the best -> models/selected/
make bench      # slsb run on the selected checkpoint -> reports/bench/xlsr300m_adapted/
make report     # aggregate into reports/results.md + results_table.csv
```

### Running training

`make train` / `make phase0` launch real, hours-to-days-long GPU jobs on
shared hardware. Run them in `tmux` (not `nohup`), and don't launch either
without checking first if you're running this pipeline as part of a larger
session. For two-node training, run `scripts/nccl_bandwidth_test.sh` on both
nodes first to confirm the ConnectX-7 link is actually being used (see
`configs/nccl-env.sh` — `NCCL_SOCKET_IFNAME` in particular needs to match
your actual interface name).

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

## Configuration

`params.yaml` is the single source of truth; every stage reads its own
section from it, and `src/pipeline/schema.py` validates the `select`,
`shard`, and `pretrain` sections with Pydantic before any stage runs (`make
test` / CI run this validation too, so a bad config is caught before it
wastes GPU hours).

## Known gaps

- **Checkpoint resumption** isn't implemented — `train.py` saves
  `training_state.pt` (optimizer state, step, Gumbel temperature) alongside
  each checkpoint, but there's no `--resume-from` flag yet to load it back.
  If a run crashes, restart manually by pointing `pretrain.base_model` at the
  last checkpoint dir (note: this restarts the LR/Gumbel schedule from step 0
  relative to the new run, which is not the same as a true resume).
- **`scripts/run_benchmark.sh`** exists to bridge slsb's real output filename
  (`results_<upstream-with-/-as-__>.json`) to the fixed `metrics.json` path
  `dvc.yaml` and `report.py` expect — see that script's header comment.
