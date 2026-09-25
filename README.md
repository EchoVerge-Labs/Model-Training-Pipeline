# Model Training Pipeline — Continued Pre-training of HuBERT Large on Sinhala/Tamil

DVC-orchestrated pipeline for continued self-supervised HuBERT pre-training, automatic
checkpoint selection, and benchmarking. Produces a Sinhala/Tamil-specialised speech
encoder from `facebook/hubert-large-ll60k`.

## Pipeline DAG

```
snapshot_catalog → index_drive → select → materialise → shard → fit_kmeans → assign_cluster_labels → pretrain → select_checkpoint → benchmark → report
```

`pretrain` continues pre-training the already-pretrained `hubert-large-ll60k`
with HuBERT's real objective: masked prediction of k-means cluster ids, not
wav2vec2-style contrastive learning. The model architecture and objective are
unchanged; this is not distillation. The cluster ids are made once, offline:
`fit_kmeans` runs the *original* checkpoint over ~40 h of the training set and
clusters the output of transformer layer 18 into 500 clusters;
`assign_cluster_labels` assigns every training cut's frames to the nearest
centroid (`data/labels/`). Waveforms are normalised (`do_normalize`, read from
the base model) identically for labelling and training. `transformers` has no
`HubertForPreTraining`, and the checkpoint carries no prediction head, so
`pipeline.hubert_model` adds a fresh linear head over the cluster ids.

Branches: `main` continues wav2vec2 XLS-R, `HuBERT-Large` (this branch)
continues HuBERT Large, and WavLM Large lives on its own branch.

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
make fit-kmeans # fit k-means on layer-18 features of the base model -> models/kmeans/
make labels     # assign cluster-id pseudo-labels to every training cut -> data/labels/
make train      # continued pre-training -> models/hubert-large-si-ta-200h/
make ckpt       # proxy-eval milestone checkpoints, copy the best -> models/selected/
make bench      # slsb run on the selected checkpoint -> reports/bench/hubert_large_adapted/
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
`shard`, `cluster`, and `pretrain` sections with Pydantic before any stage
runs (`make
test` / CI run this validation too, so a bad config is caught before it
wastes GPU hours).

## Known gaps

- **Single label round.** Cluster ids come from layer 18 of the original
  checkpoint. Re-clustering on an adapted checkpoint's hidden states (a second
  HuBERT iteration) isn't implemented.
- **Resume is per node.** Re-running `make train` resumes from the newest
  complete `checkpoint-<step>/` (backbone, head, optimizer, step; saved
  atomically). The data loader restarts its shard order rather than resuming
  mid-epoch. For 2-node runs every node must have that checkpoint locally
  (training aborts with a clear message if the nodes disagree).
- **`scripts/run_benchmark.sh`** exists to bridge slsb's real output filename
  (`results_<upstream-with-/-as-__>.json`) to the fixed `metrics.json` path
  `dvc.yaml` and `report.py` expect — see that script's header comment.
