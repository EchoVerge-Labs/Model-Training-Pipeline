# Branch ledger: where every change of the model branches went

The model branches (`WavLM-Large`, `HuBERT-Large`, `mHuBERT-147`, `Wav2Vec2-Mono`) are
unified into this branch. Nothing was merged: each change was ported by hand and checked.
The branches themselves are untouched, so every commit below is still readable there.

Fork points: `WavLM-Large`, `HuBERT-Large` and `mHuBERT-147` forked from `916a5ff`;
`Wav2Vec2-Mono` from `92300ca`. `WavLM-Large` and `mHuBERT-147` were built on top of the
HuBERT-Large commits, which is why their first commits are the same.

Names: the wav2vec2 Large model is called `wav2vec2` everywhere in the new layout
(`configs/models/wav2vec2.yaml`, `models/wav2vec2-si-ta-200h`, `reports/wav2vec2/`,
`reports/bench/wav2vec2_*`), not `wav2vec2-mono`.

## How equivalence was checked

| Check | Result |
|---|---|
| XLS-R trainer, tiny seeded model, original code vs refactored (curves, final weights, checkpoints, training-state keys) | byte-identical |
| WavLM trainer, original branch code vs ported (curves, backbone + head weights, checkpoints, optimizer LR groups) | byte-identical |
| HuBERT trainer, same comparison | byte-identical |
| Every `params.yaml` setting each branch changed vs the model's overlay | all reproduced; the only differences are intended (per-model `models/selected_<model>`, wav2vec2 renames) |
| Real-model throughput step on the GPU, all five models | run |
| End-to-end CLI run on real shards, GPU: k-means, labels and 6 training steps for wavlm, hubert, mhubert; 6 steps for xlsr and wav2vec2 | all five finished and wrote complete checkpoints |
| Resume from real checkpoints written by the old HuBERT-Large and mHuBERT-147 branches (`checkpoint-13500`) | resumed at step 13500, optimizer restored, new checkpoint written |
| MLflow: run name, params and metrics for each trainer, local store | run lands in the `ssl-pretraining` experiment as `<tag>-si-ta-<target_batch_seconds>s` (`tests/test_tracking.py`) |

HuBERT-Large and mHuBERT-147 have byte-identical trainer, model wrapper, k-means and label
code (checked with `diff`); WavLM differs from HuBERT only in class names, the `.wavlm` /
`.hubert` attribute and utterance mixing.

## Code and config

| Branch change | Disposition | Now in |
|---|---|---|
| `train.py` (masked-prediction trainer; WavLM, HuBERT, mHuBERT) | ported, on top of `main`'s resume machinery | `models/wavlm/trainer.py`, `models/hubert/trainer.py` |
| `train.py` (Wav2Vec2-Mono: base model, run name) | config only | `configs/models/wav2vec2.yaml` (xlsr trainer) |
| `wavlm_model.py`, `hubert_model.py` | ported; `HEAD_FILE` moved to common | `models/wavlm/model.py`, `models/hubert/model.py` |
| `checkpoints.py` | merged with `main`'s resume code; keeps the head-file completeness rule and still reads the branches' old checkpoints | `common/checkpoints.py` |
| `callbacks.py`: `MaskedAccuracyStallDetector` | ported (`main`'s `CodebookCollapseDetector` kept) | `callbacks.py` |
| `datamodule.py`: labels, per-utterance normalisation, `mix_utterances` | ported (superset of `main`'s) | `datamodule.py` |
| `fit_kmeans.py`, `assign_cluster_labels.py`, `layer_features.py` | ported; now model-agnostic (`AutoModel`) | `labels/` |
| `schema.py`: `ClusterConfig`, `layerdrop`, `head_lr_mult`, `utterance_mix_prob`, `labels_path` | ported; objective-specific fields optional, required per trainer | `schema.py` |
| `select.py`: name-lookup fallback for catalogs without `size` | ported unchanged (it is what produced those later manifests) | `select.py` |
| `snapshot_catalog.py`: relaxed required columns, `expected_headers` | ported unchanged | `snapshot_catalog.py` |
| `select_checkpoint.py`: completeness-aware milestone lookup | merged with `main`'s milestone-only logic; per-model reports dir | `select_checkpoint.py` |
| `scripts/phase0_throughput.py` (WavLM/HuBERT variant) | merged; now picks the model's own step | `scripts/phase0_throughput.py` |
| `scripts/run_bench_slsb.sh` (HuBERT, mHuBERT) | not ported: models are benchmarked from the SLSB-benchmark repository, not from here | - |
| `scripts/run_benchmark.sh` (HuBERT, mHuBERT) | not ported: an older copy of what `main` already has (run prefix and tags are newer on `main`) | unchanged |
| `dvc.yaml`: `fit_kmeans`, `assign_cluster_labels`, per-model pretrain/ckpt/bench | ported as `foreach` stages | `dvc.yaml` |
| `Makefile`: `fit-kmeans`, `labels`, `.env` loading, `PYTHON` | ported (HuBERT's version) plus `MODEL=` | `Makefile` |
| `requirements.txt`, `pyproject.toml`: scikit-learn, joblib | ported | same files |
| `.gitignore`: `/secrets/*` | already in `main` | unchanged |
| `README.md` | rewritten for several models | `README.md` |
| `params.yaml` changes | one overlay per model (see below) | `configs/models/*.yaml` |
| tests: `test_cluster`, `test_datamodule`, `test_smoke_train`, `test_train_e2e`, `test_train_resume`, `test_schema` | ported, parametrised over wavlm and hubert | `tests/test_cluster.py`, `tests/test_datamodule_labels_mixing.py`, `tests/test_masked_trainers.py`, `tests/test_model_configs.py` |

### `params.yaml` changes of each branch

| Setting | WavLM | HuBERT | mHuBERT | Wav2Vec2 | Disposition |
|---|---|---|---|---|---|
| `pretrain.base_model` | `microsoft/wavlm-large` | `facebook/hubert-large-ll60k` | `utter-project/mHuBERT-147` | `facebook/wav2vec2-large-lv60` | overlay |
| `pretrain.output_dir` | `models/wavlm-large-si-ta-200h` | `models/hubert-large-si-ta-200h` | `models/mhubert147-si-ta-200h` | `models/wav2vec2-mono-si-ta-200h` | overlay (wav2vec2: renamed) |
| `cluster.*` | layer 18, k=500 | layer 18, k=500 | layer 9, k=500 | | overlay |
| `cluster.kmeans_output` / `labels_output` | `wavlm-large-L18-k500.joblib`, `wavlm_train_labels.jsonl.gz` | `km_model.joblib`, `train_labels.jsonl.gz` | `mhubert147-L9-k500.joblib`, `mhubert147_train_labels.jsonl.gz` | | overlay (names kept so existing artifacts work) |
| `pretrain.layerdrop`, `head_lr_mult` | 0.1, 1.0 | 0.1, 1.0 | 0.1, 1.0 | | overlay |
| `pretrain.utterance_mix_prob` | 0.2 | | | | overlay |
| `pretrain.labels_path` | per model | per model | per model | | overlay |
| `pretrain.max_gpu_memory_fraction` | 0.8 | 0.8 | 0.8 | | already in `main` |
| Gumbel / diversity settings | removed | removed | removed | | optional in the schema; unused by those trainers |
| `benchmark.upstreams.<model>_adapted` | `models/selected` | `models/selected` | `models/selected` | `models/selected` | overlay, path is now `models/selected_<model>` so models don't overwrite each other |
| `benchmark.upstreams.mhubert147.name` | | | "mHuBERT-147 (vanilla)" | | overlay |
| `benchmark.upstreams.wav2vec2_mono_vanilla` | | | | `facebook/wav2vec2-large-lv60` | overlay, renamed `wav2vec2_vanilla` |
| `benchmark.slsb_data_dir`, `checkpoint_selection.proxy_data_dir` | `/home/anas22/EchoVerge-LABS/SLSB-benchmark/data` | `../SLSB-benchmark/data` | `../SLSB-benchmark/data` | | **not ported**: the WavLM value was one machine's absolute path (commit `175ab8b`); `main`'s `../SLSB-benchmark/data` stays |

## Commits ahead of `origin/main`

### HuBERT-Large (10)

`a010102` HuBERT Large pipeline, `d2b465c` k-means and label assignment, `36cc8af` and
`1897d04` revisions, `b71c909` mask / layerdrop / head-LR match `main`, `591925e` GPU memory
cap, `f0ffd68` read k-means and label audio from Shar, `bf05a37` benchmark configuration
cherry-picked from `main` (already on `main`), `3d08caa` and `8caf5c2` results.
Code: ported (above). Results: imported under `reports/bench/hubert_large_adapted/` and
`reports/bench/hubert_large_adapted_v0.3/`, run reports under `reports/hubert/`.

### WavLM-Large (12)

The seven HuBERT commits above, then `5d05b2a` the WavLM branch, `ac73107` GPU memory cap and
`a879eac` Shar read fix (the same two changes as `591925e` and `f0ffd68`, made again),
`bb9db43` and its revert `88eb04a` (cancel out), `175ab8b` SLSB data path (not ported, see
above), `c498298` results. Results: `reports/bench/wavlm_large_adapted/`,
`reports/bench/wavlm_large_vanilla/`, run reports under `reports/wavlm/`.

### mHuBERT-147 (12)

The HuBERT commits through `bf05a37`, then `e55e5a2` the mHuBERT branch, `1d3fa6e` new
stages, reports, label files and `dvc.lock`, `44362d3` v0.3 results. Results:
`reports/bench/mhubert147_adapted/`, `..._adapted_v0.3/`, `..._base_v0.3/`, run reports under
`reports/mhubert/`.
**Not imported:** the two label files committed in `1d3fa6e` (`data/labels/mhubert147_train_labels.jsonl.gz`,
`data/labels/train_labels.jsonl.gz`, about 37 MB each) are generated data that `make labels`
regenerates, and `dvc.lock` is regenerated by DVC; both stay readable on the branch.

### Wav2Vec2-Mono (3)

`23976bf` the branch (base model, paths, run name), `119d856` and `1dcd5de` results (SLSB v0.3
and v0.4 sid re-run). Results: `reports/bench/wav2vec2_adapted/` and `reports/bench/wav2vec2_vanilla/`
(directory renamed; the raw slsb file names inside are unchanged), run reports under
`reports/wav2vec2/`.

## Training data

All five models trained on the same shards: `main`'s selection (193.6 h realised of the 200 h
target; Sinhala 94.5 h, Tamil 99.2 h; `data/manifests/` and `reports/selection_*`). Checked on the
real files: the cluster labels committed on the HuBERT-Large and mHuBERT-147 branches cover exactly
the 36,482 cut ids of the local `data/shars/`, and those cuts all come from `main`'s manifest.

The WavLM-Large, HuBERT-Large and mHuBERT-147 branches also committed a *different*
`data/manifests/train.jsonl` (184.9 h, from a later `select` run against a catalog without file
sizes). It was never sharded or trained on, so it is not imported. It stays readable on those
branches (commit `d2b465c`).

## Not in this unification

The experiment branches (`experiment/*`: single-language runs, language-homogeneous batching,
held-out loss) are left for later, as agreed; they sit on `main` and the WavLM branch
unchanged.
