.PHONY: setup lint test phase0 snapshot index select pull shard fit-kmeans labels train ckpt bench report all clean help

# Load local credentials/config for recipes. Values already supplied by the
# shell or command line take precedence over .env.
-include .env
export DAGSHUB_USER DAGSHUB_TOKEN GOOGLE_SHEET_ID GOOGLE_CLIENT_SECRET_JSON
export HF_TOKEN MASTER_ADDR MASTER_PORT NNODES NODE_RANK

PYTHON ?= $(if $(wildcard $(CURDIR)/.venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

# Model-specific targets (fit-kmeans, labels, train, ckpt, bench, report) take
# MODEL=mhubert147|hubert|wavlm|xlsr|wav2vec2, e.g. `make train MODEL=wavlm`.
MODEL ?= mhubert147
ifeq ($(MODEL),mhubert147)
  TRAIN_SCRIPT := src/pipeline/train.py
  CONFIG := params.yaml
  DVC_FILE := dvc.yaml
  SELECT_MODULE := pipeline.select_checkpoint
  HAS_KMEANS := 1
else ifeq ($(MODEL),hubert)
  TRAIN_SCRIPT := src/pipeline/train_hubert.py
  CONFIG := params-hubert.yaml
  DVC_FILE := dvc-hubert.yaml
  SELECT_MODULE := pipeline.select_checkpoint
  HAS_KMEANS := 1
else ifeq ($(MODEL),wavlm)
  TRAIN_SCRIPT := src/pipeline/train_wavlm.py
  CONFIG := params-wavlm.yaml
  DVC_FILE := dvc-wavlm.yaml
  SELECT_MODULE := pipeline.select_checkpoint_wavlm
  HAS_KMEANS := 1
else ifeq ($(MODEL),xlsr)
  TRAIN_SCRIPT := src/pipeline/train_xlsr.py
  CONFIG := params-xlsr.yaml
  DVC_FILE := dvc-xlsr.yaml
  SELECT_MODULE := pipeline.select_checkpoint_xlsr
else ifeq ($(MODEL),wav2vec2)
  TRAIN_SCRIPT := src/pipeline/train_wav2vec2.py
  CONFIG := params-wav2vec2.yaml
  DVC_FILE := dvc-wav2vec2.yaml
  SELECT_MODULE := pipeline.select_checkpoint_xlsr
else
  $(error Unknown MODEL '$(MODEL)'; use mhubert147, hubert, wavlm, xlsr or wav2vec2)
endif
SUFFIX := -$(MODEL)

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup:  ## install package + dev deps
	$(PYTHON) -m pip install -e ".[data,dev]"

lint:  ## ruff check + format check
	ruff check src tests && ruff format --check src tests

test:  ## run unit tests
	pytest -q tests/

# ── Individual pipeline stages ──────────────────────

phase0:  ## measure GPU throughput (run first!)
	$(PYTHON) scripts/phase0_throughput.py --steps 100

snapshot:  ## pull catalog from Google Sheet
	@if [ -z "$${GOOGLE_SHEET_ID:-}" ]; then \
		echo "ERROR: set GOOGLE_SHEET_ID before running make snapshot" >&2; \
		exit 1; \
	fi
	$(PYTHON) -m pipeline.snapshot_catalog \
	  --sheet-id $${GOOGLE_SHEET_ID} --out data/catalog

index:  ## index every wav on the Drive mount (relpath + size)
	$(PYTHON) -m pipeline.drive_index --config params.yaml

select:  ## select the training subset (needs `make index` first)
	$(PYTHON) -m pipeline.select --config params.yaml

pull:  ## copy the selected clips from the Drive mount to data/raw/
	$(PYTHON) -m pipeline.materialise --config params.yaml

shard:  ## shard WAVs into Lhotse Shar tarballs
	$(PYTHON) -m pipeline.shard --config params.yaml

fit-kmeans:  ## fit k-means on layer features of MODEL's base model -> models/kmeans/
ifndef HAS_KMEANS
	$(error MODEL=$(MODEL) uses a contrastive objective and has no k-means step)
endif
	$(PYTHON) -m pipeline.fit_kmeans --config $(CONFIG) --report-suffix $(SUFFIX)

labels:  ## assign cluster-id pseudo-labels to every training cut (needs `make fit-kmeans` first)
ifndef HAS_KMEANS
	$(error MODEL=$(MODEL) uses a contrastive objective and has no cluster labels)
endif
	$(PYTHON) -m pipeline.assign_cluster_labels --config $(CONFIG)

train:  ## continued pre-training of MODEL (HuBERT-family needs `make shard` and `make labels` first)
	SCRIPT=$(TRAIN_SCRIPT) CONFIG=$(CONFIG) bash scripts/launch_pretrain.sh

ckpt:  ## select best checkpoint via proxy eval
	$(PYTHON) -m $(SELECT_MODULE) --config $(CONFIG) --report-suffix $(SUFFIX)

bench:  ## run SLSB benchmark on selected checkpoint
	dvc repro $(DVC_FILE):benchmark

report:  ## generate results table
	$(PYTHON) -m pipeline.report --config $(CONFIG) --report-suffix $(SUFFIX)

# ── Full pipeline ───────────────────────────────────

all:  ## run entire DVC pipeline
	dvc repro

clean:  ## remove generated artifacts (keeps DVC cache)
	rm -rf data/raw data/shars models/ reports/bench/ reports/*.json reports/*.csv
