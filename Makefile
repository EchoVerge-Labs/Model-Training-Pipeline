.PHONY: setup lint test phase0 snapshot index select pull shard fit-kmeans labels train ckpt bench report all clean help

# Load local credentials/config for recipes. Values already supplied by the
# shell win over .env (make's own include would otherwise overwrite them, e.g.
# NODE_RANK=1 make train would silently run as rank 0): remember what the
# environment had, include .env, then put those values back.
ENV_VARS := DAGSHUB_USER DAGSHUB_TOKEN GOOGLE_SHEET_ID GOOGLE_CLIENT_SECRET_JSON HF_TOKEN MASTER_ADDR MASTER_PORT NNODES NODE_RANK
$(foreach v,$(ENV_VARS),$(if $(filter environment,$(origin $(v))),$(eval _SHELL_$(v) := $($(v)))))
-include .env
$(foreach v,$(ENV_VARS),$(if $(filter environment,$(origin _SHELL_$(v))),,$(if $(filter file,$(origin _SHELL_$(v))),$(eval $(v) := $(_SHELL_$(v))))))
export $(ENV_VARS)

PYTHON ?= $(if $(wildcard $(CURDIR)/.venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

# Model-specific targets (fit-kmeans, labels, train, ckpt, bench, report) take
# MODEL=mhubert147|hubert|wavlm|xlsr|wav2vec2, e.g. `make train MODEL=wavlm`.
MODEL ?= mhubert147
ifeq ($(MODEL),mhubert147)
  CONFIG := params.yaml
  DVC_FILE := dvc.yaml
  HAS_KMEANS := 1
else ifeq ($(MODEL),hubert)
  CONFIG := params-hubert.yaml
  DVC_FILE := dvc-hubert.yaml
  HAS_KMEANS := 1
else ifeq ($(MODEL),wavlm)
  CONFIG := params-wavlm.yaml
  DVC_FILE := dvc-wavlm.yaml
  HAS_KMEANS := 1
else ifeq ($(MODEL),xlsr)
  CONFIG := params-xlsr.yaml
  DVC_FILE := dvc-xlsr.yaml
else ifeq ($(MODEL),wav2vec2)
  CONFIG := params-wav2vec2.yaml
  DVC_FILE := dvc-wav2vec2.yaml
else
  $(error Unknown MODEL '$(MODEL)'; use mhubert147, hubert, wavlm, xlsr or wav2vec2)
endif

# EXPERIMENT names the run: outputs go to <model>-<EXPERIMENT> (models/, reports/*-<run>.*,
# models/selected-<run>, reports/bench/<run>, MLflow run, logs/<run>.log), so runs never
# share a path. Default: the `experiment:` in the params file (si-ta-200h).
# e.g. `make train MODEL=wavlm EXPERIMENT=si-100h`, then `make ckpt|bench|report` the same.
EXPERIMENT ?=
EXP_ARG := $(if $(EXPERIMENT),--experiment=$(EXPERIMENT))
RUN = $(shell $(PYTHON) -c "from pipeline.schema import load_raw_params as l; print(l('$(CONFIG)', $(if $(EXPERIMENT),'$(EXPERIMENT)',None))['run_name'])")

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

phase0:  ## measure GPU throughput (run first!); PHASE0_ARGS=--no-freeze-cnn to measure a trainable CNN
	$(PYTHON) scripts/phase0_throughput.py --model $(MODEL) --steps 100 $(PHASE0_ARGS)

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
	$(PYTHON) -m pipeline.fit_kmeans --config $(CONFIG) $(EXP_ARG)

labels:  ## assign cluster-id pseudo-labels to every training cut (needs `make fit-kmeans` first)
ifndef HAS_KMEANS
	$(error MODEL=$(MODEL) uses a contrastive objective and has no cluster labels)
endif
	$(PYTHON) -m pipeline.assign_cluster_labels --config $(CONFIG) $(EXP_ARG)

train:  ## continued pre-training of MODEL (HuBERT-family needs `make shard` and `make labels` first)
	@mkdir -p logs
	bash -o pipefail -c 'bash scripts/launch_pretrain.sh --model $(MODEL) $(EXP_ARG) 2>&1 | tee -a logs/$(RUN).log'

ckpt:  ## select best checkpoint via proxy eval
	$(PYTHON) -m pipeline.select_checkpoint --config $(CONFIG) $(EXP_ARG)

bench:  ## SLSB benchmark of the run's selected checkpoint -> reports/bench/<run>/; UPSTREAM=<dir|HF id> for another
	$(PYTHON) -m pipeline.benchmark --config $(CONFIG) $(EXP_ARG) $(if $(UPSTREAM),--upstream=$(UPSTREAM))

report:  ## generate results table
	$(PYTHON) -m pipeline.report --config $(CONFIG) $(EXP_ARG)

# ── Full pipeline ───────────────────────────────────

all:  ## run entire DVC pipeline
	dvc repro

clean:  ## remove generated artifacts (keeps DVC cache)
	rm -rf data/raw data/shars models/ reports/bench/ reports/*.json reports/*.csv
