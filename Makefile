.PHONY: setup lint test phase0 snapshot index select pull shard fit-kmeans labels train ckpt bench bench-frozen report all clean clean-data help

# Load local credentials/config for recipes. Values already supplied by the
# shell or command line take precedence over .env.
-include .env
export DAGSHUB_USER DAGSHUB_TOKEN GOOGLE_SHEET_ID GOOGLE_CLIENT_SECRET_JSON
export HF_TOKEN MASTER_ADDR MASTER_PORT NNODES NODE_RANK

# Which model the model-specific targets (phase0 fit-kmeans labels train ckpt bench) run for:
# xlsr, wav2vec2, wavlm, hubert or mhubert -- its configs/models/<model>.yaml is merged over
# params.yaml. Unset = params.yaml as it is (the XLS-R run).  e.g.  make train MODEL=wavlm
MODEL ?=
MODEL_FLAG := $(if $(MODEL),--model $(MODEL))
export MODEL

PYTHON ?= $(if $(wildcard $(CURDIR)/.venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

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
	$(PYTHON) scripts/phase0_throughput.py --steps 100 $(MODEL_FLAG)

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

fit-kmeans:  ## fit k-means on layer features of the base model (hubert/wavlm/mhubert; needs MODEL=)
	$(PYTHON) -m pipeline.labels.fit_kmeans --config params.yaml $(MODEL_FLAG)

labels:  ## assign cluster-id pseudo-labels to every training cut (needs `make fit-kmeans` first)
	$(PYTHON) -m pipeline.labels.assign_cluster_labels --config params.yaml $(MODEL_FLAG)

train:  ## continued pre-training of MODEL (the label models need `make shard` and `make labels` first)
	bash scripts/launch_pretrain.sh

ckpt:  ## select best checkpoint via proxy eval
	$(PYTHON) -m pipeline.select_checkpoint --config params.yaml $(MODEL_FLAG)

bench:  ## benchmark the selected checkpoint against SLSB and log it (MODEL= picks the model)
	$(PYTHON) -m pipeline.benchmark --config params.yaml $(MODEL_FLAG)

bench-frozen:  ## benchmark MODEL's pre-trained base: its baseline row in the leaderboard
	$(PYTHON) -m pipeline.benchmark --config params.yaml $(MODEL_FLAG) --frozen

report:  ## generate results table (every model's upstreams)
	$(PYTHON) -m pipeline.report --config params.yaml --all-models

# ── Full pipeline ───────────────────────────────────

all:  ## run entire DVC pipeline
	dvc repro

clean:  ## remove caches (never touches data, models or results)
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

clean-data:  ## remove the regenerable data (raw clips, shards); never models or reports
	rm -rf data/raw data/shars
