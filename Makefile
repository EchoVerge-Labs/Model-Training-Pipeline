# Single-language ablation runs: keys of `experiments.runs` in params.yaml.
EXPERIMENTS := si-100h ta-100h

.PHONY: setup lint test phase0 snapshot index select pull shard fit-kmeans labels train ckpt bench report all clean help \
        $(foreach e,$(EXPERIMENTS),select-$(e) pull-$(e) shard-$(e) labels-$(e) train-$(e) ckpt-$(e) bench-$(e) heldout-$(e) run-$(e))

# Load local credentials/config for recipes. Values already supplied by the
# shell or command line take precedence over .env.
-include .env
export DAGSHUB_USER DAGSHUB_TOKEN GOOGLE_SHEET_ID GOOGLE_CLIENT_SECRET_JSON
export HF_TOKEN MASTER_ADDR MASTER_PORT NNODES NODE_RANK

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

fit-kmeans:  ## fit k-means on layer-18 features of the base model -> models/kmeans/
	$(PYTHON) -m pipeline.fit_kmeans --config params.yaml

labels:  ## assign cluster-id pseudo-labels to every training cut (needs `make fit-kmeans` first)
	$(PYTHON) -m pipeline.assign_cluster_labels --config params.yaml

train:  ## continued pre-training (needs `make shard` and `make labels` first)
	bash scripts/launch_pretrain.sh

ckpt:  ## select best checkpoint via proxy eval
	$(PYTHON) -m pipeline.select_checkpoint --config params.yaml

bench:  ## run SLSB benchmark on selected checkpoint
	dvc repro benchmark

report:  ## generate results table
	$(PYTHON) -m pipeline.report --config params.yaml

# ── Single-language runs: make <stage>-si-100h / <stage>-ta-100h ─────
# Each run writes only under its own name (see README "Single-language 100h CoPT
# Experiments"). `run-X` = select -> pull -> shard -> labels -> train -> ckpt -> bench through DVC,
# one stage at a time (`-s`), so the always-changed snapshot_catalog stage is not re-run.

define EXPERIMENT_TARGETS
select-$(1):  ## [$(1)] select this run's language slice
	$(PYTHON) -m pipeline.select --config params.yaml --experiment $(1)

labels-$(1):  ## [$(1)] assign cluster labels to this run's shards (joint run's k-means)
	$(PYTHON) -m pipeline.assign_cluster_labels --config params.yaml --experiment $(1)

pull-$(1):  ## [$(1)] copy this run's clips from the Drive mount
	$(PYTHON) -m pipeline.materialise --config params.yaml --experiment $(1)

shard-$(1):  ## [$(1)] shard this run's clips
	$(PYTHON) -m pipeline.shard --config params.yaml --experiment $(1)

train-$(1):  ## [$(1)] launch pre-training for this run (GPU, resumes automatically; needs labels-$(1))
	bash scripts/launch_pretrain.sh --experiment $(1)

ckpt-$(1):  ## [$(1)] select this run's best checkpoint
	$(PYTHON) -m pipeline.select_checkpoint --config params.yaml --experiment $(1)

bench-$(1):  ## [$(1)] run SLSB on this run's selected checkpoint
	dvc repro -s benchmark_single@$(1)

heldout-$(1):  ## [$(1)] held-out vs train loss of saved checkpoints (CPU; overfitting check)
	$(PYTHON) -m pipeline.heldout_loss --experiment $(1)

run-$(1):  ## [$(1)] whole chain: select -> pull -> shard -> labels -> train -> ckpt -> bench
	dvc repro -s select_single@$(1)
	dvc repro -s materialise_single@$(1)
	dvc repro -s shard_single@$(1)
	dvc repro -s assign_cluster_labels_single@$(1)
	dvc repro -s pretrain_single@$(1)
	dvc repro -s select_checkpoint_single@$(1)
	dvc repro -s benchmark_single@$(1)
endef
$(foreach e,$(EXPERIMENTS),$(eval $(call EXPERIMENT_TARGETS,$(e))))

# ── Full pipeline ───────────────────────────────────

all:  ## run entire DVC pipeline
	dvc repro

clean:  ## remove generated artifacts (keeps DVC cache)
	rm -rf data/raw data/shars models/ reports/bench/ reports/*.json reports/*.csv
