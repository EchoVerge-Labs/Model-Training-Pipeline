.PHONY: setup lint test phase0 snapshot index select pull shard fit-kmeans labels train ckpt bench report name run-name all clean help

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
# The shared data config the data stages read (with the same VARIANT/TAG as the run, a variant
# that changes the selection -- e.g. smoke-1h -- gets its own data id and folders).
DATA_CONFIG ?= params-data.yaml
ifeq ($(MODEL),mhubert147)
  CONFIG := params-mhubert147.yaml
  HAS_KMEANS := 1
else ifeq ($(MODEL),hubert)
  CONFIG := params-hubert.yaml
  HAS_KMEANS := 1
else ifeq ($(MODEL),wavlm)
  CONFIG := params-wavlm.yaml
  HAS_KMEANS := 1
else ifeq ($(MODEL),xlsr)
  CONFIG := params-xlsr.yaml
else ifeq ($(MODEL),wav2vec2)
  CONFIG := params-wav2vec2.yaml
else
  $(error Unknown MODEL '$(MODEL)'; use mhubert147, hubert, wavlm, xlsr or wav2vec2)
endif

# A run's name is derived from its config (pipeline.common.naming):
#   <model>-<method>-si-ta-<hours>h[-cnn][-p<N>][-homog][-<tag>]   e.g. wavlm-large-copt-si-ta-200h
# VARIANT= overlays the params file (comma-separated: VARIANT=configs/variants/cnn.yaml,...),
# TAG= adds a free-text suffix. Pass the SAME MODEL/VARIANT/TAG to every stage of a run.
VARIANT ?=
TAG ?=
comma := ,
RUN_ARGS := $(foreach v,$(subst $(comma), ,$(VARIANT)),--variant=$(v)) $(if $(TAG),--tag=$(TAG))
# Benchmark/selection refuse an unfinished run; ALLOW_INCOMPLETE=1 overrides (e.g. a smoke test).
INCOMPLETE_ARG := $(if $(ALLOW_INCOMPLETE),--allow-incomplete)
NAME_CMD = $(PYTHON) -m pipeline.common.naming --config $(CONFIG) $(RUN_ARGS)
RUN = $(shell $(NAME_CMD))
# the bench log goes with what is benchmarked: the run, or the model's frozen baseline
BENCH_LOG = logs/$(if $(FROZEN),$(shell $(NAME_CMD) --frozen),$(if $(UPSTREAM),$(or $(NAME),adhoc),$(RUN)))/bench.log

name:  ## print this MODEL/VARIANT/TAG's run name (folders, tmux session, DagsHub run and row)
	@$(NAME_CMD)

run-name: name  ## same as `make name`

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(firstword $(MAKEFILE_LIST)) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup:  ## install package + dev deps
	$(PYTHON) -m pip install -e ".[data,dev]"

lint:  ## ruff check + format check
	$(PYTHON) -m ruff check src tests scripts && $(PYTHON) -m ruff format --check src tests scripts

test:  ## run unit tests
	$(PYTHON) -m pytest -q tests/

# ── Individual pipeline stages ──────────────────────

phase0:  ## measure GPU throughput (run first!); PHASE0_ARGS=--no-freeze-cnn to measure a trainable CNN
	$(PYTHON) scripts/phase0_throughput.py --model $(MODEL) --steps 100 $(PHASE0_ARGS)

snapshot:  ## pull catalog from Google Sheet
	@if [ -z "$${GOOGLE_SHEET_ID:-}" ]; then \
		echo "ERROR: set GOOGLE_SHEET_ID before running make snapshot" >&2; \
		exit 1; \
	fi
	$(PYTHON) -m pipeline.data.snapshot_catalog \
	  --sheet-id $${GOOGLE_SHEET_ID} --out data/catalog

index:  ## index every wav on the Drive mount (relpath + size)
	$(PYTHON) -m pipeline.data.drive_index --config $(DATA_CONFIG)

select:  ## select the training subset (needs `make index` first)
	$(PYTHON) -m pipeline.data.select --config $(DATA_CONFIG) $(RUN_ARGS)

pull:  ## copy the selected clips from the Drive mount to data/raw/
	$(PYTHON) -m pipeline.data.materialise --config $(DATA_CONFIG) $(RUN_ARGS)

shard:  ## shard WAVs into Lhotse Shar tarballs
	$(PYTHON) -m pipeline.data.shard --config $(DATA_CONFIG) $(RUN_ARGS)

fit-kmeans:  ## fit k-means on layer features of MODEL's base model -> models/kmeans/ (FORCE=1 to refit)
ifndef HAS_KMEANS
	$(error MODEL=$(MODEL) uses a contrastive objective and has no k-means step)
endif
	$(PYTHON) -m pipeline.labels.fit_kmeans --config $(CONFIG) $(RUN_ARGS) $(if $(FORCE),--force)

labels:  ## assign cluster-id pseudo-labels to every training cut (needs `make fit-kmeans` first)
ifndef HAS_KMEANS
	$(error MODEL=$(MODEL) uses a contrastive objective and has no cluster labels)
endif
	$(PYTHON) -m pipeline.labels.assign_cluster_labels --config $(CONFIG) $(RUN_ARGS)

train:  ## continued pre-training of MODEL -> models/<run>/, reports/<run>/ (HuBERT family: `make labels` first)
	@mkdir -p logs/$(RUN)
	bash -o pipefail -c 'bash scripts/launch_pretrain.sh --config $(CONFIG) $(RUN_ARGS) 2>&1 | tee -a logs/$(RUN)/train.log'

ckpt:  ## select the best checkpoint via proxy eval -> models/<run>/selected/
	@mkdir -p logs/$(RUN)
	bash -o pipefail -c '$(PYTHON) -m pipeline.eval.select_checkpoint --config $(CONFIG) $(RUN_ARGS) $(INCOMPLETE_ARG) 2>&1 | tee -a logs/$(RUN)/ckpt.log'

bench:  ## SLSB benchmark -> reports/<run>/bench/ + one leaderboard row; FROZEN=1 for <model>-frozen, UPSTREAM= NAME= for any other
	@mkdir -p $(dir $(BENCH_LOG))
	bash -o pipefail -c '$(PYTHON) -m pipeline.eval.benchmark --config $(CONFIG) $(RUN_ARGS) $(INCOMPLETE_ARG) $(if $(FROZEN),--frozen) $(if $(UPSTREAM),--upstream=$(UPSTREAM)) $(if $(NAME),--name=$(NAME)) $(if $(OUT),--out=$(OUT)) $(if $(NOTE),--note="$(NOTE)") $(if $(REPLACE),--replace) $(if $(CARRY_OVER),--carry-over) $(if $(NO_LOG),--no-log) $(if $(DRY_RUN),--dry-run) 2>&1 | tee -a $(BENCH_LOG)'

report:  ## results table of the run vs its frozen baselines -> reports/<run>/results.md
	$(PYTHON) -m pipeline.eval.report --config $(CONFIG) $(RUN_ARGS)

# ── Full pipeline ───────────────────────────────────

all:  ## MODEL's whole pipeline (needs data/ selected+materialised): shard, [k-means, labels,] train, ckpt, bench, report
	$(MAKE) shard VARIANT=$(VARIANT) TAG=$(TAG)
ifdef HAS_KMEANS
	$(MAKE) fit-kmeans labels MODEL=$(MODEL) VARIANT=$(VARIANT) TAG=$(TAG)
endif
	$(MAKE) train ckpt bench report MODEL=$(MODEL) VARIANT=$(VARIANT) TAG=$(TAG)

clean:  ## remove local data and models: data/raw, data/shars, models/ (keeps reports/ and the DVC cache)
	rm -rf data/raw data/shars models/
