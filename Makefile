.PHONY: setup lint test phase0 snapshot index select pull shard train ckpt bench report all clean help \
        check-homogeneous train-homogeneous ckpt-homogeneous bench-homogeneous run-homogeneous

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup:  ## install package + dev deps
	pip install -e ".[data,dev]"

lint:  ## ruff check + format check
	ruff check src tests && ruff format --check src tests

test:  ## run unit tests
	pytest -q tests/

# ── Individual pipeline stages ──────────────────────

phase0:  ## measure GPU throughput (run first!)
	python scripts/phase0_throughput.py --steps 100

snapshot:  ## pull catalog from Google Sheet
	python -m pipeline.snapshot_catalog \
	  --sheet-id $${GOOGLE_SHEET_ID} --out data/catalog

index:  ## index every wav on the Drive mount (relpath + size)
	python -m pipeline.drive_index --config params.yaml

select:  ## select the training subset (needs `make index` first)
	python -m pipeline.select --config params.yaml

pull:  ## copy the selected clips from the Drive mount to data/raw/
	python -m pipeline.materialise --config params.yaml

shard:  ## shard WAVs into Lhotse Shar tarballs
	python -m pipeline.shard --config params.yaml

train:  ## launch pre-training
	bash scripts/launch_pretrain.sh

ckpt:  ## select best checkpoint via proxy eval
	python -m pipeline.select_checkpoint --config params.yaml

bench:  ## run SLSB benchmark on selected checkpoint
	dvc repro benchmark

report:  ## generate results table
	python -m pipeline.report --config params.yaml

# ── Joint 200h with language-homogeneous updates ────
# Same data and settings as `make train`/`ckpt`/`bench`; only batching differs (see README
# "Joint 200h CoPT with Language-Homogeneous Updates"). Overlay = configs/language_homogeneous.yaml.
HOMOGENEOUS_OVERLAY = configs/language_homogeneous.yaml
HOMOGENEOUS_OUT = $(shell python -m pipeline.variant get benchmark.out_dir --overlay $(HOMOGENEOUS_OVERLAY))

check-homogeneous:  ## [homogeneous] verify the clips are exactly the original 200h run's
	python -m pipeline.language_batching --overlay $(HOMOGENEOUS_OVERLAY)

train-homogeneous: check-homogeneous  ## [homogeneous] launch pre-training (GPU, resumes automatically)
	bash scripts/launch_pretrain.sh --overlay $(HOMOGENEOUS_OVERLAY)

ckpt-homogeneous:  ## [homogeneous] select the best checkpoint (same method as `make ckpt`)
	python -m pipeline.select_checkpoint --config params.yaml --overlay $(HOMOGENEOUS_OVERLAY)

bench-homogeneous:  ## [homogeneous] SLSB on the selected checkpoint (same settings as `make bench`)
	SELECTION_REPORT=$$(python -m pipeline.variant get checkpoint_selection.report_path --overlay $(HOMOGENEOUS_OVERLAY)) \
	bash scripts/run_benchmark.sh \
	  $$(python -m pipeline.variant get checkpoint_selection.output_dir --overlay $(HOMOGENEOUS_OVERLAY)) \
	  $$(python -m pipeline.variant get benchmark.tasks --overlay $(HOMOGENEOUS_OVERLAY)) \
	  $$(python -m pipeline.variant get benchmark.seeds --overlay $(HOMOGENEOUS_OVERLAY)) \
	  $(HOMOGENEOUS_OUT) \
	  $$(python -m pipeline.variant get benchmark.mlflow_uri --overlay $(HOMOGENEOUS_OVERLAY)) \
	  $$(python -m pipeline.variant get benchmark.slsb_data_dir --overlay $(HOMOGENEOUS_OVERLAY)) \
	  $$(python -m pipeline.variant get benchmark.mlflow_run_prefix --overlay $(HOMOGENEOUS_OVERLAY)) \
	  "$$(python -m pipeline.variant get benchmark.mlflow_run_tags --overlay $(HOMOGENEOUS_OVERLAY))"

run-homogeneous: train-homogeneous ckpt-homogeneous bench-homogeneous  ## [homogeneous] check -> train -> ckpt -> bench

# ── Full pipeline ───────────────────────────────────

all:  ## run entire DVC pipeline
	dvc repro

clean:  ## remove generated artifacts (keeps DVC cache)
	rm -rf data/raw data/shars models/ reports/bench/ reports/*.json reports/*.csv
