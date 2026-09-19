.PHONY: setup lint test phase0 snapshot select pull shard train ckpt bench report all clean help

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

select:  ## select 200h training subset
	python -m pipeline.select --config params.yaml

pull:  ## materialise selected files from Drive
	bash scripts/rclone_pull.sh

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

# ── Full pipeline ───────────────────────────────────

all:  ## run entire DVC pipeline
	dvc repro

clean:  ## remove generated artifacts (keeps DVC cache)
	rm -rf data/raw data/shars models/ reports/bench/ reports/*.json reports/*.csv
