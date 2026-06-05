PYTHON ?= .venv/bin/python
PROMPT ?=
OUT ?= reports/output.svg

.PHONY: setup test-python test-rust check cleanup validate-dataset train-tokenizer train-model train export svg

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt

test-python:
	$(PYTHON) -m pytest

test-rust:
	cd runtime && cargo test

check: test-python test-rust

cleanup:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

validate-dataset:
	$(PYTHON) -m training.validate_dataset --input data/train_1000.jsonl --report reports/dataset_report.json

train-tokenizer:
	$(PYTHON) -m training.train_tokenizer --input data/train_1000.jsonl --out tokenizer/tokenizer.json --vocab-size 4096

train-model:
	$(PYTHON) -m training.train --config configs/train_mvp.yaml --data data/train_1000.jsonl --tokenizer tokenizer/tokenizer.json --out checkpoints/svgen-0.2.5

train:
	$(MAKE) validate-dataset
	$(MAKE) train-tokenizer
	$(MAKE) train-model
	$(MAKE) export

export:
	$(PYTHON) -m training.export --checkpoint checkpoints/svgen-0.2.5 --out exports/svgen-0.2.5

svg:
	@test -n "$(PROMPT)" || (echo 'PROMPT is required, for example: make svg PROMPT="running dog icon" OUT=reports/dog.svg' >&2; exit 2)
	$(PYTHON) -m training.generate --model exports/svgen-0.2.5 --prompt "$(PROMPT)" --out "$(OUT)"
