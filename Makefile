PYTHON ?= python3

.PHONY: test-python test-rust check cleanup validate-dataset train-tokenizer train export

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

train:
	$(PYTHON) -m training.train --config configs/train_mvp.yaml --data data/train_1000.jsonl --tokenizer tokenizer/tokenizer.json --out checkpoints/svgen-0.2.5

export:
	$(PYTHON) -m training.export --checkpoint checkpoints/svgen-0.2.5 --out exports/svgen-0.2.5
