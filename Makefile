PYTHON ?= .venv/bin/python
PROMPT ?=
OUT ?= reports/output.svg
N ?= 4
ADAPTER ?= models/svgen-0.3-mono
BASE ?= mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16
ICON_PACKS = lucide-static @tabler/icons iconoir heroicons

.PHONY: train-stage1 train-stage2 setup test test-python test-rust check cleanup data-download data train release eval svg fuse \
	legacy-validate-dataset legacy-train-tokenizer legacy-train-model legacy-train legacy-export legacy-svg

# ---- SVGen 0.3 Mono (MLX, Apple Silicon) -------------------------------------

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt

## Download the open-source icon packs used for training (~10 MB).
data-download:
	mkdir -p data/raw/npm data/raw/extracted
	cd data/raw/npm && npm pack $(ICON_PACKS) --silent
	for f in data/raw/npm/*.tgz; do \
		name=$$(basename $$f .tgz); mkdir -p data/raw/extracted/$$name; \
		tar xzf $$f -C data/raw/extracted/$$name; \
	done

## Clean every icon into the canonical format and write train/valid/test splits.
data:
	$(PYTHON) -m svgen.data --raw data/raw/extracted --out data/processed

## LoRA fine-tune on the GPU at low CPU priority, so the laptop stays usable.
## Stage 1 = 2000 steps, stage 2 continues for 3000 more (~2.4 epochs total).
train: train-stage1 train-stage2

train-stage1:
	nice -n 10 $(PYTHON) -m mlx_lm lora -c configs/lora_mono.yaml

train-stage2:
	nice -n 10 $(PYTHON) -m mlx_lm lora -c configs/lora_mono_stage2.yaml

## Copy the trained adapter into models/ in half precision (what gets committed).
release:
	$(PYTHON) -m svgen.release --checkpoint checkpoints/svgen-0.3-mono-stage2 --out $(ADAPTER)

eval:
	$(PYTHON) -m svgen.evaluate --adapter $(ADAPTER) --limit 64 -n $(N)

svg:
	@test -n "$(PROMPT)" || (echo 'PROMPT is required, for example: make svg PROMPT="rocket" OUT=reports/rocket.svg' >&2; exit 2)
	$(PYTHON) -m svgen.generate "$(PROMPT)" --adapter $(ADAPTER) -n $(N) --out "$(OUT)"

## Merge the adapter into a standalone MLX model in exports/.
fuse:
	$(PYTHON) -m mlx_lm fuse --model $(BASE) --adapter-path $(ADAPTER) --save-path exports/svgen-0.3-mono

# ---- checks ------------------------------------------------------------------

test: test-python

test-python:
	$(PYTHON) -m pytest

test-rust:
	cd runtime && cargo test

check: test-python test-rust

cleanup:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

# ---- legacy 0.2 tiny PyTorch model -------------------------------------------

legacy-validate-dataset:
	$(PYTHON) -m training.validate_dataset --input data/train_1000.jsonl --report reports/dataset_report.json

legacy-train-tokenizer:
	$(PYTHON) -m training.train_tokenizer --input data/train_1000.jsonl --out tokenizer/tokenizer.json --vocab-size 4096

legacy-train-model:
	$(PYTHON) -m training.train --config configs/train_mvp.yaml --data data/train_1000.jsonl --tokenizer tokenizer/tokenizer.json --out checkpoints/svgen-0.2.5

legacy-train:
	$(MAKE) legacy-validate-dataset
	$(MAKE) legacy-train-tokenizer
	$(MAKE) legacy-train-model
	$(MAKE) legacy-export

legacy-export:
	$(PYTHON) -m training.export --checkpoint checkpoints/svgen-0.2.5 --out exports/svgen-0.2.5

legacy-svg:
	@test -n "$(PROMPT)" || (echo 'PROMPT is required' >&2; exit 2)
	$(PYTHON) -m training.generate --model exports/svgen-0.2.5 --prompt "$(PROMPT)" --out "$(OUT)"
