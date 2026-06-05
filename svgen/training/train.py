from __future__ import annotations

import argparse

from training.config import load_config
from training.dataset import iter_training_sequences, load_jsonl
from training.tokenizer import SvgTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SVGen MVP training scaffold.")
    parser.add_argument("--config", default="configs/train_mvp.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = config["data"]["train_path"]
    records = load_jsonl(train_path)
    sequences = list(iter_training_sequences(records))
    tokenizer = SvgTokenizer.train_from_texts(sequences, vocab_size=config["tokenizer"]["vocab_size"])

    print("SVGen training scaffold")
    print(f"records={len(records)}")
    print(f"vocab_size={len(tokenizer.vocab)}")
    print("status=placeholder")


if __name__ == "__main__":
    main()
