from __future__ import annotations

import argparse
from pathlib import Path

from training.dataset import iter_training_sequences, load_jsonl
from training.tokenizer import SvgTokenizer


def write_vocab(tokenizer: SvgTokenizer, path: Path) -> None:
    vocab_path = path.with_name("vocab.txt")
    inverse = sorted(tokenizer.vocab.items(), key=lambda item: item[1])
    vocab_path.write_text("\n".join(token for token, _idx in inverse) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SVGen MVP tokenizer.")
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--out", required=True, help="Output tokenizer JSON")
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    tokenizer = SvgTokenizer.train_from_texts(list(iter_training_sequences(records)), vocab_size=args.vocab_size)
    out_path = Path(args.out)
    tokenizer.save(out_path)
    write_vocab(tokenizer, out_path)
    print(f"Wrote tokenizer with {len(tokenizer.vocab)} tokens to {out_path}")


if __name__ == "__main__":
    main()
