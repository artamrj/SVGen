from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.config import load_config
from training.dataset import iter_training_sequences, load_jsonl
from training.model import build_tiny_language_model, require_torch
from training.tokenizer import SvgTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SVGen 0.2.6 tiny model.")
    parser.add_argument("--config", default="configs/train_mvp.yaml")
    parser.add_argument("--data")
    parser.add_argument("--tokenizer")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(args.config)
    train_path = args.data or config["data"]["train_path"]
    tokenizer_path = args.tokenizer or config["tokenizer"]["path"]
    out_dir = Path(args.out or config["training"]["checkpoint_dir"])
    records = load_jsonl(train_path)
    sequences = list(iter_training_sequences(records))
    tokenizer = SvgTokenizer.load(tokenizer_path) if Path(tokenizer_path).exists() else SvgTokenizer.train_from_texts(sequences, vocab_size=config["tokenizer"]["vocab_size"])
    torch, _nn = require_torch()

    model_config = config["model"]
    context = int(model_config["context"])
    model = build_tiny_language_model(
        vocab_size=len(tokenizer.vocab),
        context=context,
        hidden=int(model_config["hidden"]),
        layers=int(model_config["layers"]),
        heads=int(model_config["heads"]),
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    encoded = [tokenizer.encode(sequence)[:context] for sequence in sequences]
    batch_size = int(config["training"]["batch_size"])
    losses: list[float] = []

    for start in range(0, len(encoded), batch_size):
        batch_ids = encoded[start : start + batch_size]
        max_len = min(context, max(len(item) for item in batch_ids))
        if max_len < 2:
            continue
        batch = torch.full((len(batch_ids), max_len), tokenizer.vocab["<PAD>"], dtype=torch.long)
        for row, ids in enumerate(batch_ids):
            clipped = ids[:max_len]
            batch[row, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=tokenizer.vocab["<PAD>"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    avg_loss = sum(losses) / len(losses) if losses else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": model_config, "vocab_size": len(tokenizer.vocab)}, out_dir / "checkpoint.pt")
    (out_dir / "training_report.json").write_text(
        json.dumps({"model": "SVGen 0.2.6 Easy Python Inference", "records": len(records), "loss": avg_loss, "batches": len(losses)}, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SVGen 0.2.6 tiny training")
    print(f"records={len(records)}")
    print(f"vocab_size={len(tokenizer.vocab)}")
    print(f"loss={avg_loss:.4f}")
    print(f"checkpoint={out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
