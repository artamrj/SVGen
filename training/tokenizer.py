from __future__ import annotations

import re
from dataclasses import dataclass, field
import json
from pathlib import Path


TOKEN_PATTERN = re.compile(r"<[^>]+>|[A-Za-z_][A-Za-z0-9_-]*|[-+]?\d*\.?\d+|[^\s]")


@dataclass
class SvgTokenizer:
    vocab: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            self.vocab.setdefault(token, len(self.vocab))

    @classmethod
    def train_from_texts(cls, texts: list[str], vocab_size: int = 4096) -> "SvgTokenizer":
        tokenizer = cls()
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenizer.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:vocab_size]:
            tokenizer.vocab.setdefault(token, len(tokenizer.vocab))
        return tokenizer

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    def encode(self, text: str) -> list[int]:
        unk = self.vocab["<UNK>"]
        return [self.vocab["<BOS>"], *[self.vocab.get(token, unk) for token in self.tokenize(text)], self.vocab["<EOS>"]]

    def decode(self, ids: list[int]) -> str:
        inverse = {idx: token for token, idx in self.vocab.items()}
        tokens = [inverse.get(idx, "<UNK>") for idx in ids]
        filtered = [token for token in tokens if token not in {"<PAD>", "<BOS>", "<EOS>"}]
        return " ".join(filtered).replace("> <", "><")

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "svgen.simple_tokenizer",
            "version": 1,
            "vocab": self.vocab,
            "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SvgTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = payload.get("vocab")
        if not isinstance(vocab, dict):
            raise ValueError(f"Tokenizer artifact is missing vocab: {path}")
        return cls(vocab={str(token): int(idx) for token, idx in vocab.items()})
