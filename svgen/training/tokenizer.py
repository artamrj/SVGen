from __future__ import annotations

import re
from dataclasses import dataclass, field


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


class HuggingFaceTokenizerAdapter:
    """Thin import boundary for future Hugging Face tokenizers integration."""

    def __init__(self, path: str) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install svgen[tokenizers] to load Hugging Face tokenizer artifacts") from exc
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)
