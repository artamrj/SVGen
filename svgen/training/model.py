from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyModelConfig:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    max_position_embeddings: int


class PlaceholderSvgModel:
    def __init__(self, config: TinyModelConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> str:
        _ = prompt
        return '<svg viewBox="0 0 24 24"><path d="M4 12 L12 4 L20 12 L20 20 L4 20 Z"/></svg>'
