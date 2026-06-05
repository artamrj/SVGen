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


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to train the SVGen tiny PyTorch model") from exc
    return torch, nn


def build_tiny_language_model(vocab_size: int, context: int, hidden: int, layers: int, heads: int):
    torch, nn = require_torch()

    class TinySvgLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, hidden)
            self.position_embedding = nn.Embedding(context, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                batch_first=True,
                activation="gelu",
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden)
            self.output = nn.Linear(hidden, vocab_size)

        def forward(self, tokens):
            batch, seq_len = tokens.shape
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)
            hidden_state = self.token_embedding(tokens) + self.position_embedding(positions)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
            hidden_state = self.blocks(hidden_state, mask=mask)
            return self.output(self.norm(hidden_state))

    return TinySvgLanguageModel()
