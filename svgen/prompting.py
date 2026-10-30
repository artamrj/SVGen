"""Chat format shared by training and inference, so they can never drift apart."""

from __future__ import annotations

SYSTEM_PROMPT = "Draw a 24x24 black-and-white line icon as SVG shapes."


def messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
    ]


def chat_record(prompt: str, body: str) -> dict:
    return {"messages": [*messages(prompt), {"role": "assistant", "content": body}]}
