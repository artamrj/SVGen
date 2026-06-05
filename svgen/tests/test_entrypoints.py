from __future__ import annotations

from training.config import load_config


def test_train_config_loads() -> None:
    config = load_config("configs/train_mvp.yaml")
    assert config["project"]["name"] == "SVGen"


def test_generation_config_loads() -> None:
    config = load_config("configs/generation_default.json")
    assert config["model"] == "SVGen 0.1"
