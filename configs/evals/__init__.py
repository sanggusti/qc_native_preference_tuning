"""Evaluation configuration utilities."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalsConfig:
    """Configuration for the evaluation pipeline."""

    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    num_eval_samples: int = 200
    batch_size: int = 8
    metrics: List[str] = field(
        default_factory=lambda: ["win_rate", "length_controlled_win_rate", "reward_score"]
    )
    languages: List[str] = field(default_factory=lambda: ["en", "id"])

    @classmethod
    def from_dict(cls, cfg: dict) -> "EvalsConfig":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


__all__ = ["EvalsConfig"]
