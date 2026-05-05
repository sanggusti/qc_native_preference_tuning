"""Data generator configuration utilities."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataGeneratorConfig:
    """Configuration for the preference data generator."""

    generator_model: str = "gpt-4o-mini"
    num_samples: int = 1000
    temperature: float = 0.8
    max_tokens: int = 1024
    topics: List[str] = field(
        default_factory=lambda: [
            "general_knowledge",
            "coding",
            "math",
            "writing",
            "reasoning",
        ]
    )
    languages: List[str] = field(default_factory=lambda: ["en", "id"])

    @classmethod
    def from_dict(cls, cfg: dict) -> "DataGeneratorConfig":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


__all__ = ["DataGeneratorConfig"]
