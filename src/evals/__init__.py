"""Evaluation utilities and base classes."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""

    prompt: str
    response_a: str
    response_b: str
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result for a single evaluated sample."""

    prompt: str
    response_a: str
    response_b: str
    winner: str  # "a", "b", or "tie"
    reasoning: Optional[str] = None
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for preference evaluators."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def evaluate_pair(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single (prompt, response_a, response_b) triplet."""

    def evaluate_batch(self, samples: List[EvalSample]) -> Tuple[List[EvalResult], int]:
        """Evaluate a batch of samples.

        Args:
            samples: List of EvalSample instances.

        Returns:
            Tuple of (list of EvalResult instances, number of failed evaluations).
        """
        results = []
        failures = 0
        for sample in samples:
            try:
                result = self.evaluate_pair(sample)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error evaluating sample: %s", exc)
                failures += 1
        if failures:
            logger.warning(
                "%d / %d samples failed evaluation and were skipped.",
                failures,
                len(samples),
            )
        return results, failures


def compute_win_rate(results: List[EvalResult], model_is_a: bool = True) -> float:
    """Compute win rate from a list of eval results.

    Args:
        results: Evaluated result list.
        model_is_a: If True, count wins for response_a; otherwise for response_b.

    Returns:
        Win rate as a float in [0, 1].
    """
    if not results:
        return 0.0
    target = "a" if model_is_a else "b"
    wins = sum(1 for r in results if r.winner == target)
    ties = sum(1 for r in results if r.winner == "tie")
    return (wins + 0.5 * ties) / len(results)


__all__ = [
    "EvalSample",
    "EvalResult",
    "BaseEvaluator",
    "compute_win_rate",
]
