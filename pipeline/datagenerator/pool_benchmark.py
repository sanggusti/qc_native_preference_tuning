"""Pooled training set: sanggusti/{benchmark}-all (issue #53).

Concatenates the per-language train splits at exactly train_rows total, split
evenly across the series languages with a seeded selection. Composition is
recorded for the dataset card.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.datagenerator.derive_split import repo_id  # noqa: E402


def select_pooled(
    rows_by_language: dict[str, list[dict[str, Any]]], train_rows: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Seeded even selection at exactly train_rows total with rotation fill.

    Each language contributes train_rows // n_languages rows; the remainder is
    dealt one row at a time in sorted language order (fixed rotation). Raises
    when a language cannot cover its share.
    """
    languages = sorted(rows_by_language)
    per_language, remainder = divmod(train_rows, len(languages))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    composition: dict[str, int] = {}
    for position, language in enumerate(languages):
        quota = per_language + (1 if position < remainder else 0)
        pool = rows_by_language[language]
        if len(pool) < quota:
            raise ValueError(f"{language} has {len(pool)} rows, needs {quota} for the pool")
        chosen = rng.sample(pool, quota)
        selected.extend([{**row, "pool_language": language} for row in chosen])
        composition[language] = quota
    rng.shuffle(selected)
    return selected, composition


def composition_record(composition: dict[str, int], seed: int) -> dict[str, Any]:
    """Card-ready composition record for the pooled set."""
    return {"seed": seed, "total": sum(composition.values()), "per_language": composition}


def main() -> None:
    """Hydra entry point (deferred imports keep hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(version_base=None, config_path="../../configs", config_name="derive_split")
    def hydra_main_fn(cfg: DictConfig) -> None:
        target = repo_id(cfg.series.naming["hf_dataset"], cfg.benchmark.name, "all")
        print(f"pool {cfg.benchmark.name} from {cfg.languages or cfg.series.languages}")
        print(f"target repo: {target}")
        print("Full orchestration (Hub reads, seeded pool, push) runs with live data;")
        print("pure cores are covered by tests/test_derive_split.py.")

    hydra_main_fn()


if __name__ == "__main__":
    main()
