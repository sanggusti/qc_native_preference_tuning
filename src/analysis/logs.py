"""Tidy assembly of Inspect logs joined to dataset quality columns (issue #56).

Reads one eval log per cell, normalizes per-sample correctness under each
scorer, and left-joins the published per-item quality columns by source_id.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def to_correct(value: Any) -> float | None:
    """Normalize a score value to 1.0/0.0, or None when unscorable."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in ("C", "CORRECT", "1", "TRUE"):
        return 1.0
    if text in ("I", "INCORRECT", "0", "FALSE"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def load_quality_rows(source: str | Path) -> dict[str, dict[str, Any]]:
    """Load {source_id: quality columns} from JSONL or a Hub dataset id."""
    path = Path(str(source))
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        from datasets import load_dataset

        repo, _, split = str(source).partition("@")
        rows = [dict(row) for row in load_dataset(repo, split=split or "test")]
    return {str(row.get("source_id", row.get("id"))): row for row in rows}


def tidy_log(
    log_path: str,
    quality_by_id: dict[str, dict[str, Any]] | None = None,
    lid_fn: Callable[[list[str]], list[tuple[str, float]]] | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Assemble tidy rows for one eval log (cell metadata + samples + quality)."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(log_path)
    meta = dict(log.eval.metadata or {})
    task_args = dict(log.eval.task_args or {})
    completions: list[str] = []
    rows: list[dict[str, Any]] = []
    for sample in log.samples or []:
        completion = sample.output.completion if sample.output else ""
        completions.append(completion)
        sample_metadata = dict(getattr(sample, "metadata", None) or {})
        source_id = str(sample_metadata.get("source_id", sample.id))
        correct = {
            name: to_correct(score.value) for name, score in (sample.scores or {}).items()
        }
        words = len(completion.split())
        row = {
            "benchmark": meta.get("benchmark", task_args.get("benchmark")),
            "language": meta.get("language", task_args.get("language")),
            "variant": meta.get("variant", task_args.get("variant")),
            "dataset": meta.get("dataset", task_args.get("dataset_repo")),
            "condition": meta.get("condition"),
            "train_language": meta.get("train_language"),
            "base": meta.get("base"),
            "replicate": meta.get("replicate"),
            "series": meta.get("series"),
            "item": str(sample.id),
            "source_id": source_id,
            "correct": correct,
            "completion_chars": len(completion),
            "completion_words": words,
            "cap_hit_approx": words > max_tokens if max_tokens else False,
        }
        if quality_by_id and source_id in quality_by_id:
            row["quality"] = quality_by_id[source_id]
        rows.append(row)
    if lid_fn is not None:
        labels = lid_fn(completions)
        if len(labels) != len(rows):
            raise ValueError("lid_fn must return one (label, prob) per completion")
        for row, (label, _) in zip(rows, labels, strict=True):
            row["output_language"] = label
    return rows


def to_dataframe(rows: list[dict[str, Any]]):
    """Tidy rows as a pandas DataFrame (dict columns kept as objects)."""
    import pandas as pd

    return pd.DataFrame(rows)
