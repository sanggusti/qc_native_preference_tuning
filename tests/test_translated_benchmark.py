"""No-network tests for the translated benchmark task (local JSONL fixtures)."""

from pathlib import Path

import pytest
from inspect_ai import Task

from src.evals.tasks.translated_benchmark import translated_benchmark

FIXTURES = Path(__file__).parent / "fixtures"


def test_numeric_benchmark_builds_task():
    result = translated_benchmark(
        benchmark="mgsm", language="id", dataset_path=str(FIXTURES / "mgsm_id_sample.jsonl")
    )
    assert isinstance(result, Task)
    assert result.metadata["benchmark"] == "mgsm"
    assert result.metadata["language"] == "id"
    assert result.metadata["domain"] == "math"
    sample = result.dataset[0]
    assert sample.target == "7"
    assert sample.metadata["source_id"] == "gsm8k-test-0001"
    assert sample.choices is None


def test_choice_benchmark_builds_task():
    result = translated_benchmark(
        benchmark="medqa", language="id", dataset_path=str(FIXTURES / "medqa_id_sample.jsonl")
    )
    assert isinstance(result, Task)
    sample = result.dataset[0]
    assert sample.choices == [
        "Angina stabil",
        "Perikarditis",
        "Refluks gastroesofageal",
        "Pneumotoraks",
    ]
    assert sample.target == "A"
    assert "ANSWER: $LETTER" in sample.metadata["instruction"]


def test_unknown_language_rejected():
    with pytest.raises(FileNotFoundError):
        translated_benchmark(
            benchmark="mgsm", language="zz", dataset_path=str(FIXTURES / "mgsm_id_sample.jsonl")
        )


def test_limit_applies():
    result = translated_benchmark(
        benchmark="mgsm",
        language="id",
        dataset_path=str(FIXTURES / "mgsm_id_sample.jsonl"),
        limit=1,
    )
    assert len(result.dataset) == 1
