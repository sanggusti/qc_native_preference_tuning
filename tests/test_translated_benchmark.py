"""No-network tests for the translated benchmark task (local JSONL fixtures)."""

from pathlib import Path

import pytest
from inspect_ai import Task

from src.evals.tasks.translated_benchmark import translated_benchmark

FIXTURES = Path(__file__).parent / "fixtures"


def test_numeric_benchmark_builds_task():
    result = translated_benchmark(
        benchmark="gsm8k", language="id", dataset_path=str(FIXTURES / "gsm8k_id_sample.jsonl")
    )
    assert isinstance(result, Task)
    assert result.metadata["benchmark"] == "gsm8k"
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
            benchmark="gsm8k", language="zz", dataset_path=str(FIXTURES / "gsm8k_id_sample.jsonl")
        )


def test_limit_applies():
    result = translated_benchmark(
        benchmark="gsm8k",
        language="id",
        dataset_path=str(FIXTURES / "gsm8k_id_sample.jsonl"),
        limit=1,
    )
    assert len(result.dataset) == 1


def test_variant_and_dataset_repo_resolution():
    from src.evals.tasks.translated_benchmark import resolve_repo

    assert (
        resolve_repo("gsm8k", "jv", None, None, "sanggusti/{benchmark}-{language}")
        == "sanggusti/gsm8k-jv"
    )
    assert (
        resolve_repo("gsm8k", "jv", "rt", None, "sanggusti/{benchmark}-{language}")
        == "sanggusti/gsm8k-jv-rt"
    )
    assert resolve_repo("gsm8k", "jv", "rt", "sanggusti/custom", "x") == "sanggusti/custom"
    result = translated_benchmark(
        benchmark="gsm8k",
        language="jv",
        variant="rt",
        dataset_path=str(FIXTURES / "gsm8k_id_sample.jsonl"),
    )
    assert result.metadata["variant"] == "rt"
    assert result.metadata["dataset"].endswith("gsm8k_id_sample.jsonl")


def test_locale_numeral_normalization():
    from inspect_ai.scorer._common import match_str

    from src.evals.tasks.translated_benchmark import normalize_numerals

    # Indonesian thousands separator against an integer target
    assert normalize_numerals("Jawaban: 1.500", "1500") == "Jawaban: 1500"
    assert match_str("Jawaban: 1.500", "1500", location="end", numeric=True)[1] is False
    assert match_str(
        normalize_numerals("Jawaban: 1.500", "1500"), "1500", location="end", numeric=True
    )[1]
    # decimal comma against a fractional target
    assert normalize_numerals("Jawaban: 2,5", "2.5") == "Jawaban: 2.5"
    assert match_str(
        normalize_numerals("Jawaban: 2,5", "2.5"), "2.5", location="end", numeric=True
    )[1]
    # English formatting is untouched
    assert normalize_numerals("Answer: 1,500", "1500") == "Answer: 1,500"
    assert normalize_numerals("Answer: 18", "18") == "Answer: 18"
    # a period inside a non-integer target is not a thousands separator
    assert normalize_numerals("Jawaban: 3.25", "3.25") == "Jawaban: 3.25"


def test_numeric_benchmark_has_two_scorers():
    result = translated_benchmark(
        benchmark="gsm8k", language="id", dataset_path=str(FIXTURES / "gsm8k_id_sample.jsonl")
    )
    assert isinstance(result.scorer, list) and len(result.scorer) == 2
