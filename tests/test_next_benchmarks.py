"""Tests for next benchmarks: registry entries, record layouts, code_exec wiring."""

import json

from inspect_ai import Task

from src.evals.tasks.translated_benchmark import (
    code_exec_verify,
    record_to_sample,
    translated_benchmark,
)
from src.utils.registry import expand_matrix, load_entry, load_series


def test_arc_entry_schema_and_placeholders():
    entry = load_entry("benchmark", "arc")
    assert entry["scorer"] == "choice"
    assert entry["scorer_language_agnostic"] is True
    assert entry["record_layout"] == "arc"
    assert "{language}" in entry["translate"]["blueprint"]
    assert "{register_instruction}" in entry["translate"]["blueprint"]
    assert "ANSWER:" in entry["instruction_en"]
    assert "{question}" in entry["instruction_en"] and "{choices}" in entry["instruction_en"]


def test_mmlu_entry_mmmlu_layout():
    entry = load_entry("benchmark", "mmlu")
    assert entry["scorer"] == "choice"
    assert entry["record_layout"] == "mmmlu"
    assert "{language}" in entry["translate"]["blueprint"]
    assert "ANSWER:" in entry["instruction_en"]


def test_humaneval_entry_code_exec():
    entry = load_entry("benchmark", "humaneval")
    assert entry["scorer"] == "code_exec"
    assert entry["scorer_language_agnostic"] is True
    assert entry["record_layout"] == "humaneval"
    assert "{language}" in entry["translate"]["blueprint"]
    assert "test" in entry["translate"]["preserve"][0] or entry["translate"]["fields"] == ["prompt"]


def test_record_to_sample_mmmlu_layout():
    record = {
        "id": "m-1", "Question": "Q?", "A": "a", "B": "b", "C": "c", "D": "d",
        "Answer": "C", "Subject": "math", "instruction": "I", "language": "jv",
    }
    sample = record_to_sample(record, {"record_layout": "mmmlu"})
    assert sample.input == "Q?"
    assert sample.choices == ["a", "b", "c", "d"]
    assert sample.target == "C"
    assert sample.metadata["source_id"] == "m-1" and sample.metadata["subject"] == "math"


def test_record_to_sample_arc_nested_layout():
    record = {
        "id": "a-1", "question": "Q?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
        "answerKey": "B", "instruction": "I", "language": "jv",
    }
    sample = record_to_sample(record, {"record_layout": "arc"})
    assert sample.target == "B"
    assert sample.choices == ["a", "b", "c", "d"]
    assert sample.metadata["answerKey"] == "B"


def test_record_to_sample_humaneval_layout_carries_code_metadata():
    record = {
        "id": "h-1", "question": "def f():\n    '''Do x.'''\n    pass",
        "test": "def check(f):\n    assert f() is None",
        "entry_point": "f", "canonical_solution": "def f():\n    pass",
        "instruction": "Complete it.", "language": "jv",
    }
    sample = record_to_sample(record, {"record_layout": "humaneval"})
    assert sample.input.startswith("def f():")
    assert sample.metadata["test"].startswith("def check")
    assert sample.metadata["entry_point"] == "f"
    assert sample.metadata["source_id"] == "h-1"


def test_record_to_sample_rejects_unknown_layout():
    try:
        record_to_sample({"id": "x"}, {"record_layout": "nope"})
    except ValueError as exc:
        assert "unknown record_layout" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_code_exec_task_builds_from_fixture_without_sandbox(tmp_path):
    rows = [
        {
            "id": "h-1", "question": "def f():\n    '''Tambah.'''\n    pass",
            "test": "def check(f):\n    assert True", "entry_point": "f",
            "canonical_solution": "def f():\n    pass",
            "instruction": "Complete the function.", "language": "jv",
        }
    ]
    path = tmp_path / "humaneval.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    task = translated_benchmark(benchmark="humaneval", language="jv", dataset_path=str(path))
    assert isinstance(task, Task)
    assert len(task.scorer) == 1
    assert len(task.dataset) == 1
    assert task.dataset[0].metadata["entry_point"] == "f"


def test_code_exec_verify_scorer_constructs():
    assert code_exec_verify(timeout=5) is not None


def test_extended_series_expands_new_benchmarks():
    series = load_series("s01_extended")
    assert series["benchmarks"] == ["gsm8k", "medqa", "arc", "mmlu", "humaneval"]
    cells = expand_matrix(series)
    by_benchmark = {benchmark for benchmark in (cell["benchmark"] for cell in cells)}
    assert {"arc", "mmlu", "humaneval"} <= by_benchmark
    assert all(cell["tier"] >= 3 for cell in cells if cell["benchmark"] in ("arc", "mmlu", "humaneval"))
    tier0 = [cell for cell in cells if cell["tier"] == 0]
    assert {cell["benchmark"] for cell in tier0} == {"gsm8k"}
