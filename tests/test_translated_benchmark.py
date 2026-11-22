"""No-network tests for the translated_benchmark eval task."""

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task

from src.evals.tasks.translated_benchmark import samples_from_rows, translated_benchmark


def _gsm8k_rows() -> list[dict[str, Any]]:
    base = {
        "instruction": "Jawab pertanyaan matematika berikut.",
        "instruction_en": "Answer the following math question.",
        "language": "id",
        "register": "formal",
        "blueprint_hash": "abc123",
        "translator_version": "adaption-v1",
    }
    return [
        {
            **base,
            "id": "gsm8k-id-0001",
            "question": "Budi punya 3 apel dan membeli 4 lagi. Berapa semuanya?",
            "choices": [],
            "target": "7",
            "question_en": "Budi has 3 apples and buys 4 more. How many total?",
            "answer_en": "7",
        },
        {
            **base,
            "id": "gsm8k-id-0002",
            "question": "Siti punya 10 kelereng dan memberi 3. Berapa sisanya?",
            "choices": [],
            "target": "7",
            "question_en": "Siti has 10 marbles and gives away 3. How many remain?",
            "answer_en": "7",
        },
    ]


def _medqa_rows() -> list[dict[str, Any]]:
    base = {
        "instruction": "Pilih jawaban yang benar.",
        "instruction_en": "Choose the correct answer.",
        "language": "id",
        "register": "formal",
        "blueprint_hash": "def456",
        "translator_version": "adaption-v1",
    }
    return [
        {
            **base,
            "id": "medqa-id-0001",
            "question": "Manakah organel penghasil energi sel?",
            "choices": ["Nukleus", "Mitokondria", "Ribosom", "Membran sel"],
            "target": "B",
            "question_en": "Which organelle produces energy for the cell?",
            "answer_en": "Mitochondria",
        },
        {
            **base,
            "id": "medqa-id-0002",
            "question": "Manakah pembuluh yang membawa darah ke jantung?",
            "choices": ["Arteri", "Kapiler", "Vena", "Aorta"],
            "target": "C",
            "question_en": "Which vessel carries blood to the heart?",
            "answer_en": "Vein",
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def test_translated_benchmark_returns_task(tmp_path: Path) -> None:
    """translated_benchmark() returns an Inspect Task."""
    path = _write_jsonl(tmp_path / "gsm8k.jsonl", _gsm8k_rows())
    result = translated_benchmark(dataset_path=path)
    assert isinstance(result, Task)


def test_translated_benchmark_metadata(tmp_path: Path) -> None:
    """Task metadata records benchmark, language, split, variant, scorer."""
    path = _write_jsonl(tmp_path / "medqa.jsonl", _medqa_rows())
    result = translated_benchmark(
        benchmark="medqa", language="jv", split="test", dataset_path=path
    )
    assert result.metadata == {
        "benchmark": "medqa",
        "language": "jv",
        "split": "test",
        "instruction_variant": "native",
        "scorer": "choice",
    }


def test_samples_count_matches_rows() -> None:
    """One sample per row."""
    rows = _gsm8k_rows()
    assert len(samples_from_rows(rows, "gsm8k")) == len(rows)


def test_sample_input_contains_instruction_and_question() -> None:
    """Sample input joins instruction, a blank line, and the question."""
    sample = samples_from_rows(_gsm8k_rows(), "gsm8k")[0]
    row = _gsm8k_rows()[0]
    assert sample.input == f"{row['instruction']}\n\n{row['question']}"


def test_scorer_wiring_per_benchmark(tmp_path: Path) -> None:
    """Numeric benchmarks use match, choice benchmarks use choice."""
    numeric = translated_benchmark(dataset_path=_write_jsonl(tmp_path / "g.jsonl", _gsm8k_rows()))
    choice_task = translated_benchmark(
        benchmark="medqa", dataset_path=_write_jsonl(tmp_path / "m.jsonl", _medqa_rows())
    )
    assert numeric.metadata["scorer"] == "numeric"
    assert choice_task.metadata["scorer"] == "choice"
    assert len(numeric.scorer) == 1
    assert len(choice_task.scorer) == 1


def test_choice_rows_pass_choices() -> None:
    """Choice tasks pass options; numeric tasks leave choices empty."""
    medqa = samples_from_rows(_medqa_rows(), "medqa")
    assert medqa[0].choices == _medqa_rows()[0]["choices"]
    gsm8k = samples_from_rows(_gsm8k_rows(), "gsm8k")
    assert gsm8k[0].choices is None


def test_english_cot_uses_instruction_en() -> None:
    """english_cot swaps the native instruction for the English one."""
    native = samples_from_rows(_gsm8k_rows(), "gsm8k")[0]
    cot = samples_from_rows(_gsm8k_rows(), "gsm8k", "english_cot")[0]
    row = _gsm8k_rows()[0]
    assert row["instruction"] in str(native.input)
    assert row["instruction_en"] in str(cot.input)
    assert row["instruction"] not in str(cot.input)
