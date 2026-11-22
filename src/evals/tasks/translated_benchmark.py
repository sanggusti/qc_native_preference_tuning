"""Inspect eval task for translated benchmarks.

Loads canonical translated rows from a local JSONL file or from
``sanggusti/{benchmark}-{language}`` on the Hub, then scores with the scorer
registered for the benchmark: numeric match for math tasks, letter choice
for multiple-choice tasks.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer, choice, match
from inspect_ai.solver import generate

SCORER_NUMERIC = "numeric"
SCORER_CHOICE = "choice"

_FALLBACK_SCORER = {
    "gsm8k": SCORER_NUMERIC,
    "mgsm": SCORER_NUMERIC,
    "medqa": SCORER_CHOICE,
    "mmlu": SCORER_CHOICE,
}


def resolve_scorer(benchmark: str) -> str:
    """Return the registered scorer for a benchmark.

    Prefers ``src.utils.registry`` when it knows the benchmark, else falls
    back to the local table so JSONL-driven runs need no registry entry.
    """
    try:
        from src.utils.registry import load_benchmark

        scorer = load_benchmark(benchmark).get("scorer")
    except (ImportError, FileNotFoundError, KeyError, ValueError):
        scorer = None
    if scorer:
        return str(scorer)
    try:
        return _FALLBACK_SCORER[benchmark]
    except KeyError:
        raise ValueError(f"unknown benchmark: {benchmark!r}") from None


def scorer_for(scorer: str) -> Scorer:
    """Return the Inspect scorer for a scorer name."""
    if scorer == SCORER_NUMERIC:
        return match(numeric=True)
    if scorer == SCORER_CHOICE:
        return choice()
    raise ValueError(f"unknown scorer: {scorer!r}")


def samples_from_rows(
    rows: list[dict[str, Any]],
    benchmark_name: str,
    instruction_variant: str = "native",
) -> list[Sample]:
    """Build Inspect samples from canonical translated rows.

    Sample input is the instruction, a blank line, then the question. The
    ``english_cot`` variant swaps in the English instruction. Choice tasks
    pass the options through; other tasks leave choices empty.
    """
    scorer = resolve_scorer(benchmark_name)
    samples = []
    for row in rows:
        if instruction_variant == "english_cot":
            instruction = row["instruction_en"]
        else:
            instruction = row["instruction"]
        text = f"{instruction}\n\n{row['question']}"
        choices = list(row["choices"]) if scorer == SCORER_CHOICE else None
        samples.append(
            Sample(
                input=text,
                target=row["target"],
                choices=choices,
                id=row["id"],
                metadata={
                    "benchmark": benchmark_name,
                    "language": row["language"],
                    "register": row["register"],
                    "instruction_variant": instruction_variant,
                    "question_en": row["question_en"],
                    "answer_en": row["answer_en"],
                    "blueprint_hash": row["blueprint_hash"],
                    "translator_version": row["translator_version"],
                },
            )
        )
    return samples


def _load_rows(
    dataset_path: str | None, benchmark: str, language: str, split: str
) -> list[dict[str, Any]]:
    if dataset_path is not None:
        rows: list[dict[str, Any]] = []
        with open(dataset_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(f"sanggusti/{benchmark}-{language}", split=split)]


@task
def translated_benchmark(
    benchmark: str = "gsm8k",
    language: str = "id",
    split: str = "test",
    dataset_path: str | None = None,
    instruction_variant: str = "native",
) -> Task:
    """Evaluate a model on one translated benchmark split.

    Args:
        benchmark: Benchmark name (e.g. 'gsm8k', 'medqa').
        language: Target language code (e.g. 'id', 'jv', 'su', 'min').
        split: Dataset split to evaluate.
        dataset_path: Local JSONL with canonical rows; when None, rows load
            from ``sanggusti/{benchmark}-{language}`` on the Hub.
        instruction_variant: 'native' uses the translated instruction,
            'english_cot' uses the English instruction.

    Returns:
        An Inspect Task ready to run.
    """
    rows = _load_rows(dataset_path, benchmark, language, split)
    scorer = resolve_scorer(benchmark)
    dataset = MemoryDataset(
        samples=samples_from_rows(rows, benchmark, instruction_variant),
        name=f"{benchmark}_{language}_{split}",
    )
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=scorer_for(scorer),
        metadata={
            "benchmark": benchmark,
            "language": language,
            "split": split,
            "instruction_variant": instruction_variant,
            "scorer": scorer,
        },
    )
