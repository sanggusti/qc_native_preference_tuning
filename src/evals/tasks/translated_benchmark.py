"""Translated standard benchmarks as one parameterized Inspect task.

Every cell of a series runs this task with `-T benchmark=<name> -T language=<code>`.
The benchmark's registry entry (configs/benchmark/{name}.yaml) decides the solver and
scorer; the language decides which translated dataset is loaded. Scorers are the
language-agnostic ones from Inspect (numeric match, choice letter), so the metric itself
cannot favor a language.

Canonical dataset schema, published by the datagen stage for every language:

    id           stable source item id, identical across languages (pairs the items)
    question     translated question text (vignette only, for multiple choice)
    choices      translated answer options, same order as the source (multiple choice only)
    target       numeric answer as a string, or the correct option letter
    instruction  translated instruction template for this benchmark, same string on
                 every row; must keep the literal formatting marker the scorer parses
                 ("ANSWER: $LETTER" for choice tasks) and, for choice tasks, the
                 {question}, {choices} and {letters} placeholders
    question_en  English source text (provenance only, never shown to the model)
    language     language code
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset, json_dataset
from inspect_ai.scorer import accuracy, choice, match, stderr
from inspect_ai.solver import generate, multiple_choice, prompt_template

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.registry import load_entry  # noqa: E402

DEFAULT_REPO_TEMPLATE = "sanggusti/{benchmark}-{language}"


def record_to_sample(record: dict[str, Any]) -> Sample:
    metadata = {
        "source_id": str(record["id"]),
        "instruction": record["instruction"],
        "language": record.get("language"),
        "question_en": record.get("question_en"),
    }
    return Sample(
        id=str(record["id"]),
        input=record["question"],
        target=str(record["target"]),
        choices=list(record["choices"]) if record.get("choices") else None,
        metadata=metadata,
    )


def load_translated_dataset(
    benchmark: str,
    language: str,
    dataset_path: str | None,
    split: str,
    limit: int | None,
    repo_template: str,
) -> Dataset:
    if dataset_path is not None:
        return json_dataset(dataset_path, sample_fields=record_to_sample, limit=limit)
    return hf_dataset(
        path=repo_template.format(benchmark=benchmark, language=language),
        split=split,
        sample_fields=record_to_sample,
        limit=limit,
    )


def instruction_template(dataset: Dataset) -> str:
    if len(dataset) == 0:
        raise ValueError("translated dataset is empty")
    first = dataset[0]
    instruction = (first.metadata or {}).get("instruction")
    if not instruction:
        raise ValueError("every row needs an 'instruction' column (translated template)")
    return instruction


@task
def translated_benchmark(
    benchmark: str = "gsm8k",
    language: str = "id",
    dataset_path: str | None = None,
    split: str = "test",
    limit: int | None = None,
    repo_template: str = DEFAULT_REPO_TEMPLATE,
) -> Task:
    """Evaluate a model on a translated standard benchmark.

    Args:
        benchmark: registry name under configs/benchmark/ (gsm8k, medqa, ...).
        language: language code under configs/language/ (id, jv, su, min, ace, en).
        dataset_path: local JSONL in the canonical schema; overrides the HF repo.
        split: HF split to load (test for evaluation).
        limit: cap on samples, for smoke runs.
        repo_template: HF repo id template; defaults to the series naming convention.
    """
    spec = load_entry("benchmark", benchmark)
    load_entry("language", language)
    dataset = load_translated_dataset(
        benchmark, language, dataset_path, split, limit, repo_template
    )
    template = instruction_template(dataset)

    if spec.scorer == "numeric_match":
        solver = [prompt_template("{instruction}\n\n{prompt}"), generate()]
        scorer = match(numeric=True)
    elif spec.scorer == "choice":
        solver = [multiple_choice(template=template)]
        scorer = choice()
    else:
        raise NotImplementedError(
            f"scorer {spec.scorer!r} for benchmark {benchmark!r} has no task wiring yet"
        )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorer,
        metrics=[accuracy(), stderr(cluster="source_id")],
        metadata={
            "benchmark": benchmark,
            "domain": str(spec.domain),
            "language": language,
            "inspect_reference": str(spec.inspect_reference),
        },
    )
