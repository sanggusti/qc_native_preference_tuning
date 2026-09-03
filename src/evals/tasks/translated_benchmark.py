"""Translated standard benchmarks as one parameterized Inspect task.

Every cell of a series runs this task with `-T benchmark=<name> -T language=<code>` and,
from the planner, `-T dataset_repo=<resolved repo>` (plus `-T variant=<rt|nllb|pro1>` for
derived evaluation splits). The benchmark's registry entry (configs/benchmark/{name}.yaml)
decides the solver and scorer; the language decides which translated dataset is loaded.
Scorers are the language-agnostic ones from Inspect (numeric match, choice letter), so the
metric itself cannot favor a language.

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

Numeric benchmarks are scored twice: with Inspect's `match(numeric=True)` (the strict
protocol, English number formatting) and with `numeric_match_locale`, which first rewrites
Indonesian-style numerals (1.500 for 1500, 2,5 for 2.5) so that an untuned base model
answering in the local convention is not scored wrong for a formatting reason. Both scores
land in the log; docs/methodology.md says which is primary.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset, json_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    choice,
    match,
    scorer,
    stderr,
)
from inspect_ai.scorer._common import match_str
from inspect_ai.solver import TaskState, generate, multiple_choice, prompt_template

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.registry import load_entry  # noqa: E402

DEFAULT_REPO_TEMPLATE = "sanggusti/{benchmark}-{language}"

_THOUSANDS_PERIOD = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+)(?![\d.])")
_DECIMAL_COMMA = re.compile(r"(?<![\d.,])(\d+),(\d+)(?![\d,])")


def normalize_numerals(text: str, target: str) -> str:
    """Rewrite Indonesian-style numerals in `text` into English formatting.

    A period between digit groups of three is a thousands separator when the target is an
    integer (1.500 -> 1500). A single comma between digits is a decimal separator when the
    target has a fractional part (2,5 -> 2.5). Anything else is left untouched, so English
    formatted answers are unaffected.
    """
    if not text:
        return text
    if "." in target:
        return _DECIMAL_COMMA.sub(r"\1.\2", text)
    return _THOUSANDS_PERIOD.sub(lambda m: m.group(1).replace(".", ""), text)


@scorer(metrics=[accuracy(), stderr(cluster="source_id")])
def numeric_match_locale() -> Scorer:
    """Numeric match after rewriting local numeral conventions into English formatting."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        normalized = normalize_numerals(completion, target.text)
        answer, matched = match_str(
            value=normalized, target=target.text, location="end", ignore_case=True, numeric=True
        )
        return Score(
            value=CORRECT if matched else INCORRECT,
            answer=answer,
            explanation=completion,
            metadata={"normalized": normalized != completion},
        )

    return score


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


def resolve_repo(
    benchmark: str,
    language: str,
    variant: str | None,
    dataset_repo: str | None,
    repo_template: str,
) -> str:
    """The Hub repository for a cell: an explicit id from the planner, or template plus variant."""
    if dataset_repo:
        return dataset_repo
    repo = repo_template.format(benchmark=benchmark, language=language)
    return f"{repo}-{variant}" if variant else repo


def load_translated_dataset(
    repo: str,
    dataset_path: str | None,
    split: str,
    limit: int | None,
) -> Dataset:
    if dataset_path is not None:
        return json_dataset(dataset_path, sample_fields=record_to_sample, limit=limit)
    return hf_dataset(path=repo, split=split, sample_fields=record_to_sample, limit=limit)


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
    dataset_repo: str | None = None,
    variant: str | None = None,
    split: str = "test",
    limit: int | None = None,
    repo_template: str = DEFAULT_REPO_TEMPLATE,
) -> Task:
    """Evaluate a model on a translated standard benchmark.

    Args:
        benchmark: registry name under configs/benchmark/ (gsm8k, medqa, ...).
        language: language code under configs/language/ (id, jv, su, min, ace, en).
        dataset_path: local JSONL in the canonical schema; overrides the HF repo.
        dataset_repo: explicit HF repo id resolved by the planner from the series naming.
        variant: derived evaluation split suffix (rt, nllb, pro1) appended to the repo id.
        split: HF split to load (test for evaluation).
        limit: cap on samples, for smoke runs.
        repo_template: fallback HF repo id template when dataset_repo is not given.
    """
    spec = load_entry("benchmark", benchmark)
    load_entry("language", language)
    repo = resolve_repo(benchmark, language, variant, dataset_repo, repo_template)
    dataset = load_translated_dataset(repo, dataset_path, split, limit)
    template = instruction_template(dataset)

    if spec.scorer == "numeric_match":
        solver = [prompt_template("{instruction}\n\n{prompt}"), generate()]
        scorers = [match(numeric=True), numeric_match_locale()]
    elif spec.scorer == "choice":
        solver = [multiple_choice(template=template)]
        scorers = [choice()]
    else:
        raise NotImplementedError(
            f"scorer {spec.scorer!r} for benchmark {benchmark!r} has no task wiring yet"
        )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        metrics=[accuracy(), stderr(cluster="source_id")],
        metadata={
            "benchmark": benchmark,
            "domain": str(spec.domain),
            "language": language,
            "variant": variant,
            "dataset": dataset_path or repo,
            "inspect_reference": str(spec.inspect_reference),
        },
    )
