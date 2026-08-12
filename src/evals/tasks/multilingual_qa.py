"""
Multilingual question-answering evaluation task.

Evaluates finetuned models on translated QA benchmarks across
Indonesian low-resource languages (Medicine, Programming, General, Science).
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate


@task
def multilingual_qa(
    dataset_path: str | None = None,
    language: str = "id",
    domain: str = "general",
) -> Task:
    """Evaluate a model on multilingual QA in a specific domain and language.

    Args:
        dataset_path: Path to a JSONL file with 'question' and 'answer' fields.
            If None, uses a small built-in sample for smoke testing.
        language: Target language code (e.g. 'id', 'jv', 'su', 'min').
        domain: Evaluation domain ('medicine', 'programming', 'general', 'science').

    Returns:
        An Inspect Task ready to run.
    """
    if dataset_path is not None:
        from inspect_ai.dataset import json_dataset

        dataset = json_dataset(
            dataset_path,
            sample_fields={
                "input": "question",
                "target": "answer",
            },
        )
    else:
        dataset = MemoryDataset(
            samples=[
                Sample(
                    input="Apa ibu kota Indonesia?",
                    target="Jakarta",
                    metadata={"domain": "general", "language": "id"},
                ),
                Sample(
                    input="Apa fungsi utama dari mitokondria?",
                    target="Mitokondria adalah organel yang menghasilkan energi (ATP) untuk sel melalui respirasi seluler.",
                    metadata={"domain": "science", "language": "id"},
                ),
            ],
            name=f"sample_{domain}_{language}",
        )

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=model_graded_qa(),
        metadata={
            "domain": domain,
            "language": language,
        },
    )
