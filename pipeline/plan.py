"""Plan a series matrix and print the commands that execute it.

Resolves one series config through ``src.utils.registry``, expands the
finetune and eval matrices, and prints either a count summary or the
shell commands for the datagen, finetune, and eval stages.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra import main as hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.utils.registry import EvalCell, TrainCell

DATAGEN_MODULE = "pipeline.datagenerator.translate_benchmark"
FINETUNE_MODULE = "pipeline.training.autoscientist_finetune"
EVAL_TASK = "src/evals/tasks/translated_benchmark.py"


def _split_dataset_id(dataset_id: str) -> tuple[str, str] | None:
    """Split ``sanggusti/{benchmark}-{language}`` into (benchmark, language).

    Returns None for pooled ids (derived by mixing, not translated directly)
    and for ids that do not carry a language suffix.
    """
    name = dataset_id.split("/", 1)[1] if "/" in dataset_id else dataset_id
    benchmark, sep, language = name.rpartition("-")
    if not sep or not benchmark or not language or language == "all":
        return None
    return benchmark, language


def build_datagen_commands(datasets: list[str], series: str) -> list[str]:
    """Build one benchmark translation command per dataset id."""
    commands = []
    for dataset_id in datasets:
        parsed = _split_dataset_id(dataset_id)
        if parsed is None:
            continue
        benchmark, language = parsed
        commands.append(
            "uv run python -m "
            f"{DATAGEN_MODULE} benchmark={benchmark} language={language} "
            f"series={series} push_to_hub={dataset_id}"
        )
    return commands


def build_finetune_commands(cells: Sequence[TrainCell]) -> list[str]:
    """Build one AutoScientist finetune command per train cell."""
    return [
        "uv run python -m "
        f"{FINETUNE_MODULE} series={cell.series} benchmark={cell.benchmark} "
        f"base={cell.base_role} language={cell.train_language} "
        f"replicate={cell.replicate} hub_model_id={cell.hub_model_id}"
        for cell in cells
    ]


def build_eval_commands(cells: Sequence[EvalCell]) -> list[str]:
    """Build one inspect eval command per eval cell."""
    return [
        "uv run inspect eval "
        f"{EVAL_TASK} --model {cell.model_ref} "
        f"-T benchmark={cell.benchmark} -T language={cell.eval_language} "
        f"-T split=test --log-dir {cell.log_dir}"
        for cell in cells
    ]


def _count_by_condition(cells: Sequence[Any], label: str) -> list[str]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.condition] = counts.get(cell.condition, 0) + 1
    lines = [f"{label} by condition:"]
    for condition in sorted(counts):
        lines.append(f"  {condition}: {counts[condition]}")
    return lines


def format_matrix(
    train_cells: Sequence[TrainCell],
    eval_cells: Sequence[EvalCell],
    datasets: Sequence[str],
    series_name: str = "",
) -> str:
    """Format matrix counts and the per-condition breakdown as text."""
    lines = [
        f"Series: {series_name}",
        f"Finetune cells: {len(train_cells)}",
        f"Eval cells: {len(eval_cells)}",
        f"Unique datasets: {len(datasets)}",
        *_count_by_condition(train_cells, "Finetune cells"),
        *_count_by_condition(eval_cells, "Eval cells"),
        "Datasets:",
        *[f"  {dataset_id}" for dataset_id in datasets],
    ]
    return "\n".join(lines)


def _series_name(cfg: DictConfig) -> str:
    group = cfg.get("series")
    if isinstance(group, str):
        return group
    if group is not None and "series" in group:
        return str(group["series"])
    raise ValueError("series config must declare its 'series' id")


@hydra_main(version_base=None, config_path="../configs", config_name="plan")
def main(cfg: DictConfig) -> None:
    from src.utils.registry import expand_matrix, unique_datasets

    name = _series_name(cfg)
    filt = cfg.filter
    train_cells = expand_matrix(
        name,
        stage="finetune",
        tier=filt.tier,
        benchmark=filt.benchmark,
        language=filt.language,
        condition=filt.condition,
        base=filt.base,
    )
    eval_cells = expand_matrix(
        name,
        stage="eval",
        tier=filt.tier,
        benchmark=filt.benchmark,
        language=filt.language,
        condition=filt.condition,
        base=filt.base,
    )
    datasets = unique_datasets(name)

    stages = ["datagen", "finetune", "eval"] if cfg.stage in (None, "all") else [cfg.stage]
    if cfg.format in ("matrix", "all"):
        print(format_matrix(train_cells, eval_cells, datasets, name))
    if cfg.format in ("commands", "all"):
        if "datagen" in stages:
            for command in build_datagen_commands(list(datasets), name):
                print(command)
        if "finetune" in stages:
            for command in build_finetune_commands(train_cells):
                print(command)
        if "eval" in stages:
            for command in build_eval_commands(eval_cells):
                print(command)


if __name__ == "__main__":
    main()
