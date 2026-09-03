"""Expand an experiment series into its matrix and the commands that run it.

    uv run python -m pipeline.plan                      # table for the default series
    uv run python -m pipeline.plan series=s01_language_medium format=commands stage=eval

Nothing here submits work. It exists so that the full sweep is derived from configs and
can be reviewed (and pasted into docs/experiments.md) before any paid run.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from hydra import main as hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.registry import (  # noqa: E402
    expand_matrix,
    load_series,
    unique_datasets,
    unique_finetunes,
)


def render_table(series: DictConfig, cells: list[dict]) -> str:
    lines = [
        f"series: {series.series}",
        f"base_model: {series.base_model}",
        f"languages: {list(series.languages)}",
        f"benchmarks: {list(series.benchmarks)}",
        f"conditions: {list(series.active_conditions)}",
        f"seeds: {list(series.seeds)}",
        "",
        f"translated datasets: {len(unique_datasets(cells))}",
        f"finetune runs:       {len(unique_finetunes(cells))}",
        f"eval runs:           {len(cells)}",
        "",
        "eval runs per condition:",
    ]
    for condition, count in sorted(Counter(c["condition"] for c in cells).items()):
        lines.append(f"  {condition:<16}{count}")
    lines.append("")
    lines.append("benchmark  eval  condition       train  seed  model")
    for cell in cells:
        lines.append(
            f"{cell['benchmark']:<10} {cell['eval_language']:<5} {cell['condition']:<15} "
            f"{cell['train_language'] or '-':<6} {cell['seed']:<5} {cell['model']}"
        )
    return "\n".join(lines)


def render_commands(series: DictConfig, cells: list[dict], stage: str) -> str:
    lines: list[str] = [
        "# Entry points below are the stage modules specified in docs/reproducibility.md;",
        "# a module that does not exist yet is on the roadmap in docs/experiments.md.",
    ]
    if stage in ("all", "datagen"):
        lines.append("# datagen: one Adaption translation run per benchmark x language")
        for ds in unique_datasets(cells):
            if ds["language"] == "en":
                continue
            if ds["language"] == "all":
                lines.append(
                    "uv run python -m pipeline.datagenerator.pool_benchmark "
                    f"benchmark={ds['benchmark']} series={series.series} push_to_hub={ds['repo']}"
                )
                continue
            lines.append(
                "uv run python -m pipeline.datagenerator.translate_benchmark "
                f"benchmark={ds['benchmark']} language={ds['language']} "
                f"series={series.series} push_to_hub={ds['repo']}"
            )
    if stage in ("all", "finetune"):
        lines.append("# finetune: one AutoScientist run per benchmark x train language x seed")
        for job in unique_finetunes(cells):
            lines.append(
                "uv run python -m pipeline.training.autoscientist_finetune "
                f"series={series.series} benchmark={job['benchmark']} "
                f"language={job['train_language']} seed={job['seed']} "
                f"hub_model_id={job['model']}"
            )
    if stage in ("all", "eval"):
        lines.append("# eval: one Inspect run per cell")
        for cell in cells:
            model = cell["model"] if cell["condition"] == "base" else f"hf/{cell['model']}"
            lines.append(
                "uv run inspect eval src/evals/tasks/translated_benchmark.py "
                f"--model {model} -T benchmark={cell['benchmark']} "
                f"-T language={cell['eval_language']} "
                f"--temperature {series.generation.temperature} "
                f"--max-tokens {series.generation.max_tokens} "
                f"--metadata condition={cell['condition']} "
                f"--metadata train_language={cell['train_language'] or 'none'} "
                f"--metadata seed={cell['seed']} --metadata series={series.series}"
            )
    return "\n".join(lines)


@hydra_main(version_base=None, config_path="../configs", config_name="plan")
def main(cfg: DictConfig) -> None:
    series = load_series(cfg.series.series)
    cells = expand_matrix(series)
    if cfg.format == "commands":
        print(render_commands(series, cells, cfg.stage))
    else:
        print(render_table(series, cells))


if __name__ == "__main__":
    main()
