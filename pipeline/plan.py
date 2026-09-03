"""Expand an experiment series into its matrix and the commands that run it.

    uv run python -m pipeline.plan                      # table for the default series
    uv run python -m pipeline.plan tier=0               # only the minimum publishable unit
    uv run python -m pipeline.plan format=commands stage=eval

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
    finetunes = unique_finetunes(cells)
    datasets = unique_datasets(cells)
    lines = [
        f"series: {series.series}",
        f"base models: {dict(series.base_models)}",
        f"languages: {list(series.languages)}",
        f"benchmarks: {list(series.benchmarks)}",
        f"conditions: {list(series.active_conditions)}",
        f"replicates: {list(series.replicates)}",
        "",
        f"datasets (translated and derived): {len(datasets)}",
        f"finetune runs:                     {len(finetunes)}",
        f"eval runs:                         {len(cells)}",
        "",
        "per tier (finetunes / evals):",
    ]
    for tier in sorted({c["tier"] for c in cells}):
        n_ft = sum(1 for f in finetunes if f["tier"] == tier)
        n_ev = sum(1 for c in cells if c["tier"] == tier)
        lines.append(f"  tier {tier}: {n_ft:>3} / {n_ev:>3}")
    lines.append("")
    lines.append("eval runs per condition:")
    for condition, count in sorted(Counter(c["condition"] for c in cells).items()):
        lines.append(f"  {condition:<24}{count}")
    lines.append("")
    lines.append("benchmark  eval  variant  condition               base      train  rep  model")
    for cell in cells:
        lines.append(
            f"{cell['benchmark']:<10} {cell['eval_language']:<5} {cell['eval_variant'] or '-':<8} "
            f"{cell['condition']:<23} {cell['base_role']:<9} {cell['train_language'] or '-':<6} "
            f"{cell['replicate']:<4} {cell['model']}"
        )
    return "\n".join(lines)


def render_commands(series: DictConfig, cells: list[dict], stage: str) -> str:
    lines: list[str] = [
        "# Entry points below are the stage modules specified in docs/reproducibility.md;",
        "# a module that does not exist yet is on the roadmap in docs/experiments.md.",
    ]
    if stage in ("all", "datagen"):
        lines.append(
            "# datagen: one translation run per benchmark x language, plus derived eval splits"
        )
        for ds in unique_datasets(cells):
            if ds["language"] == "en" and not ds["variant"]:
                continue
            if ds["language"] == "all":
                lines.append(
                    "uv run python -m pipeline.datagenerator.pool_benchmark "
                    f"benchmark={ds['benchmark']} series={series.series} push_to_hub={ds['repo']}"
                )
                continue
            if ds["variant"]:
                lines.append(
                    "uv run python -m pipeline.datagenerator.derive_split "
                    f"benchmark={ds['benchmark']} language={ds['language']} "
                    f"variant={ds['variant']} series={series.series} push_to_hub={ds['repo']}"
                )
                continue
            lines.append(
                "uv run python -m pipeline.datagenerator.translate_benchmark "
                f"benchmark={ds['benchmark']} language={ds['language']} "
                f"series={series.series} push_to_hub={ds['repo']}"
            )
    if stage in ("all", "finetune"):
        lines.append(
            "# finetune: one AutoScientist run per benchmark x base x train language x replicate"
        )
        for job in unique_finetunes(cells):
            lines.append(
                "uv run python -m pipeline.training.autoscientist_finetune "
                f"series={series.series} benchmark={job['benchmark']} "
                f"base={job['base_role']} language={job['train_language']} "
                f"replicate={job['replicate']} hub_model_id={job['model']}"
            )
    if stage in ("all", "eval"):
        lines.append("# eval: one Inspect run per cell")
        for cell in cells:
            model = cell["model"] if cell["condition"] == "base" else f"hf/{cell['model']}"
            variant = f" -T variant={cell['eval_variant']}" if cell["eval_variant"] else ""
            lines.append(
                "uv run inspect eval src/evals/tasks/translated_benchmark.py "
                f"--model {model} -T benchmark={cell['benchmark']} "
                f"-T language={cell['eval_language']}{variant} "
                f"--temperature {series.generation.temperature} "
                f"--max-tokens {series.generation.max_tokens} "
                f"--metadata condition={cell['condition']} "
                f"--metadata train_language={cell['train_language'] or 'none'} "
                f"--metadata base={cell['base_role']} "
                f"--metadata replicate={cell['replicate']} --metadata series={series.series}"
            )
    return "\n".join(lines)


@hydra_main(version_base=None, config_path="../configs", config_name="plan")
def main(cfg: DictConfig) -> None:
    series = load_series(cfg.series.series)
    cells = expand_matrix(series)
    if cfg.tier is not None:
        cells = [c for c in cells if c["tier"] <= int(cfg.tier)]
    if cfg.format == "commands":
        print(render_commands(series, cells, cfg.stage))
    else:
        print(render_table(series, cells))


if __name__ == "__main__":
    main()
