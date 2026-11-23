"""Eval sweep runner: smoke-gate every cell, run it, log it (issue #55).

Cells come from the series registry; argv matches the planner exactly
(render via pipeline.plan.eval_argv). Live paths (subprocess, Hub, wandb,
GlotLID) sit behind injectable seams; tests use fakes.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.plan import eval_argv  # noqa: E402

ANSWER_LINE = re.compile(r"^answer\s*:", re.IGNORECASE | re.MULTILINE)


def filter_cells(
    cells: list[dict[str, Any]],
    tier: int | None = None,
    benchmark: str | None = None,
    language: str | None = None,
    condition: str | None = None,
    base: str | None = None,
) -> list[dict[str, Any]]:
    """Filter expanded cells (language matches the eval language)."""
    return [
        cell
        for cell in cells
        if (tier is None or cell["tier"] == tier)
        and (benchmark is None or cell["benchmark"] == benchmark)
        and (language is None or cell["eval_language"] == language)
        and (condition is None or cell["condition"] == condition)
        and (base is None or cell["base_role"] == base)
    ]


def order_cells_checkpoint_first(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by model with en/id cells first inside each group for early abort."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(cell["model"], []).append(cell)
    ordered = []
    for model in sorted(groups):
        group = groups[model]
        ordered.extend(sorted(group, key=lambda c: (c["eval_language"] not in ("en", "id"), c["eval_language"])))
    return ordered


def should_abort_checkpoint(smoke_results: dict[str, bool]) -> bool:
    """Abort remaining cells when both en and id smokes failed."""
    return smoke_results.get("en") is False and smoke_results.get("id") is False


def smoke_gate(
    samples: list[dict[str, Any]], scorer: str
) -> tuple[bool, dict[str, Any]]:
    """Smoke verdict from limit-20 rows: parsed answers plus ANSWER lines for choice."""
    accuracy = statistics.fmean([sample["score"] for sample in samples]) if samples else 0.0
    answer_lines = sum(1 for sample in samples if ANSWER_LINE.search(sample.get("completion", "")))
    passes = accuracy > 0 and (scorer != "choice" or answer_lines > 0)
    return passes, {"accuracy": accuracy, "answer_lines": answer_lines, "n": len(samples)}


def cell_log_dir(log_root: str, series: str, cell: dict[str, Any]) -> str:
    """One log directory per cell (no collisions across replicates/variants)."""
    variant = f"-{cell['eval_variant']}" if cell.get("eval_variant") else ""
    return str(
        Path(log_root)
        / series
        / cell["benchmark"]
        / cell["eval_language"]
        / f"{cell['condition']}{variant}"
        / f"{cell['base_role']}-r{cell['replicate']}"
    )


def cell_argv(
    series: Any, cell: dict[str, Any], log_dir: str, limit: int | None = None
) -> list[str]:
    """Planner argv for one cell plus log dir and an optional smoke limit."""
    argv = ["uv", "run", *eval_argv(series, cell), "--log-dir", log_dir]
    if limit is not None:
        argv += ["--limit", str(limit)]
    return argv


def export_cell_results(
    log_path: str,
    max_tokens: int,
    lid_fn: Callable[[list[str]], list[tuple[str, float]]] | None = None,
) -> dict[str, Any]:
    """Export per-sample scores, tokens, cap hits and language labels from a log."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(log_path)
    samples = []
    completions = []
    for sample in log.samples or []:
        completion = sample.output.completion if sample.output else ""
        completions.append(completion)
        scores = {name: score.value for name, score in (sample.scores or {}).items()}
        words = len(completion.split())
        samples.append({
            "source_id": (sample.metadata or {}).get("source_id"),
            "scores": scores,
            "completion_chars": len(completion),
            "completion_words": words,
            "cap_hit_approx": words > max_tokens,
        })
    if lid_fn is not None:
        labels = lid_fn(completions)
        if len(labels) != len(samples):
            raise ValueError("lid_fn must return one (label, prob) per completion")
        for sample, (label, _) in zip(samples, labels, strict=True):
            sample["lid_label"] = label
    accuracy: dict[str, float] = {}
    for name in {key for sample in samples for key in sample["scores"]}:
        values = [float(sample["scores"][name]) for sample in samples if name in sample["scores"]]
        accuracy[name] = statistics.fmean(values) if values else 0.0
    return {"n": len(samples), "accuracy": accuracy, "samples": samples}


def run_name_for(cell: dict[str, Any]) -> str:
    """wandb run name from cell fields."""
    return (
        f"eval-{cell['benchmark']}-{cell['eval_language']}-{cell['condition']}-"
        f"{cell['base_role']}-r{cell['replicate']}"
    )


def maybe_log_wandb(
    project: str, run_name: str, tags: list[str], config: Any, metrics: dict[str, Any]
) -> None:
    """Log one cell to wandb; warn and continue when wandb is not installed."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping run logging (Inspect log still kept)")
        return
    run = wandb.init(project=project, name=run_name, tags=tags, config=dict(config))
    run.log(metrics)
    run.finish()


def run_command(argv: list[str]) -> str:
    """Run one command, returning stdout (raises on failure)."""
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed: {completed.stderr[-2000:]}")
    return completed.stdout


def main() -> None:
    """Hydra entry point (deferred imports keep hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(version_base=None, config_path="../../configs", config_name="eval_sweep")
    def hydra_main_fn(cfg: DictConfig) -> None:
        from src.utils.registry import expand_matrix

        series = cfg.series
        cells = filter_cells(
            expand_matrix(series),
            cfg.filter.tier, cfg.filter.benchmark, cfg.filter.language,
            cfg.filter.condition, cfg.filter.base,
        )
        cells = order_cells_checkpoint_first(cells)
        print(f"Sweep {series['series']}: {len(cells)} cells (dry_run={cfg.dry_run})")
        print(f"Pre-flight: Modal serving on {cfg.gpu} with timeout {cfg.timeout}; "
              "one server per checkpoint; smoke --limit 20 gates every full run.")
        if cfg.dry_run:
            for cell in cells:
                log_dir = cell_log_dir(cfg.output_dir, series["series"], cell)
                print(" ".join(cell_argv(series, cell, log_dir)))
            return
        raise NotImplementedError("live sweep executes with deployed endpoints (Phase 2/5, gated)")

    hydra_main_fn()


if __name__ == "__main__":
    main()
