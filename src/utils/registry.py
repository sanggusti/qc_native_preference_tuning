"""Registry loaders for the language, benchmark and series config groups.

Experiments are configs: a series file lists language codes and benchmark names, and
each of those resolves to one YAML under configs/language/ or configs/benchmark/.
`expand_matrix` turns a series into the list of experiment cells that the datagen,
finetune and eval stages iterate over, with every artifact name derived from the
series naming templates. No stage should compute a name by hand.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

LANGUAGE_REQUIRED = ("code", "iso_639_3", "name", "flores_code", "script", "resource_tier")
BENCHMARK_REQUIRED = ("name", "domain", "scorer", "eval_source", "train_source", "translate")
SERIES_REQUIRED = (
    "series",
    "base_models",
    "languages",
    "benchmarks",
    "conditions",
    "active_conditions",
    "replicates",
    "naming",
)


def load_entry(group: str, name: str) -> DictConfig:
    """Load one registry entry (configs/{group}/{name}.yaml) and validate required keys."""
    path = CONFIGS_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No {group} entry named {name!r}: expected {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{path} must be a mapping")
    required = {
        "language": LANGUAGE_REQUIRED,
        "benchmark": BENCHMARK_REQUIRED,
        "series": SERIES_REQUIRED,
    }[group]
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    return cfg


def list_entries(group: str) -> list[str]:
    """Names of all non-template entries in a config group."""
    return sorted(
        p.stem for p in (CONFIGS_DIR / group).glob("*.yaml") if not p.stem.startswith("_")
    )


def load_series(name: str) -> DictConfig:
    """Load a series and validate every language, benchmark and condition it references."""
    series = load_entry("series", name)
    for code in series.languages:
        load_entry("language", code)
    for bench in series.benchmarks:
        load_entry("benchmark", bench)
    unknown = [c for c in series.active_conditions if c not in series.conditions]
    if unknown:
        raise ValueError(f"series {name!r} activates unknown conditions: {unknown}")
    for cond_name, cond in series.conditions.items():
        for base in cond.get("bases") or []:
            if base not in series.base_models:
                raise ValueError(f"condition {cond_name!r} names unknown base model role {base!r}")
        for bench in cond.get("benchmarks") or []:
            if bench not in series.benchmarks:
                raise ValueError(f"condition {cond_name!r} names benchmark {bench!r} not in series")
    return series


def _train_languages(series: DictConfig, condition: str, eval_language: str) -> list[str | None]:
    spec = series.conditions[condition].train
    if spec is None:
        return [None]
    if spec == "same":
        return [eval_language]
    if spec == "other":
        return [code for code in series.languages if code != eval_language]
    if spec == "all":
        return ["all"]  # one model trained on the pooled dataset of every language
    return [str(spec)]


def _format(template: str, **values: Any) -> str:
    return template.format(**{k: ("" if v is None else v) for k, v in values.items()})


def expand_matrix(series: DictConfig) -> list[dict[str, Any]]:
    """Expand a series into experiment cells.

    Each cell is one (benchmark, eval_language, condition, base, train_language, replicate)
    tuple with the HF and wandb names it should produce. Untuned baseline cells carry
    replicate 0. A cell that repeats an earlier (benchmark, eval language, eval variant,
    base, train language, replicate) is dropped, so conditions that coincide for some
    language (english_anchor in English) are not double counted.
    """
    naming = series.naming
    cells: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for bench, lang, condition in product(
        series.benchmarks, series.languages, series.active_conditions
    ):
        cond = series.conditions[condition]
        if cond.get("benchmarks") and bench not in cond.benchmarks:
            continue
        eval_only = cond.get("eval_languages")
        if eval_only is not None and lang not in eval_only:
            continue
        eval_not = cond.get("exclude_eval_languages")
        if eval_not is not None and lang in eval_not:
            continue
        variant = cond.get("eval_variant")
        eval_dataset = _format(naming.hf_dataset, benchmark=bench, language=lang)
        if variant:
            eval_dataset = f"{eval_dataset}-{variant}"
        for base_role in cond.get("bases") or ["primary"]:
            base_id = series.base_models[base_role]
            base_slug = naming.base_slugs[base_role]
            for train_lang in _train_languages(series, condition, lang):
                replicates = [0] if train_lang is None else list(series.replicates)
                for rep in replicates:
                    key = (bench, lang, variant, base_role, train_lang, rep)
                    if key in seen:
                        continue
                    seen.add(key)
                    model = (
                        base_id
                        if train_lang is None
                        else _format(
                            naming.hf_model,
                            benchmark=bench,
                            train_language=train_lang,
                            series=series.series,
                            base=base_slug,
                            replicate=rep,
                        )
                    )
                    cells.append(
                        {
                            "series": series.series,
                            "benchmark": bench,
                            "eval_language": lang,
                            "eval_variant": variant,
                            "condition": condition,
                            "tier": int(cond.get("tier", 0)),
                            "base_role": base_role,
                            "base_model": base_id,
                            "train_language": train_lang,
                            "replicate": rep,
                            "eval_dataset": eval_dataset,
                            "train_dataset": (
                                None
                                if train_lang is None
                                else _format(
                                    naming.hf_dataset, benchmark=bench, language=train_lang
                                )
                            ),
                            "model": model,
                            "wandb_run": _format(
                                naming.wandb_run,
                                stage="eval",
                                benchmark=bench,
                                language=lang,
                                condition=condition,
                                base=base_slug,
                                replicate=rep,
                            ),
                        }
                    )
    return cells


def unique_finetunes(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Finetune jobs implied by the cells (one per benchmark, base, train language, replicate)."""
    seen: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for cell in cells:
        if cell["train_language"] is None:
            continue
        key = (cell["benchmark"], cell["base_role"], cell["train_language"], cell["replicate"])
        seen.setdefault(
            key,
            {
                "benchmark": cell["benchmark"],
                "base_role": cell["base_role"],
                "base_model": cell["base_model"],
                "train_language": cell["train_language"],
                "replicate": cell["replicate"],
                "train_dataset": cell["train_dataset"],
                "model": cell["model"],
                "tier": cell["tier"],
            },
        )
    return list(seen.values())


def unique_datasets(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Datasets implied by the cells: translated train/test repos and derived eval variants."""
    seen: dict[str, dict[str, Any]] = {}
    for cell in cells:
        for key, lang, variant in (
            ("eval_dataset", cell["eval_language"], cell["eval_variant"]),
            ("train_dataset", cell["train_language"], None),
        ):
            repo = cell[key]
            if repo and repo not in seen:
                seen[repo] = {
                    "benchmark": cell["benchmark"],
                    "language": lang,
                    "variant": variant,
                    "repo": repo,
                }
    return list(seen.values())
