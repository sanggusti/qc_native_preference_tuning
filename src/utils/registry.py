"""Series registry: load configs and expand the train/eval cell matrix.

Pipelines stay generic; all experiment structure lives in YAML under configs/.
Schemas: configs/language/_template.yaml, configs/benchmark/_template.yaml,
configs/series/s01_language_medium.yaml.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import yaml

CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "configs"

LANGUAGE_DIR = CONFIG_ROOT / "language"
BENCHMARK_DIR = CONFIG_ROOT / "benchmark"
SERIES_DIR = CONFIG_ROOT / "series"

_TEMPLATE_NAME = "_template.yaml"


def _available(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.yaml") if p.name != _TEMPLATE_NAME)


def _load(directory: Path, kind: str, name: str) -> dict[str, Any]:
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"Unknown {kind} {name!r}. Available: {_available(directory)}")
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_language(code: str) -> dict[str, Any]:
    """Return the language entry for code, e.g. load_language("jv")."""
    return _load(LANGUAGE_DIR, "language", code)


def load_benchmark(name: str) -> dict[str, Any]:
    """Return the benchmark entry for name, e.g. load_benchmark("gsm8k")."""
    return _load(BENCHMARK_DIR, "benchmark", name)


def load_series(name: str) -> dict[str, Any]:
    """Return the series entry for name, e.g. load_series("s01_language_medium")."""
    return _load(SERIES_DIR, "series", name)


@dataclass(frozen=True)
class TrainCell:
    """One finetuning run. train_language is None only for base (never trained)."""

    series: str
    benchmark: str
    condition: str
    train_language: str | None
    base_role: str
    base_model: str | None
    replicate: int
    tier: int
    train_dataset_id: str
    hub_model_id: str


@dataclass(frozen=True)
class EvalCell:
    """One eval run. train_language is None for the base condition."""

    series: str
    benchmark: str
    eval_language: str
    condition: str
    train_language: str | None
    replicate: int
    tier: int
    test_dataset_id: str
    model_ref: str
    log_dir: str
    base_role: str
    base_model: str | None


def _train_languages(condition: dict[str, Any], series_languages: list[str]) -> list[str]:
    raw = condition["train_languages"]
    if isinstance(raw, str):
        if raw == "{languages}":
            return list(series_languages)
        return [raw]
    return [str(lang) for lang in raw]


def _base_slug(base_slugs: dict[str, str], base_model: str | None, base_role: str) -> str:
    if base_model is None:
        return base_role
    return base_slugs.get(base_model, base_role)


def _dataset_id(naming: dict[str, Any], benchmark: str, language: str) -> str:
    template = str(naming["dataset_template"])
    return template.format(benchmark=benchmark, language=language, suffix="")


def _hub_model_id(
    naming: dict[str, Any],
    benchmark: str,
    language: str,
    condition: str,
    slug: str,
    replicate: int,
) -> str:
    template = str(naming["model_template"])
    return template.format(
        benchmark=benchmark, language=language, condition=condition,
        base=slug, replicate=replicate,
    )


def _log_dir(
    naming: dict[str, Any],
    series: str,
    benchmark: str,
    eval_language: str,
    condition: str,
    slug: str,
    replicate: int,
) -> str:
    template = str(naming["log_template"])
    return template.format(
        series=series, benchmark=benchmark, eval_language=eval_language,
        condition=condition, base=slug, replicate=replicate,
    )


@overload
def expand_matrix(
    series: str,
    stage: Literal["eval"] = ...,
    tier: int | None = ...,
    benchmark: str | None = ...,
    language: str | None = ...,
    condition: str | None = ...,
    base: str | None = ...,
) -> list[EvalCell]:
    ...


@overload
def expand_matrix(
    series: str,
    stage: Literal["finetune"] = ...,
    tier: int | None = ...,
    benchmark: str | None = ...,
    language: str | None = ...,
    condition: str | None = ...,
    base: str | None = ...,
) -> list[TrainCell]:
    ...


def expand_matrix(
    series: str,
    stage: str = "eval",
    tier: int | None = None,
    benchmark: str | None = None,
    language: str | None = None,
    condition: str | None = None,
    base: str | None = None,
) -> list[EvalCell] | list[TrainCell]:
    """Expand a series into train or eval cells.

    Stage "finetune" returns one TrainCell per (benchmark, condition, train
    language, base role, replicate); the base condition trains nothing so it
    yields no cells. Stage "eval" returns one EvalCell per run: native models
    are evaluated in their own train language only (diagonal), anchor and
    pooled models in every series language, and the base condition (replicate
    1 only) in every series language. Degenerate (condition, eval language)
    pairs from the series degenerate_cells list keep their own label but
    reference the other condition's checkpoint of the same replicate.
    Filters: tier and condition match condition entries, benchmark matches the
    benchmark, language matches train_language (finetune) or eval_language
    (eval), base matches base_role. Unpinned base_model None is allowed and
    carried through on every cell.
    """
    if stage not in ("eval", "finetune"):
        raise ValueError(f"Unknown stage {stage!r}. Expected 'eval' or 'finetune'.")
    cfg = load_series(series)
    naming: dict[str, Any] = cfg["naming"]
    base_slugs: dict[str, str] = dict(naming.get("base_slugs") or {})
    provider = str(naming["model_provider"])
    benchmarks = [b for b in cfg["benchmarks"] if benchmark is None or b == benchmark]
    conditions = [
        c
        for c in cfg["conditions"]
        if (tier is None or c["tier"] == tier)
        and (condition is None or c["name"] == condition)
    ]
    roles = [
        (role, model)
        for role, model in cfg["base_models"].items()
        if base is None or role == base
    ]
    max_replicates = int(cfg["replicates"])
    if stage == "finetune":
        return _finetune_cells(
            cfg, naming, base_slugs, benchmarks, conditions, roles, max_replicates,
            language, series,
        )
    return _eval_cells(
        cfg, naming, base_slugs, provider, benchmarks, conditions, roles,
        max_replicates, language, series,
    )


def _finetune_cells(
    cfg: dict[str, Any],
    naming: dict[str, Any],
    base_slugs: dict[str, str],
    benchmarks: list[str],
    conditions: list[dict[str, Any]],
    roles: list[tuple[str, Any]],
    max_replicates: int,
    language: str | None,
    series: str,
) -> list[TrainCell]:
    cells: list[TrainCell] = []
    for bm in benchmarks:
        for cond in conditions:
            if not cond["finetune"]:
                continue
            for train_lang in _train_languages(cond, cfg["languages"]):
                if language is not None and train_lang != language:
                    continue
                for role, model in roles:
                    slug = _base_slug(base_slugs, model, role)
                    for rep in range(1, max_replicates + 1):
                        cells.append(
                            TrainCell(
                                series=series,
                                benchmark=bm,
                                condition=cond["name"],
                                train_language=train_lang,
                                base_role=role,
                                base_model=model,
                                replicate=rep,
                                tier=cond["tier"],
                                train_dataset_id=_dataset_id(naming, bm, train_lang),
                                hub_model_id=_hub_model_id(
                                    naming, bm, train_lang, cond["name"], slug, rep
                                ),
                            )
                        )
    return cells


def _train_index(cfg: dict[str, Any]) -> dict[tuple[str, str, str, str, int], str]:
    """Map (benchmark, condition, train language, role, replicate) to hub id."""
    naming: dict[str, Any] = cfg["naming"]
    base_slugs: dict[str, str] = dict(naming.get("base_slugs") or {})
    cells = _finetune_cells(
        cfg, naming, base_slugs, list(cfg["benchmarks"]),
        [c for c in cfg["conditions"] if c["finetune"]],
        list(cfg["base_models"].items()), int(cfg["replicates"]), None, cfg["series"],
    )
    return {
        (c.benchmark, c.condition, str(c.train_language), c.base_role, c.replicate):
        c.hub_model_id for c in cells
    }


def _eval_cells(
    cfg: dict[str, Any],
    naming: dict[str, Any],
    base_slugs: dict[str, str],
    provider: str,
    benchmarks: list[str],
    conditions: list[dict[str, Any]],
    roles: list[tuple[str, Any]],
    max_replicates: int,
    language: str | None,
    series: str,
) -> list[EvalCell]:
    cells: list[EvalCell] = []
    index = _train_index(cfg)
    cond_by_name = {c["name"]: c for c in cfg["conditions"]}
    reroutes = [
        (r["condition"], r["eval_language"], r["use_condition"])
        for r in cfg.get("degenerate_cells") or []
    ]
    eval_languages = [lang for lang in cfg["languages"] if language is None or lang == language]
    for bm in benchmarks:
        for cond in conditions:
            for role, model in roles:
                slug = _base_slug(base_slugs, model, role)
                if not cond["finetune"]:
                    for eval_lang in eval_languages:
                        cells.append(
                            EvalCell(
                                series=series, benchmark=bm, eval_language=eval_lang,
                                condition=cond["name"], train_language=None,
                                replicate=1, tier=cond["tier"],
                                test_dataset_id=_dataset_id(naming, bm, eval_lang),
                                model_ref=f"{provider}/{model if model is not None else role}",
                                log_dir=_log_dir(
                                    naming, series, bm, eval_lang, cond["name"], slug, 1,
                                ),
                                base_role=role, base_model=model,
                            )
                        )
                    continue
                trains = _train_languages(cond, cfg["languages"])
                for train_lang in trains:
                    for rep in range(1, max_replicates + 1):
                        hub_id = index[(bm, cond["name"], train_lang, role, rep)]
                        if cond["name"] == "native":
                            langs = [train_lang] if train_lang in eval_languages else []
                        else:
                            langs = eval_languages
                        for eval_lang in langs:
                            cells.append(
                                EvalCell(
                                    series=series, benchmark=bm, eval_language=eval_lang,
                                    condition=cond["name"], train_language=train_lang,
                                    replicate=rep, tier=cond["tier"],
                                    test_dataset_id=_dataset_id(naming, bm, eval_lang),
                                    model_ref=f"{provider}/{hub_id}",
                                    log_dir=_log_dir(
                                        naming, series, bm, eval_lang, cond["name"],
                                        slug, rep,
                                    ),
                                    base_role=role, base_model=model,
                                )
                            )
                for cond_name, eval_lang, use_cond in reroutes:
                    if cond_name != cond["name"]:
                        continue
                    if language is not None and eval_lang != language:
                        continue
                    use_trains = _train_languages(cond_by_name[use_cond], cfg["languages"])
                    for rep in range(1, max_replicates + 1):
                        hub_id = index[(bm, use_cond, use_trains[0], role, rep)]
                        cells.append(
                            EvalCell(
                                series=series, benchmark=bm, eval_language=eval_lang,
                                condition=cond["name"], train_language=eval_lang,
                                replicate=rep, tier=cond["tier"],
                                test_dataset_id=_dataset_id(naming, bm, eval_lang),
                                model_ref=f"{provider}/{hub_id}",
                                log_dir=_log_dir(
                                    naming, series, bm, eval_lang, cond["name"],
                                    slug, rep,
                                ),
                                base_role=role, base_model=model,
                            )
                        )
    return cells


def unique_datasets(series: str) -> list[str]:
    """Return sorted unique HF dataset ids for a series.

    Covers the canonical per benchmark-language dataset plus the pooled
    {benchmark}-all dataset per benchmark. Train and test splits share ids.
    """
    cfg = load_series(series)
    naming: dict[str, Any] = cfg["naming"]
    ids = {
        _dataset_id(naming, benchmark, language)
        for benchmark in cfg["benchmarks"]
        for language in cfg["languages"]
    }
    ids.update(_dataset_id(naming, benchmark, "all") for benchmark in cfg["benchmarks"])
    return sorted(ids)
