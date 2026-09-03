"""No-network tests for the config registries and series expansion."""

import pytest
from omegaconf import OmegaConf

from src.utils.registry import (
    expand_matrix,
    list_entries,
    load_entry,
    load_series,
    unique_datasets,
    unique_finetunes,
)


def test_every_language_entry_is_valid():
    codes = list_entries("language")
    assert {"en", "id", "jv", "su", "min", "ace"} <= set(codes)
    for code in codes:
        cfg = load_entry("language", code)
        assert cfg.code == code
        assert cfg.flores_code.endswith(f"_{cfg.script}")


def test_every_benchmark_entry_is_valid():
    for name in list_entries("benchmark"):
        cfg = load_entry("benchmark", name)
        assert cfg.name == name
        assert cfg.scorer_language_agnostic, f"{name}: scorer must not favor a language"
        assert "{language}" in cfg.translate.blueprint
        assert cfg.train_source.min_rows >= 1000


def test_missing_entry_raises():
    with pytest.raises(FileNotFoundError):
        load_entry("language", "zz")


def test_series_expands_to_expected_matrix():
    series = load_series("s01_language_medium")
    cells = expand_matrix(series)
    n_lang, n_bench, n_seed = len(series.languages), len(series.benchmarks), len(series.seeds)
    # base (1 per language) + native (seeds per language) + english_anchor (seeds per
    # non-English language; in English it coincides with native and is not repeated)
    # + indonesian_anchor (seeds per regional language) + pooled (seeds per language)
    # + regression (every non-English finetune scored once on the English test split)
    n_regional = n_lang - 2
    expected = n_bench * (
        n_lang * (1 + n_seed)
        + (n_lang - 1) * n_seed
        + n_regional * n_seed
        + n_lang * n_seed
        + (n_lang - 1) * n_seed
    )
    assert len(cells) == expected
    assert len(
        {(c["benchmark"], c["eval_language"], c["train_language"], c["seed"]) for c in cells}
    ) == len(cells)
    assert all(c["seed"] == 0 for c in cells if c["condition"] == "base")
    assert all(
        c["train_language"] == c["eval_language"] for c in cells if c["condition"] == "native"
    )
    assert all(c["train_language"] == "en" for c in cells if c["condition"] == "english_anchor")
    regression = [c for c in cells if c["condition"] == "regression"]
    assert regression and all(c["eval_language"] == "en" for c in regression)
    assert all(c["train_language"] != "en" for c in regression)
    # one finetune per benchmark x train language x seed (plus the pooled model), shared
    # across eval languages
    assert len(unique_finetunes(cells)) == n_bench * (n_lang + 1) * n_seed
    assert len(unique_datasets(cells)) == n_bench * (n_lang + 1)
    pooled = [c for c in cells if c["condition"] == "pooled"]
    assert pooled and all(c["train_language"] == "all" for c in pooled)
    anchor = [c for c in cells if c["condition"] == "indonesian_anchor"]
    assert anchor and all(c["train_language"] == "id" for c in anchor)
    assert all(c["eval_language"] not in ("en", "id") for c in anchor)


def test_transfer_condition_is_off_diagonal():
    series = load_series("s01_language_medium")
    series = OmegaConf.merge(series, {"active_conditions": ["transfer"]})
    cells = expand_matrix(series)
    assert cells and all(c["train_language"] != c["eval_language"] for c in cells)
    n_lang, n_bench, n_seed = len(series.languages), len(series.benchmarks), len(series.seeds)
    assert len(cells) == n_bench * n_lang * (n_lang - 1) * n_seed
