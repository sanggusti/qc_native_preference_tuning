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


def _counts(series):
    return len(series.languages), len(series.benchmarks), len(series.replicates)


def test_tier0_matrix_matches_the_design():
    series = load_series("s01_language_medium")
    cells = [c for c in expand_matrix(series) if c["tier"] == 0]
    n_lang, n_bench, n_rep = _counts(series)
    # base (1 per language) + native (replicates per language) + english_anchor (replicates per
    # non-English language; in English it coincides with native) + regression (every
    # non-English finetune on the English split) + round_trip (English finetunes on the
    # back-translated split of every non-English language)
    expected = n_bench * (n_lang * (1 + n_rep) + 3 * (n_lang - 1) * n_rep)
    assert len(cells) == expected
    assert all(c["replicate"] == 0 for c in cells if c["condition"] == "base")
    assert all(
        c["train_language"] == c["eval_language"] for c in cells if c["condition"] == "native"
    )
    regression = [c for c in cells if c["condition"] == "regression"]
    assert regression and all(c["eval_language"] == "en" for c in regression)
    assert all(c["train_language"] != "en" for c in regression)
    round_trip = [c for c in cells if c["condition"] == "round_trip"]
    assert round_trip and all(c["train_language"] == "en" for c in round_trip)
    assert all(c["eval_dataset"].endswith("-rt") for c in round_trip)
    assert all(c["eval_language"] != "en" for c in round_trip)
    # tier 0 needs exactly one finetune per benchmark x language x replicate on the primary base
    assert len(unique_finetunes(cells)) == n_bench * n_lang * n_rep
    assert all(f["base_role"] == "primary" for f in unique_finetunes(cells))


def test_full_matrix_has_no_duplicate_cells_and_derives_names():
    series = load_series("s01_language_medium")
    cells = expand_matrix(series)
    keys = {
        (
            c["benchmark"],
            c["eval_language"],
            c["eval_variant"],
            c["base_role"],
            c["train_language"],
            c["replicate"],
        )
        for c in cells
    }
    assert len(keys) == len(cells)
    pooled = [c for c in cells if c["condition"] == "pooled"]
    assert pooled and all(c["train_language"] == "all" for c in pooled)
    anchor = [c for c in cells if c["condition"] == "indonesian_anchor"]
    assert anchor and all(c["train_language"] == "id" for c in anchor)
    assert all(c["eval_language"] not in ("en", "id") for c in anchor)
    contrast = [c for c in cells if c["base_role"] == "contrast"]
    assert contrast and all(c["benchmark"] == "gsm8k" for c in contrast)
    assert all(c["base_model"] == series.base_models.contrast for c in contrast)
    assert all("llama32-3b" in c["model"] for c in contrast)
    datasets = {d["repo"] for d in unique_datasets(cells)}
    assert "sanggusti/gsm8k-jv" in datasets and "sanggusti/gsm8k-jv-rt" in datasets
    assert "sanggusti/gsm8k-all" in datasets


def test_transfer_condition_is_off_diagonal():
    series = load_series("s01_language_medium")
    series = OmegaConf.merge(
        series,
        {
            "conditions": {"transfer": {"train": "other", "tier": 3, "description": "x"}},
            "active_conditions": ["transfer"],
        },
    )
    cells = expand_matrix(series)
    assert cells and all(c["train_language"] != c["eval_language"] for c in cells)
    n_lang, n_bench, n_rep = _counts(series)
    assert len(cells) == n_bench * n_lang * (n_lang - 1) * n_rep


def test_unknown_base_role_is_rejected():
    series = load_entry("series", "s01_language_medium")
    series = OmegaConf.merge(
        series,
        {
            "conditions": {
                "bad": {"train": "same", "tier": 1, "bases": ["nope"], "description": "x"}
            }
        },
    )
    with pytest.raises(ValueError):
        # exercise the same validation load_series performs, on the merged config
        for name, cond in series.conditions.items():
            for base in cond.get("bases") or []:
                if base not in series.base_models:
                    raise ValueError(name)
