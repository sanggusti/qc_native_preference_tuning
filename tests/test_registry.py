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
    validate_series,
)


def test_every_language_entry_is_valid():
    codes = list_entries("language")
    assert {"en", "id", "jv", "su", "min", "ace"} <= set(codes)
    for code in codes:
        cfg = load_entry("language", code)
        assert cfg.code == code
        if cfg.flores_code is not None:  # null = not in FLORES-200, calibrate on NusaX
            assert cfg.flores_code.endswith(f"_{cfg.script}")
        assert cfg.register_instruction


def test_every_benchmark_entry_is_valid():
    for name in list_entries("benchmark"):
        cfg = load_entry("benchmark", name)
        assert cfg.name == name
        assert cfg.scorer_language_agnostic, f"{name}: scorer must not favor a language"
        assert "{language}" in cfg.translate.blueprint
        assert "{register_instruction}" in cfg.translate.blueprint
        assert cfg.train_source.min_rows >= 1000


def test_missing_entry_raises():
    with pytest.raises(FileNotFoundError):
        load_entry("language", "zz")


def _tier0_axes(series):
    tiers = series.get("tiers") or {}
    langs = [c for c in series.languages if int((tiers.get("languages") or {}).get(c, 0)) == 0]
    benches = [
        b for b in series.benchmarks if int((tiers.get("benchmarks") or {}).get(b, 0)) == 0
    ]
    return langs, benches, len(series.replicates)


def test_tier0_matrix_matches_the_design():
    series = load_series("s01_language_medium")
    cells = [c for c in expand_matrix(series) if c["tier"] == 0]
    langs, benches, n_rep = _tier0_axes(series)
    # the minimum publishable unit is gsm8k on the languages expected to clear the floor rule
    assert benches == ["gsm8k"] and "ace" not in langs
    n_lang = len(langs)
    # base + native on every language, the same two conditions on the digit re-instantiated
    # split, english_anchor + regression + round_trip on every non-English language (in
    # English the anchor coincides with native), and the Indonesian pivot on the regional
    # languages
    expected = 2 * n_lang * (1 + n_rep) + 3 * (n_lang - 1) * n_rep + (n_lang - 2) * n_rep
    assert len(cells) == expected
    assert {c["benchmark"] for c in cells} == {"gsm8k"}
    assert "ace" not in {c["eval_language"] for c in cells}
    assert "ace" not in {c["train_language"] for c in cells}
    pro1 = [c for c in cells if c["eval_variant"] == "pro1"]
    assert pro1 and {c["condition"] for c in pro1} == {"base_pro1", "native_pro1"}
    assert all(c["replicate"] == 0 and c["backend"] is None for c in cells if c["condition"] == "base")
    assert all(
        c["train_language"] == c["eval_language"] for c in cells if c["condition"] == "native"
    )
    pivot = [c for c in cells if c["condition"] == "indonesian_anchor"]
    assert pivot and all(c["train_language"] == "id" for c in pivot)
    assert {c["eval_language"] for c in pivot} == set(langs) - {"en", "id"}
    regression = [c for c in cells if c["condition"] == "regression"]
    assert regression and all(c["eval_language"] == "en" for c in regression)
    assert all(c["train_language"] != "en" for c in regression)
    round_trip = [c for c in cells if c["condition"] == "round_trip"]
    assert round_trip and all(c["train_language"] == "en" for c in round_trip)
    assert all(c["eval_dataset"].endswith("-rt") for c in round_trip)
    assert all(c["eval_language"] != "en" for c in round_trip)
    # tier 0 needs exactly one finetune per language x replicate on the primary base; the
    # pivot arm reuses the Indonesian native model
    finetunes = unique_finetunes(cells)
    assert len(finetunes) == n_lang * n_rep
    assert all(f["base_role"] == "primary" and f["backend"] == "autoscientist" for f in finetunes)


def test_tiers_compose_over_condition_benchmark_and_language():
    series = load_series("s01_language_medium")
    cells = expand_matrix(series)
    assert all(c["tier"] >= 3 for c in cells if c["benchmark"] == "medqa")
    assert all(
        c["tier"] >= 2 for c in cells if "ace" in (c["eval_language"], c["train_language"])
    )
    # a tier 0 condition on a parked language is lifted, not dropped
    assert any(c["condition"] == "native" and c["eval_language"] == "ace" for c in cells)


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
            c["backend"],
            c["replicate"],
        )
        for c in cells
    }
    assert len(keys) == len(cells)
    pooled = [c for c in cells if c["condition"] == "pooled"]
    assert pooled and all(c["train_language"] == "all" for c in pooled)
    contrast = [c for c in cells if c["base_role"] == "contrast"]
    assert contrast and all(c["benchmark"] == "gsm8k" for c in contrast)
    assert all(c["base_model"] == series.base_models.contrast for c in contrast)
    assert all("llama32-3b" in c["model"] for c in contrast)
    sft = [c for c in cells if c["condition"] == "sft_check"]
    assert sft and all(c["backend"] == "sft" and c["tier"] == 1 for c in sft)
    assert {c["eval_language"] for c in sft} == {"en", "id"}
    assert all(c["model"].endswith(f"-sft-r{c['replicate']}") for c in sft)
    managed = [c for c in cells if c["backend"] == "autoscientist"]
    assert managed and not any("-sft-" in c["model"] for c in managed)
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
    n_lang, n_bench, n_rep = len(series.languages), len(series.benchmarks), len(series.replicates)
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
    with pytest.raises(ValueError, match="unknown base model role"):
        validate_series(series)


def test_backend_without_slug_is_rejected():
    series = load_entry("series", "s01_language_medium")
    series = OmegaConf.merge(
        series,
        {
            "conditions": {
                "bad": {"train": "same", "tier": 1, "backend": "nope", "description": "x"}
            }
        },
    )
    with pytest.raises(ValueError, match="backend_slugs"):
        validate_series(series)


def test_tier_for_unknown_language_is_rejected():
    series = load_entry("series", "s01_language_medium")
    series = OmegaConf.merge(series, {"tiers": {"languages": {"zz": 1}}})
    with pytest.raises(ValueError, match="language 'zz'"):
        validate_series(series)
