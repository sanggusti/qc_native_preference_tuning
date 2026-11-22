"""No-network tests for the series registry using the real configs."""

import pytest

from src.utils.registry import (
    EvalCell,
    TrainCell,
    expand_matrix,
    load_benchmark,
    load_language,
    load_series,
    unique_datasets,
)

SERIES = "s01_language_medium"

LANGUAGE_CODES = ["en", "id", "jv", "su", "min", "ace", "ban", "bjn", "bug"]

LANGUAGE_KEYS = {
    "code", "name", "iso_639_3", "flores_code", "script",
    "register", "register_instruction", "exposure_note",
}

BENCHMARK_KEYS = {
    "name", "source_dataset", "source_config", "source_split_train",
    "source_split_test", "record_fields", "preserved_fields", "scorer",
    "scorer_language_agnostic", "instruction_template_key", "blueprint",
}


def test_load_language_entries():
    """Every registry language loads with the exact schema keys."""
    expected = {
        "en": ("English", "eng", "neutral"),
        "id": ("Indonesian", "ind", "formal"),
        "jv": ("Javanese", "jav", "ngoko"),
        "su": ("Sundanese", "sun", "loma"),
        "min": ("Minangkabau", "min", "standard"),
        "ace": ("Acehnese", "ace", "standard"),
        "ban": ("Balinese", "ban", "standard"),
        "bjn": ("Banjarese", "bjn", "standard"),
        "bug": ("Buginese", "bug", "standard"),
    }
    for code, want in expected.items():
        entry = load_language(code)
        assert set(entry) == LANGUAGE_KEYS
        assert entry["code"] == code
        assert (entry["name"], entry["iso_639_3"], entry["register"]) == want


def test_load_benchmark_entries():
    """Benchmark entries carry source, scorer, and blueprint placeholders."""
    gsm8k = load_benchmark("gsm8k")
    assert set(gsm8k) == BENCHMARK_KEYS
    assert gsm8k["source_dataset"] == "openai/gsm8k"
    assert gsm8k["source_config"] == "main"
    assert gsm8k["scorer"] == "numeric"
    assert gsm8k["scorer_language_agnostic"] is True
    medqa = load_benchmark("medqa")
    assert set(medqa) == BENCHMARK_KEYS
    assert medqa["source_dataset"] == "GBaker/MedQA-USMLE-4-options"
    assert medqa["scorer"] == "choice"
    assert medqa["scorer_language_agnostic"] is True
    for entry in (gsm8k, medqa):
        assert "{language}" in entry["blueprint"]
        assert "{register_instruction}" in entry["blueprint"]


def test_load_series_entry():
    """Series entry carries languages, benchmarks, conditions, and replicates."""
    series = load_series(SERIES)
    assert series["series"] == SERIES
    assert series["languages"] == LANGUAGE_CODES
    assert series["benchmarks"] == ["gsm8k", "medqa"]
    assert [c["name"] for c in series["conditions"]] == [
        "base", "native", "english_anchor", "indonesian_anchor", "pooled",
    ]
    assert series["replicates"] == 3
    assert series["sampling_seed"] == 47
    assert set(series["base_models"]) == {"primary", "contrast"}


def test_unknown_entries_raise_keyerror():
    """Unknown language, benchmark, and series names raise KeyError with options."""
    with pytest.raises(KeyError) as excinfo:
        load_language("xx")
    assert "jv" in str(excinfo.value)
    with pytest.raises(KeyError) as excinfo:
        load_benchmark("nope")
    assert "gsm8k" in str(excinfo.value)
    with pytest.raises(KeyError) as excinfo:
        load_series("nope")
    assert SERIES in str(excinfo.value)


def test_invalid_stage_raises_valueerror():
    """Unknown expand_matrix stage raises ValueError."""
    with pytest.raises(ValueError, match="Unknown stage"):
        expand_matrix(SERIES, stage="datagen")


def test_finetune_cell_count_per_benchmark():
    """Finetune cells per benchmark per base role: (7 + 1 + 1 + 1) * 3 = 30."""
    for benchmark in ("gsm8k", "medqa"):
        cells = expand_matrix(SERIES, stage="finetune", benchmark=benchmark, base="primary")
        assert len(cells) == 30
        assert all(isinstance(c, TrainCell) for c in cells)
        assert {c.condition for c in cells} == {
            "native", "english_anchor", "indonesian_anchor", "pooled",
        }


def test_finetune_cell_count_total():
    """Both base roles double the per benchmark count: 30 * 2 benchmarks * 2 roles."""
    cells = expand_matrix(SERIES, stage="finetune")
    assert len(cells) == 30 * 2 * 2


def test_no_native_train_cells_for_anchor_media():
    """en/id have no native train cells; those media coincide with the anchors."""
    cells = expand_matrix(SERIES, stage="finetune", condition="native")
    assert {c.train_language for c in cells} == {"jv", "su", "min", "ace", "ban", "bjn", "bug"}


def test_eval_cell_count_per_benchmark_per_base():
    """Eval cells per benchmark per base role: 27 + 54 + 27 + 9 = 117."""
    for benchmark in ("gsm8k", "medqa"):
        cells = expand_matrix(SERIES, stage="eval", benchmark=benchmark, base="primary")
        assert len(cells) == 117
        assert all(isinstance(c, EvalCell) for c in cells)


def test_base_condition_replicate_one_only():
    """Base eval cells exist in every language but only for replicate 1."""
    cells = expand_matrix(SERIES, stage="eval", condition="base", base="primary")
    assert {c.replicate for c in cells} == {1}
    assert {c.train_language for c in cells} == {None}
    assert len(cells) == 2 * 9


def test_native_eval_is_diagonal():
    """Native eval cells evaluate each model in its own train language."""
    cells = expand_matrix(
        SERIES, stage="eval", benchmark="gsm8k", base="primary", condition="native",
    )
    assert len(cells) == 9 * 3
    for cell in cells:
        assert cell.eval_language == cell.train_language


def test_degenerate_reroute_reuses_anchor_checkpoints():
    """Native en/id eval cells reference the anchor checkpoints of the same replicate."""
    en_cells = expand_matrix(
        SERIES, stage="eval", benchmark="gsm8k", base="primary",
        condition="native", language="en",
    )
    anchor = expand_matrix(
        SERIES, stage="finetune", benchmark="gsm8k", base="primary",
        condition="english_anchor", language="en",
    )
    assert len(en_cells) == 3
    for cell in en_cells:
        want = next(a.hub_model_id for a in anchor if a.replicate == cell.replicate)
        assert cell.model_ref == f"openai-api/{want}"
    id_cells = expand_matrix(
        SERIES, stage="eval", benchmark="gsm8k", base="primary",
        condition="native", language="id",
    )
    id_anchor = expand_matrix(
        SERIES, stage="finetune", benchmark="gsm8k", base="primary",
        condition="indonesian_anchor", language="id",
    )
    assert len(id_cells) == 3
    for cell in id_cells:
        want = next(a.hub_model_id for a in id_anchor if a.replicate == cell.replicate)
        assert cell.model_ref == f"openai-api/{want}"


def test_tier_filter():
    """Tier filter selects conditions by tier."""
    cells = expand_matrix(SERIES, stage="eval", tier=2, base="primary")
    assert cells
    assert {c.condition for c in cells} == {"indonesian_anchor"}
    cells = expand_matrix(SERIES, stage="finetune", tier=0, base="primary")
    assert {c.condition for c in cells} == {"native", "english_anchor"}


def test_benchmark_filter():
    """Benchmark filter selects a single benchmark."""
    cells = expand_matrix(SERIES, stage="eval", benchmark="medqa", base="primary")
    assert len(cells) == 117
    assert {c.benchmark for c in cells} == {"medqa"}


def test_language_filter():
    """Language filter matches eval_language on eval and train_language on finetune."""
    evals = expand_matrix(SERIES, stage="eval", language="jv", base="primary")
    assert evals
    assert {c.eval_language for c in evals} == {"jv"}
    assert len(evals) == 13 * 2
    trains = expand_matrix(
        SERIES, stage="finetune", language="jv", benchmark="gsm8k", base="primary",
    )
    assert {c.train_language for c in trains} == {"jv"}
    assert len(trains) == 3


def test_condition_filter():
    """Condition filter selects a single condition."""
    cells = expand_matrix(SERIES, stage="eval", condition="pooled", base="primary")
    assert cells
    assert {c.condition for c in cells} == {"pooled"}
    assert {c.train_language for c in cells} == {"all"}


def test_base_filter():
    """Base filter matches base_role."""
    cells = expand_matrix(SERIES, stage="eval", base="contrast")
    assert cells
    assert {c.base_role for c in cells} == {"contrast"}


def test_unpinned_base_carried_through():
    """Unpinned base_model None is allowed and carried through on cells."""
    trains = expand_matrix(SERIES, stage="finetune", benchmark="gsm8k", base="primary")
    assert {c.base_model for c in trains} == {None}
    evals = expand_matrix(SERIES, stage="eval", benchmark="gsm8k", base="primary")
    assert {c.base_model for c in evals} == {None}


def test_unique_datasets():
    """Unique datasets: 9 languages + 1 pooled per benchmark = 20 ids, sorted."""
    ids = unique_datasets(SERIES)
    assert len(ids) == 20
    assert ids == sorted(ids)
    assert "sanggusti/gsm8k-jv" in ids
    assert "sanggusti/medqa-id" in ids
    assert "sanggusti/gsm8k-all" in ids
    assert "sanggusti/medqa-all" in ids


def test_hub_model_id_format():
    """Train cells render dataset and hub model ids from the naming templates."""
    cells = expand_matrix(
        SERIES, stage="finetune", benchmark="gsm8k", base="primary",
        language="jv", condition="native",
    )
    by_rep = {c.replicate: c for c in cells}
    assert by_rep[1].train_dataset_id == "sanggusti/gsm8k-jv"
    assert by_rep[1].hub_model_id == "sanggusti/gsm8k-jv-native-primary-r1"
    pooled = expand_matrix(
        SERIES, stage="finetune", benchmark="medqa", base="contrast",
        condition="pooled",
    )
    assert pooled[0].train_dataset_id == "sanggusti/medqa-all"
    assert pooled[0].hub_model_id == "sanggusti/medqa-all-pooled-contrast-r1"


def test_log_dir_format():
    """Eval cells render log dirs from the log template."""
    cells = expand_matrix(
        SERIES, stage="eval", benchmark="gsm8k", base="primary", language="jv",
        condition="native",
    )
    match = [c for c in cells if c.train_language == "jv" and c.replicate == 1]
    assert len(match) == 1
    assert match[0].log_dir == "logs/s01_language_medium/gsm8k/jv/native/primary-r1"
    assert match[0].test_dataset_id == "sanggusti/gsm8k-jv"
    assert match[0].model_ref == "openai-api/sanggusti/gsm8k-jv-native-primary-r1"


def test_base_model_ref_format():
    """Base eval cells reference the pinned base model or the role while unpinned."""
    cells = expand_matrix(
        SERIES, stage="eval", benchmark="gsm8k", base="primary", language="en",
        condition="base",
    )
    assert len(cells) == 1
    assert cells[0].model_ref == "openai-api/primary"
