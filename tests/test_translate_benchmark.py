"""No-network tests for translate_benchmark with a stub Adaption client."""

import json

import pytest

from pipeline.datagenerator.translate_benchmark import (
    blueprint_hash,
    build_blueprint,
    build_run_kwargs,
    canonical_row,
    dataset_card,
    expansion_probe_kwargs,
    map_adapted_row,
    probe_language_expansion,
    select_train_indices,
    source_id_for,
    split_instruction_row,
    training_pair,
    translator_version,
    verify_instruction,
    verify_row_survival,
    wait_until_ready,
)
from src.utils.registry import load_entry


class _Status:
    def __init__(self, row_count=None, status="ready", message=""):
        self.row_count = row_count
        self.status = status
        self.error_data = type("E", (), {"message": message})()


class _Datasets:
    def __init__(self, statuses=None, reject=()):
        self.statuses = list(statuses or [_Status(row_count=10)])
        self.reject = set(reject)
        self.run_calls = []

    def run(self, dataset_id, **kwargs):
        self.run_calls.append((dataset_id, kwargs))
        expansion = kwargs.get("language_expansion")
        if expansion and expansion["languages"][0] in self.reject:
            raise RuntimeError("400 unknown language")
        return {"ok": True}

    def get_status(self, dataset_id):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


class _Client:
    def __init__(self, datasets):
        self.datasets = datasets


def _translation():
    return {
        "recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False},
        "length": None,
        "hallucination_mitigation": False,
        "safety_categories": [],
    }


def test_build_blueprint_substitutes_language_and_register():
    text = build_blueprint(load_entry("benchmark", "gsm8k"), load_entry("language", "jv"))
    assert "Javanese" in text
    assert "{language}" not in text and "{register_instruction}" not in text
    assert len(blueprint_hash(text)) == 16


def test_select_train_indices_deterministic_and_shared():
    first = select_train_indices(7473, 1000, 47)
    assert first == select_train_indices(7473, 1000, 47)
    assert first == sorted(first) and len(set(first)) == 1000
    with pytest.raises(ValueError, match="exceeds source size"):
        select_train_indices(10, 1000, 47)


def test_source_id_prefers_id_column():
    assert source_id_for("gsm8k", "train", {"id": "abc"}, 3) == "abc"
    assert source_id_for("gsm8k", "train", {"question": "q"}, 3) == "gsm8k-train-00003"


def test_build_run_kwargs_pins_series_settings():
    kwargs = build_run_kwargs(_translation(), "bp", {"prompt": "q", "completion": "a"}, None, "k")
    assert kwargs["recipe_specification"] == {
        "recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False}
    }
    assert kwargs["brand_controls"] == {
        "blueprint": "bp", "hallucination_mitigation": False, "safety_categories": []
    }
    assert kwargs["job_specification"] == {"idempotency_key": "k"}
    assert kwargs["training_type"] == "instruction_dataset"
    assert "idempotency_key" not in kwargs


def test_build_run_kwargs_sets_length_and_max_rows_when_configured():
    translation = _translation()
    translation["length"] = "concise"
    kwargs = build_run_kwargs(translation, "bp", {}, 20, "k")
    assert kwargs["brand_controls"]["length"] == "concise"
    assert kwargs["job_specification"]["max_rows"] == 20.0


def test_expansion_probe_kwargs_shape():
    assert expansion_probe_kwargs("jv") == {
        "type": "translate", "languages": ["jv"], "sample_rate": 1.0
    }


def test_probe_records_accepted_and_rejected_without_calling_for_639_3_only():
    client = _Client(_Datasets(reject=("xx",)))
    results = probe_language_expansion(client, "ds", {"prompt": "q"}, ["en", "jv", "min", "ace"])
    assert results["en"]["accepted"] and results["jv"]["accepted"]
    assert results["min"] == {"accepted": False, "detail": "no ISO 639-1 code"}
    assert results["ace"]["accepted"] is False
    probed = [call[1]["language_expansion"]["languages"][0] for call in client.datasets.run_calls]
    assert probed == ["en", "jv"]


def test_probe_records_api_rejection():
    client = _Client(_Datasets())
    client.datasets.run = lambda dataset_id, **kwargs: (_ for _ in ()).throw(RuntimeError("400 nope"))
    results = probe_language_expansion(client, "ds", {}, ["en"])
    assert results["en"] == {"accepted": False, "detail": "400 nope"}


def test_wait_until_ready_polls_row_count_and_fails_loudly():
    client = _Client(_Datasets([_Status(), _Status(row_count=5)]))
    wait_until_ready(client, "ds", sleep_fn=lambda s: None)
    failing = _Client(_Datasets([_Status(status="failed", message="boom")]))
    with pytest.raises(RuntimeError, match="boom"):
        wait_until_ready(failing, "ds", sleep_fn=lambda s: None)
    hanging = _Client(_Datasets([_Status()]))
    with pytest.raises(TimeoutError):
        wait_until_ready(hanging, "ds", timeout=0, sleep_fn=lambda s: None)


def test_map_adapted_row_maps_enhanced_and_raises_otherwise():
    assert map_adapted_row({"enhanced_prompt": "p", "enhanced_completion": "c"}) == ("p", "c")
    with pytest.raises(KeyError, match="enhanced_prompt"):
        map_adapted_row({"question": "p", "answer": "c"})


def test_verify_row_survival_equal_only():
    verify_row_survival(10, 10)
    with pytest.raises(ValueError, match="re-run"):
        verify_row_survival(9, 10)


def test_verify_instruction_requires_markers():
    verify_instruction("Do x.\nANSWER: 1", "numeric_match")
    verify_instruction("Q {question} C {choices}\nANSWER: B", "choice")
    with pytest.raises(ValueError, match="ANSWER"):
        verify_instruction("Do x.", "numeric_match")
    with pytest.raises(ValueError, match="question"):
        verify_instruction("Q?\nANSWER: B", "choice")


def test_split_instruction_row_extracts_first_row():
    rows = [
        {"enhanced_prompt": "Do x.\nANSWER: 1", "enhanced_completion": "Do x.\nANSWER: 1"},
        {"enhanced_prompt": "a", "enhanced_completion": "b #### 1"},
    ]
    instruction, rest = split_instruction_row(rows, "numeric_match")
    assert instruction.startswith("Do x.")
    assert len(rest) == 1
    with pytest.raises(ValueError, match="empty download"):
        split_instruction_row([], "numeric_match")


def test_canonical_row_numeric_and_choice():
    numeric = canonical_row(
        {}, "pitakon", "jawaban #### 7", "T", "TE", "q_en", "a_en",
        "jv", "ngoko", "hash", "tr", "numeric_match", "s-1",
    )
    assert numeric["target"] == "7" and numeric["id"] == "s-1"
    assert numeric["language"] == "jv" and "choices" not in numeric
    with pytest.raises(ValueError, match="####"):
        canonical_row(
            {}, "p", "no number here", "T", "TE", "q", "a",
            "jv", "ngoko", "h", "tr", "numeric_match", "s-1",
        )
    choice = canonical_row(
        {"choices": ["a", "b", "c", "d"], "answer": ["b"]},
        "pitakon", json.dumps(["a", "b", "c", "d"]), "T", "TE", "q_en", "a_en",
        "jv", "ngoko", "hash", "tr", "choice", "s-2",
    )
    assert choice["target"] == "B" and choice["choices"] == ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="unknown scorer"):
        canonical_row({}, "p", "c", "T", "TE", "q", "a", "jv", "n", "h", "tr", "essay", "s-3")


def test_training_pair_and_card_and_version():
    pair = training_pair("Do x.", "What?", "7")
    assert pair == {"prompt": "Do x.\n\nWhat?", "completion": "7"}
    card = dataset_card("gsm8k", "jv", {"train": 1000}, "ngoko", "bp", {"train": "d1"}, "MIT")
    assert "gsm8k-jv" in card and "pending" in card
    assert translator_version().startswith("adaption-datasets/")
