"""No-network tests for the AutoScientist stage with a stub client."""

from types import SimpleNamespace

import pytest

from pipeline.training.autoscientist_finetune import (
    build_create_kwargs,
    check_dataset,
    check_match,
    count_tokens,
    idempotency_key_for,
    needs_pinning,
    recommend_for_series,
    run_name_for,
    validate_base,
    write_series_hyperparams,
)


def _auto():
    return {
        "data_format": "instruction",
        "max_iterations": 1,
        "target_win_rate": 1.0,
        "augmentation_domain_rows": 0,
        "augmentation_general_rows": 0,
    }


def _hyperparams():
    return {"lora_r": 8, "lora_alpha": 16, "training_type": "lora", "train_on_inputs": True}


class _Datasets:
    def __init__(self, row_count=1000):
        self.row_count = row_count

    def get_status(self, dataset_id):
        return SimpleNamespace(row_count=self.row_count, dataset_id=dataset_id)

    def get(self, dataset_id):
        return SimpleNamespace(processing_mode="raw")


class _Autoscientist:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.calls = []

    def recommend_hyperparams(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(model=kwargs.get("model"), hyperparams=dict(self.hyperparams))


class _Client:
    def __init__(self, hyperparams, row_count=1000):
        self.datasets = _Datasets(row_count)
        self.autoscientist = _Autoscientist(hyperparams)


def test_validate_base_records_capabilities():
    models = [
        {"id": "a", "methods": ["sft"], "training_types": ["lora"], "context_length": 4096},
        {"id": "b", "methods": ["sft", "dpo"], "training_types": ["lora", "full"], "context_length": 8192},
    ]
    assert validate_base(models, "b") == {
        "id": "b", "methods": ["sft", "dpo"],
        "training_types": ["lora", "full"], "context_length": 8192,
    }
    with pytest.raises(ValueError, match="available"):
        validate_base(models, "zzz")


def test_check_dataset_row_count_and_raw_note():
    record = check_dataset(_Client({}), "ds", 1000)
    assert record["row_count"] == 1000
    with pytest.raises(ValueError, match="expected 1000"):
        check_dataset(_Client({}, row_count=999), "ds", 1000)


def test_build_create_kwargs_passes_constants_verbatim():
    kwargs = build_create_kwargs(_auto(), _hyperparams(), "ds", "m", {"prompt": "p"}, "k")
    assert kwargs["hyperparams"] == _hyperparams()
    assert kwargs["max_iterations"] == 1 and kwargs["target_win_rate"] == 1.0
    assert kwargs["data_format"] == "instruction"
    assert kwargs["idempotency_key"] == "k"


def test_idempotency_key_unique_per_replicate():
    keys = {idempotency_key_for("s", "b", "l", "base", r) for r in (1, 2, 3)}
    assert len(keys) == 3
    assert idempotency_key_for("s", "b", "l", "base", 1) == "s-b-l-base-r1"


def test_run_name_from_train_identity():
    assert run_name_for("gsm8k", "jv", "primary", 2) == "finetune-gsm8k-jv-primary-r2"


def test_needs_pinning_and_recommend_flow():
    assert needs_pinning(None)
    assert not needs_pinning(_hyperparams())
    client = _Client(_hyperparams())
    result = recommend_for_series(client, "ds-en", "m", 0, 0)
    assert result["hyperparams"] == _hyperparams()
    assert result["model"] == "m"
    assert client.autoscientist.calls[0]["dataset_id"] == "ds-en"


def test_write_series_hyperparams_preserves_other_lines(tmp_path):
    path = tmp_path / "series.yaml"
    path.write_text(
        "series: s01\n"
        "  hyperparams: null         # filled once\n"
        "                            # second comment\n"
        "  hyperparams_source: null  # dataset id and date\n"
        "other: 1\n",
        encoding="utf-8",
    )
    write_series_hyperparams(path, {"lora_r": 8}, "ds on 2026-11-22")
    text = path.read_text(encoding="utf-8")
    assert "hyperparams: null" not in text
    assert "lora_r: 8" in text
    assert "ds on 2026-11-22" in text
    assert "series: s01\n" in text and "other: 1\n" in text


def test_check_match_pass_and_mismatch():
    run = SimpleNamespace(best_hyperparams=dict(_hyperparams()), iterations_completed=1.0)
    check_match(_hyperparams(), 1, run)
    drifted = SimpleNamespace(
        best_hyperparams={**_hyperparams(), "lora_r": 16}, iterations_completed=1.0
    )
    with pytest.raises(ValueError, match="lora_r"):
        check_match(_hyperparams(), 1, drifted)
    short = SimpleNamespace(best_hyperparams=dict(_hyperparams()), iterations_completed=0.0)
    with pytest.raises(ValueError, match="iterations_completed"):
        check_match(_hyperparams(), 1, short)


def test_count_tokens_uses_injected_encode():
    assert count_tokens(["ab cd", "e"], lambda text: text.split()) == 3
