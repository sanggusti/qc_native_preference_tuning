"""No-network tests for the SFT backend: mapping, seeds, naming."""

import pytest

from pipeline.training.sft_finetune import (
    check_backend_slug,
    format_example,
    map_trainer_args,
    require_pinned_hyperparams,
    resolve_train_repo,
    run_name_for,
    select_seed,
)


def _hyperparams():
    return {
        "learning_rate": 0.0002,
        "n_epochs": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "batch_size": 4,
        "train_on_inputs": True,
    }


def test_select_seed_one_based_with_bounds_check():
    assert select_seed([11, 22, 33], 1) == 11
    assert select_seed([11, 22, 33], 3) == 33
    with pytest.raises(ValueError, match="out of range"):
        select_seed([11, 22, 33], 0)
    with pytest.raises(ValueError, match="out of range"):
        select_seed([11, 22, 33], 4)


def test_require_pinned_hyperparams_refuses_null_and_partial():
    with pytest.raises(ValueError, match="null"):
        require_pinned_hyperparams(None)
    partial = dict(_hyperparams())
    del partial["lora_alpha"]
    with pytest.raises(ValueError, match="lora_alpha"):
        require_pinned_hyperparams(partial)
    assert require_pinned_hyperparams(_hyperparams())["lora_r"] == 8


def test_map_trainer_args_copies_fields_and_applies_overrides():
    args = map_trainer_args(_hyperparams(), 22)
    assert args == {
        "learning_rate": 0.0002,
        "num_train_epochs": 3.0,
        "per_device_train_batch_size": 4,
        "lora_r": 8,
        "lora_alpha": 16,
        "train_on_inputs": True,
        "seed": 22,
        "max_rows": None,
    }
    smoke = map_trainer_args(_hyperparams(), 11, max_rows=50, epochs=1)
    assert smoke["num_train_epochs"] == 1.0
    assert smoke["max_rows"] == 50
    assert smoke["seed"] == 11


def test_format_example_joins_instruction_question_target():
    assert format_example("Do x.", "What?", "7") == "Do x.\n\nWhat?\n\n7"


def test_resolve_train_repo_uses_naming_template():
    assert (
        resolve_train_repo("sanggusti/{benchmark}-{language}", "gsm8k", "id")
        == "sanggusti/gsm8k-id"
    )


def test_run_name_marks_sft_backend():
    assert run_name_for("gsm8k", "en", "primary", 1) == "finetune-gsm8k-en-primary-r1-sft"


def test_check_backend_slug_warns_only(capsys):
    check_backend_slug("m-sft", "-sft")
    assert capsys.readouterr().out == ""
    check_backend_slug("m", "-sft")
    assert "warning" in capsys.readouterr().out
