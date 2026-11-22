"""No-network tests for pipeline.plan command builders."""

from types import SimpleNamespace

from pipeline.plan import (
    build_datagen_commands,
    build_eval_commands,
    build_finetune_commands,
    format_matrix,
)


def _train_cells() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            series="s01_language_medium",
            benchmark="gsm8k",
            condition="native",
            train_language="jv",
            base_role="small",
            base_model="Qwen/Qwen3-4B",
            replicate=1,
            tier=0,
            train_dataset_id="sanggusti/gsm8k-jv",
            hub_model_id="sanggusti/gsm8k-jv-r1",
        ),
        SimpleNamespace(
            series="s01_language_medium",
            benchmark="gsm8k",
            condition="native",
            train_language="jv",
            base_role="small",
            base_model="Qwen/Qwen3-4B",
            replicate=2,
            tier=0,
            train_dataset_id="sanggusti/gsm8k-jv",
            hub_model_id="sanggusti/gsm8k-jv-r2",
        ),
    ]


def _eval_cells() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            series="s01_language_medium",
            benchmark="gsm8k",
            eval_language="jv",
            condition="native",
            train_language="jv",
            replicate=1,
            tier=0,
            test_dataset_id="sanggusti/gsm8k-jv",
            model_ref="openai-api/sanggusti/gsm8k-jv-r1",
            log_dir="logs/s01/gsm8k-jv-native-r1",
            base_role="small",
            base_model="Qwen/Qwen3-4B",
        ),
        SimpleNamespace(
            series="s01_language_medium",
            benchmark="gsm8k",
            eval_language="jv",
            condition="base",
            train_language=None,
            replicate=1,
            tier=0,
            test_dataset_id="sanggusti/gsm8k-jv",
            model_ref="openai-api/small",
            log_dir="logs/s01/gsm8k-jv-base",
            base_role="small",
            base_model=None,
        ),
    ]


def test_build_datagen_commands() -> None:
    """One translate command per id; pooled -all ids are skipped."""
    commands = build_datagen_commands(
        ["sanggusti/gsm8k-id", "sanggusti/gsm8k-all", "sanggusti/medqa-jv"],
        "s01_language_medium",
    )
    assert commands == [
        "uv run python -m pipeline.datagenerator.translate_benchmark "
        "benchmark=gsm8k language=id series=s01_language_medium "
        "push_to_hub=sanggusti/gsm8k-id",
        "uv run python -m pipeline.datagenerator.translate_benchmark "
        "benchmark=medqa language=jv series=s01_language_medium "
        "push_to_hub=sanggusti/medqa-jv",
    ]


def test_build_finetune_commands() -> None:
    """One finetune command per train cell with exact flags."""
    commands = build_finetune_commands(_train_cells())
    assert len(commands) == 2
    assert commands[0] == (
        "uv run python -m pipeline.training.autoscientist_finetune "
        "series=s01_language_medium benchmark=gsm8k base=small language=jv "
        "replicate=1 hub_model_id=sanggusti/gsm8k-jv-r1"
    )
    assert commands[1] == (
        "uv run python -m pipeline.training.autoscientist_finetune "
        "series=s01_language_medium benchmark=gsm8k base=small language=jv "
        "replicate=2 hub_model_id=sanggusti/gsm8k-jv-r2"
    )


def test_build_eval_commands() -> None:
    """Eval commands use the cell model ref verbatim."""
    commands = build_eval_commands(_eval_cells())
    assert len(commands) == 2
    assert commands[0] == (
        "uv run inspect eval src/evals/tasks/translated_benchmark.py "
        "--model openai-api/sanggusti/gsm8k-jv-r1 "
        "-T benchmark=gsm8k -T language=jv -T split=test "
        "--log-dir logs/s01/gsm8k-jv-native-r1"
    )
    assert commands[1] == (
        "uv run inspect eval src/evals/tasks/translated_benchmark.py "
        "--model openai-api/small "
        "-T benchmark=gsm8k -T language=jv -T split=test "
        "--log-dir logs/s01/gsm8k-jv-base"
    )


def test_format_matrix_contains_counts() -> None:
    """Matrix text reports cell counts, dataset count, and conditions."""
    text = format_matrix(
        _train_cells(),
        _eval_cells(),
        ["sanggusti/gsm8k-id", "sanggusti/gsm8k-jv"],
        "s01_language_medium",
    )
    assert "Series: s01_language_medium" in text
    assert "Finetune cells: 2" in text
    assert "Eval cells: 2" in text
    assert "Unique datasets: 2" in text
    assert "base: 1" in text
    assert "native: 1" in text
