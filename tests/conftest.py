"""Shared test fixtures for qc_native_preference_tuning."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture()
def sample_config() -> dict:
    """Return a minimal configuration dictionary suitable for tests."""
    return {
        "model": {
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "torch_dtype": "bfloat16",
            "attn_implementation": "eager",
            "model_max_length": 512,
        },
        "data": {
            "dataset_name": "test_dataset",
            "train_split": "train",
            "eval_split": "test",
            "prompt_column": "prompt",
            "chosen_column": "chosen",
            "rejected_column": "rejected",
            "max_prompt_length": 256,
            "max_length": 512,
            "seed": 42,
        },
        "training": {
            "method": "dpo",
            "output_dir": "/tmp/test_outputs/model",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-6,
            "bf16": False,
            "beta": 0.1,
            "report_to": "none",
        },
        "lora": {
            "use_lora": False,
        },
        "evals": {
            "judge_model": "gpt-4o-mini",
            "judge_temperature": 0.0,
            "num_eval_samples": 10,
            "batch_size": 2,
        },
        "datagenerator": {
            "generator_model": "gpt-4o-mini",
            "num_samples": 5,
            "temperature": 0.8,
            "max_tokens": 256,
            "topics": ["general_knowledge"],
            "languages": ["en"],
        },
        "modal": {
            "app_name": "test_app",
            "gpu": "T4",
            "gpu_count": 1,
            "timeout": 3600,
            "secret_names": [],
        },
    }


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Return a mocked OpenAI client that returns a predefined judge response."""
    client = MagicMock()

    def _mock_create(**kwargs):
        response = MagicMock()
        response.choices[0].message.content = "A\nResponse A was more accurate and helpful."
        return response

    client.chat.completions.create.side_effect = _mock_create
    return client


@pytest.fixture()
def preference_samples() -> list:
    """Return a small list of synthetic preference samples."""
    return [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "France is a country in Europe.",
        },
        {
            "prompt": "Explain what a neural network is.",
            "chosen": (
                "A neural network is a computational model inspired by the human brain, "
                "consisting of interconnected nodes (neurons) organized in layers."
            ),
            "rejected": "Neural networks are a type of machine learning algorithm.",
        },
    ]
