"""Tests for configs package."""

from pathlib import Path
from configs import load_config, CONFIG_DIR
from configs.datagenerator import DataGeneratorConfig
from configs.evals import EvalsConfig
from configs.training import TrainingConfig, LoRAConfig


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "model" in cfg
    assert "training" in cfg
    assert "evals" in cfg


def test_load_config_custom_path(tmp_path):
    yaml_content = "key: value\nnested:\n  a: 1\n"
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml_content)
    cfg = load_config(p)
    assert cfg["key"] == "value"
    assert cfg["nested"]["a"] == 1


def test_config_dir_exists():
    assert CONFIG_DIR.is_dir()


def test_data_generator_config_defaults():
    cfg = DataGeneratorConfig()
    assert cfg.generator_model == "gpt-4o-mini"
    assert "en" in cfg.languages


def test_data_generator_config_from_dict():
    cfg = DataGeneratorConfig.from_dict({"generator_model": "gpt-4", "num_samples": 50})
    assert cfg.generator_model == "gpt-4"
    assert cfg.num_samples == 50


def test_evals_config_defaults():
    cfg = EvalsConfig()
    assert cfg.judge_model == "gpt-4o-mini"
    assert "win_rate" in cfg.metrics


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.method == "dpo"
    assert cfg.beta == 0.1


def test_lora_config_defaults():
    cfg = LoRAConfig()
    assert cfg.r == 16
    assert "q_proj" in cfg.target_modules
