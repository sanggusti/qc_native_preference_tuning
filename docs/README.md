# qc_native_preference_tuning Documentation

## Overview

This project implements **English and Native Language Preference Tuning** for small LLMs.  
It provides a complete pipeline for:

1. **Data Generation** — synthesising preference datasets (chosen / rejected pairs)  
2. **Training** — fine-tuning small LLMs with DPO (Direct Preference Optimization) or SFT  
3. **Evaluation** — measuring win-rates and quality with an LLM judge  
4. **Modal Runner** — running GPU workloads on [Modal](https://modal.com) cloud infrastructure  

---

## Project Structure

```
qc_native_preference_tuning/
├── configs/                   # YAML config + Python config dataclasses
│   ├── config.yaml            # Main configuration file
│   ├── datagenerator/         # Data generator config
│   ├── evals/                 # Evaluation config
│   └── training/              # Training config (LoRA, hyperparams)
├── pipeline/                  # End-to-end pipeline runners
│   ├── datagenerator/         # Preference data generation
│   ├── evals/                 # Evaluation pipeline
│   ├── training/              # DPO / SFT training pipeline
│   └── modal_runner/          # Modal cloud launcher
├── src/                       # Core library
│   ├── models/                # Model loading & LoRA utilities
│   ├── evals/                 # Evaluator base classes & metrics
│   │   └── tasks/             # LLM-judge task implementation
│   └── utils/                 # Shared helpers (logging, seeding, …)
├── tests/                     # Pytest test suite
│   └── conftest.py            # Shared fixtures
└── docs/                      # Project documentation
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `configs/config.yaml` to set your model, dataset, and training hyperparameters.

### 3. Generate preference data

```python
from configs import load_config
from pipeline.datagenerator import run_data_generation_pipeline

config = load_config()
prompts = ["What is machine learning?", "Explain transformers in NLP."]
samples = run_data_generation_pipeline(config, prompts, "data/train.jsonl")
```

### 4. Train

```python
from configs import load_config
from pipeline.training import run_training_pipeline

config = load_config()
run_training_pipeline(config)
```

### 5. Evaluate

```python
from configs import load_config
from pipeline.evals import run_eval_pipeline

config = load_config()
metrics = run_eval_pipeline(config, model_path="outputs/model")
print(metrics)  # {"win_rate": 0.65, "num_samples": 200}
```

### 6. Run on Modal

```bash
modal run pipeline/modal_runner/__init__.py
```

---

## Configuration Reference

See `configs/config.yaml` for the full list of configurable options, including:

| Section         | Key fields                                                     |
|-----------------|----------------------------------------------------------------|
| `model`         | `base_model`, `torch_dtype`, `model_max_length`                |
| `data`          | `dataset_name`, `max_prompt_length`, `max_length`              |
| `training`      | `method`, `learning_rate`, `num_train_epochs`, `beta`          |
| `lora`          | `r`, `lora_alpha`, `target_modules`                            |
| `evals`         | `judge_model`, `num_eval_samples`, `metrics`                   |
| `datagenerator` | `generator_model`, `num_samples`, `temperature`                |
| `modal`         | `app_name`, `gpu`, `gpu_count`, `secret_names`                 |

---

## Supported Training Methods

| Method | Description                               |
|--------|-------------------------------------------|
| `dpo`  | Direct Preference Optimization (default)  |
| `sft`  | Supervised Fine-Tuning on chosen responses|

---

## Languages

The project currently supports:

- `en` — English  
- `id` — Indonesian (Bahasa Indonesia)  

Additional languages can be added via the `languages` key in `configs/config.yaml`.
