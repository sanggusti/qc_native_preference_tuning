# Documentation

## Overview

This project evaluates finetuned LLMs on Indonesian low-resource language tasks.
It uses **Adaption** for data enhancement/translation and **Inspect AI** for
structured evaluation with full tracing.

## Pipeline

1. **Data Enhancement** — Use Adaption to translate and enhance evaluation datasets
   into target languages (Indonesian, Javanese, Sundanese, Minangkabau)
2. **Model Finetuning** — Finetuned models from Adaption Autoscientists
3. **Evaluation** — Run Inspect AI tasks across domains (Medicine, Programming,
   General, Science) with translated datasets
4. **Analysis** — Compare win-rates across languages, domains, and model variants

## Key Files

| File | Purpose |
|------|---------|
| `src/evals/tasks/multilingual_qa.py` | Inspect AI evaluation task definition |
| `pipeline/datagenerator/evals_translate/mgsm_convert.py` | MGSM dataset translation via Adaption |
| `pipeline/evals/litai_tools.py` | LitAI inference helpers |
| `configs/minimal_config.yaml` | Main project configuration |

## Further Reading

- [experiments.md](experiments.md) — Experiment logs
- [multilingual_tasks.md](multilingual_tasks.md) — Benchmark survey and dataset notes
