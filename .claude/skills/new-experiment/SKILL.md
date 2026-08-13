---
name: new-experiment
description: Scaffold a new experiment (new domain/language/condition combination) in this repo. Use when starting an experiment, adding an experiment config, setting up a new run series, or when the user says "new experiment", "add a condition", or names a domain x language combo to study. Enforces config-driven experiments and controlled-experiment constants.
---

# New experiment

Pipelines are generic; experiments are configs. A new experiment is a new Hydra YAML plus CLI overrides, never a forked script.

## Protocol

1. **Define the cell**: domain (`medical`, `math`, `programming`, `science`, `general`) x language (`en`, `id`, `jv`, `su`, `min`) x stage (`datagen`, `finetune`, `eval`). Check `docs/research.md` for the experiment matrix and source datasets per domain (Medical=MedQA-USMLE, Math=MGSM, Programming=HumanEval/MBPP, Science=ARC/SciQ, General=MMLU-subset/TyDiQA).
2. **Copy the template config** for the stage:
   - datagen: `configs/datagenerator/translate/_template.yaml` -> `configs/datagenerator/translate/{domain}_{language}.yaml`
   - eval: `configs/evals/_template.yaml` -> `configs/evals/{domain}_{language}.yaml`
   - finetune: `configs/training/autoscientist.yaml` (shared constants; per-condition values via CLI overrides or a small derived YAML)
3. **Reuse the pipeline module.** `pipeline/datagenerator/evals_translate/mgsm_convert.py` is the reference datagen pipeline; `src/evals/tasks/multilingual_qa.py` the reference Inspect task. If the pipeline can't express the experiment, add config keys (keep old configs working); don't hardcode.
4. **Names** (from AGENTS.md):
   - HF dataset `sanggusti/{domain}-qa-{language}`, HF model `sanggusti/{domain}-{language}-finetuned`
   - wandb project `qc_native_preference_tuning`, run `{stage}-{domain}-{language}`, tags `[domain, language, condition]`, config = resolved Hydra config
5. **Register** the experiment in `docs/experiments.md`: config path, HF artifact URLs, wandb run URL, status.
6. **Pre-flight** (AGENTS.md checklist) before any paid run; smoke-test with `max_rows`/`--limit` small first.

## Controlled-experiment constants

All finetune conditions in a series must share: base model (pick from `client.training_models.list()`), `max_iterations`, `target_win_rate`, augmentation row counts, and `data_format`. These live in `configs/training/autoscientist.yaml`. Changing any constant starts a new experiment series with its own configs; never edit a constant in place after runs exist.

## Stage skills

Hand off to the stage skill for execution details: `datagen` (Adaption Adaptive Data + HF publish), `finetune` (AutoScientist + wandb + HF), `evaluate` (Inspect AI), `compute` (Modal / Lightning).
