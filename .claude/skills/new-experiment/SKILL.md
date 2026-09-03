---
name: new-experiment
description: Scaffold a new experiment (new domain/language/condition combination) in this repo. Use when starting an experiment, adding an experiment config, setting up a new run series, or when the user says "new experiment", "add a condition", or names a domain x language combo to study. Enforces config-driven experiments and controlled-experiment constants.
---

# New experiment

Pipelines are generic; experiments are configs. A new experiment is a new Hydra YAML plus CLI overrides, never a forked script.

## Protocol

1. **Define what is new**: a language, a benchmark, a condition, or a series. Read `docs/methodology.md` (the design and its tiers) and `docs/reproducibility.md` (registries and steps) first.
2. **Add the registry entry**:
   - language: copy `configs/language/_template.yaml` to `configs/language/{code}.yaml`; fill every field, decide the register and record why; add the code to `languages` in the series file.
   - benchmark: confirm the scorer is language-agnostic (`docs/benchmarks.md`); copy `configs/benchmark/_template.yaml` to `configs/benchmark/{name}.yaml`; add the name to `benchmarks` in the series file; extend `src/evals/tasks/translated_benchmark.py` only if the scorer type is new.
   - condition: add it under `conditions` in the series file with `train`, `tier`, `description` and any restriction keys; activate it in `active_conditions`.
   - series: copy `configs/series/s01_language_medium.yaml` to a new file when any constant changes; never edit constants of a series that has runs.
3. **Derive the matrix**: `uv run pytest tests/test_registry.py` and `uv run python -m pipeline.plan [tier=0]`; the new cells, names and commands appear. Paste the plan into `docs/experiments.md`.
4. **Reuse the stage modules.** `src/evals/tasks/translated_benchmark.py` is the eval task for every benchmark; `pipeline/datagenerator/evals_translate/mgsm_convert.py` is the Adaption reference until `translate_benchmark.py` exists. If a stage can't express the experiment, add config keys (keep old configs working); don't hardcode.
5. **Names** come only from the series `naming` templates through `src/utils/registry.py` (see AGENTS.md).
6. **Phases**: for a new language run Phase 0 and Phase 1 of the methodology (covariates, calibration, probe, gate) before any full translation or finetune; record the results in `docs/experiments.md`.
7. **Pre-flight** (AGENTS.md checklist) before any paid run; smoke-test with `max_rows`/`--limit` small first.

## Controlled-experiment constants

All finetune conditions in a series must share: base models (ids from `client.autoscientist.list_models()`), `max_iterations`, `target_win_rate`, augmentation row counts, `data_format`, `training_type`, `train_on_inputs`, the pinned `hyperparams`, the translation settings and the decoding settings. These live in the series file under `autoscientist`, `translation` and `generation`. Changing any constant starts a new series file; never edit a constant in place after runs exist.

## Stage skills

Hand off to the stage skill for execution details: `datagen` (Adaption Adaptive Data + HF publish), `finetune` (AutoScientist + wandb + HF), `evaluate` (Inspect AI), `compute` (Modal / Lightning).
