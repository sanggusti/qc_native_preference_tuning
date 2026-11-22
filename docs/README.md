# Documentation

## Overview

This project tests whether the language used as the finetuning and eval medium changes LLM accuracy, holding base model, task content, and pipeline fixed. Series S01 (`s01_language_medium`) crosses 2 benchmarks (gsm8k, medqa) with 9 languages (en, id, jv, su, min, ace, ban, bjn, bug) under 5 train-medium conditions (base, native, english_anchor, indonesian_anchor, pooled). Toolchain: **Adaption** for translation and finetuning, **HuggingFace Hub** for artifacts, **Inspect AI** for evals, **wandb** for tracking, **Modal** for GPU compute.

## S01 pipeline stages

1. **Datagen:** translate 1000 train + full test rows per benchmark into id, jv, su, min, ace, ban, bjn, bug; QA gates (answer invariance, GlotLID, gate steps 0-5); push 20 datasets.
2. **Train:** AutoScientist runs per benchmark, train medium, base role, and replicate (120 runs); pinned constants; push 120 checkpoints.
3. **Eval:** Inspect AI runs per benchmark, eval language, condition, base role, and replicate (468 runs); deterministic scorers on 250 items each.
4. **Analysis:** estimands E1-E9 (native effect, exposure ordering, anchor deltas, pooled contrast, translator-loss bound, replication, covariate adjustment, contamination check); hypotheses H1-H4.

## Key docs

| File | Purpose |
|------|---------|
| `research.md` | S01 research plan: question, design, conditions, hypotheses, estimands |
| `methodology.md` | Method detail for datagen, training, eval, and analysis |
| `benchmarks.md` | Benchmark protocol, registry rules, roadmap |
| `reproducibility.md` | Logging and provenance contract (env, seeds, versions, covariates) |
| `plan_review.md` | Review record of the original plan |
| `experiments.md` | S01 ledger: phases, artifacts, amendments, results, costs, pre-flight log |
| `multilingual_tasks.md` | Benchmark survey and dataset notes |
| `research/` | Literature notes 1-5 (see below) |

## Research notes

1. [Introduction and Problem Statements](research/1_intro_problem_statements.md)
2. [Preliminaries and Fundamentals](research/2_preliminaries_fundamentals.md)
3. [Contexts and Available Methodologies](research/3_contexts_methodologies.md)
4. [Available Datasets and Evaluation Tasks](research/4_datasets_evaluation.md)
5. [Language as Medium](research/5_language_as_medium.md)

## Key code

| File | Purpose |
|------|---------|
| `src/evals/tasks/translated_benchmark.py` | Parameterized Inspect eval task (benchmark, language) |
| `src/utils/registry.py` | Registries for languages, benchmarks, series, cell expansion |
| `pipeline/plan.py` | Series planner: matrix counts and stage commands |
| `pipeline/datagenerator/translate_benchmark.py` | Adaption translation pipeline with QA and HF push (#50) |
| `configs/language/` | Per-language configs (en, id, jv, su, min, ace, ban, bjn, bug) |
| `configs/benchmark/` | Per-benchmark configs (gsm8k, medqa) |
| `configs/series/s01_language_medium.yaml` | S01 series config |
