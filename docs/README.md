# Documentation

Research project on the language of finetuning data for Indonesian and its regional languages: when a model is evaluated in Indonesian, Javanese, Sundanese, Minangkabau or Acehnese, is it better finetuned on data translated into that language, kept in English, translated into Indonesian as a pivot, or pooled across the languages? The cross-language ordering of accuracy is reported alongside with its covariates. Data translation and enhancement run on Adaption Adaptive Data, finetuning on Adaption AutoScientist, evaluation on Inspect AI, tracking on wandb, with artifacts on the Hugging Face Hub.

## Read in this order

1. [research.md](research.md): the proposal. Question, why it is worth doing and what is not the contribution, hypotheses, contributions, design in brief, relation to prior work, program.
2. [methodology.md](methodology.md): the pre-registered design for series S01. Estimands and what is identified, hypotheses with difference and equivalence verdicts, conditions and tiers (tiers compose over condition, benchmark and language), data and translation protocol, the translation quality gate, the managed and the transparent finetuning backends, evaluation protocol, covariates, analysis plan with power, phases with go/no-go rules, threats to validity.
3. [plan_review.md](plan_review.md): the original plan, what was at risk, what changed and why, the toolchain audit against the installed SDKs, and the second review that asked whether the topic is worth running and reordered the design around the pivot question.
4. [reproducibility.md](reproducibility.md): registries, the planner, naming, stage commands, how to add a language or a benchmark, what is implemented and what remains.
5. [benchmarks.md](benchmarks.md): every standard Inspect task, whether its scorer is language-agnostic, and what a translated variant needs.
6. [experiments.md](experiments.md): the series registry, the S01 plan, the spend ledger, amendments, artifacts and the module roadmap.

## Literature

[research/](research/README.md) holds the literature review in five notes: introduction and problem statements, preliminaries and multilingual internals, adaptation methods, datasets and evaluation, and the new note on language as the medium (language of thought, language difficulty, the target languages, prior crossed designs, base-model exposure, regional-language resources, translation quality assessment, evaluation statistics). Figures under `research/figures/` are taken from the cited papers' repositories and credited in captions.

## Diagrams

`diagrams/` holds editable draw.io sources with SVG and PNG renders: `pipeline_flow` (configs to analysis), `language_medium_design` (the S01 matrix and estimands), `confound_structure` (what stands between the language and the measured accuracy). `make_diagrams.py` regenerates the sources from Python specs and `render.py` renders them with the draw.io viewer in headless Chromium. The older `experiment_design`, `evals_datagenerator` and `rlhf` diagrams describe the pre-revival plan.

## Key files

| File | Purpose |
|---|---|
| `configs/language/*.yaml` | language registry |
| `configs/benchmark/*.yaml` | benchmark registry |
| `configs/series/*.yaml` | experiment series with pinned constants, tiers and analysis bounds |
| `configs/training/autoscientist.yaml`, `configs/training/sft.yaml` | per-run override schemas for the managed and the transparent finetuning backends |
| `pipeline/plan.py` | expands a series into datasets, finetunes, evals and commands |
| `src/utils/registry.py` | registry loader and matrix expansion |
| `src/evals/tasks/translated_benchmark.py` | the parameterized Inspect task for every translated benchmark |
| `pipeline/datagenerator/evals_translate/mgsm_convert.py` | legacy Adaption translation reference, fixed for SDK 0.10.0 |
| [multilingual_tasks.md](multilingual_tasks.md) | earlier benchmark survey, kept for reference |
