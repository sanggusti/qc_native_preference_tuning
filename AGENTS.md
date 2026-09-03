# AGENTS.md

## Project overview

Research codebase on the **language of finetuning data** for Indonesian and its regional languages: holding base model, task content and pipeline fixed, when a model is evaluated in a target language (Indonesian `id`, Javanese `jv`, Sundanese `su`, Minangkabau `min`, Acehnese `ace`, English `en` as reference, more to be added), is it better finetuned on data translated into that language, kept in English, translated into Indonesian as a pivot, or pooled across the languages? The cross-language ordering of accuracy is reported alongside with its covariates (exposure, translation quality, tokenizer fertility); a claim that one language is a "smarter" medium is not estimable with this design and is not made. Experiments cross benchmarks (gsm8k first; the second benchmark is decided in Phase 2, other standard Inspect tasks later) x languages x finetuning conditions, with the same items translated per language. The proposal is `docs/research.md`, the pre-registered design is `docs/methodology.md`, the mechanics are `docs/reproducibility.md`, and the literature review is in `docs/research/`.

This is a research repo, not a product. The workflow is tool-driven:

**Adaption** (dataset enhancement + managed finetuning) -> **HuggingFace Hub** (publish datasets/models) -> **Inspect AI** (evals) -> **wandb** (experiment tracking), with compute on **Modal** (GPU: inference, NLLB, and the transparent LoRA backend that checks the managed one) and **Lightning AI** (data processing, evals, hosted inference).

## Setup and commands

- Package manager is `uv`. Always use the repo venv, never system Python: `uv run <cmd>` or `.venv/bin/python`.
- Install: `uv sync` (add `--extra training --extra modal --extra dev` as needed).
- Tests: `uv run pytest` (live-API tests skip automatically when keys are missing).
- Lint: `uv run ruff check .`
- Run an Inspect eval: `uv run inspect eval src/evals/tasks/multilingual_qa.py --model [publisher/target_model] -T language=jv -T domain=medicine`
- Run a pipeline (Hydra, config + CLI overrides): `uv run python -m pipeline.datagenerator.evals_translate.mgsm_convert target_language=Javanese max_rows=50`

## Tool routing

Use the platform built for the job. Do not hand-roll what a platform already does.

| Need | Tool | Entry point |
| --- | --- | --- |
| Dataset enhancement, translation, preference pairs | Adaption Adaptive Data (`client.datasets`) | `pipeline/datagenerator/` |
| Finetuning (managed SFT loop, the default backend) | Adaption AutoScientist (`client.autoscientist`) | `pipeline/training/autoscientist_finetune.py` |
| Finetuning (transparent backend `sft`: plain LoRA with fixed seeds, the platform-independence check) | Modal with peft and trl | `pipeline/training/sft_finetune.py` |
| Evaluations | Inspect AI tasks | `src/evals/tasks/` |
| Experiment tracking, run metadata | wandb | wraps every stage |
| Publishing datasets and models | HF Hub (`push_to_hub`) | after datagen / finetune |
| GPU compute (inference, custom training) | Modal | `pipeline/modal_runner/` |
| Data processing, eval compute, hosted LLM inference | Lightning AI (`litai`) | `pipeline/evals/litai_tools.py` |

Reuse existing functions before writing new ones. Reference implementations: `pipeline/datagenerator/evals_translate/mgsm_convert.py` (Adaption translate -> preference pairs -> HF push), `src/evals/tasks/multilingual_qa.py` (parameterized Inspect task), `pipeline/evals/litai_tools.py` (litai inference), `pipeline/modal_runner/sample_modal_inference.py` (Modal app + secrets).

## Experiment configuration

**Pipelines are generic; experiments are configs.** This is mandatory.

- Three registries define an experiment: `configs/language/{code}.yaml` (one file per language), `configs/benchmark/{name}.yaml` (one file per standard task; the scorer must be language-agnostic), and `configs/series/{series}.yaml` (base models, languages, benchmarks, a `tiers` block that parks a benchmark or a language above the minimum publishable unit, conditions with tiers and an optional training `backend`, replicates, pinned AutoScientist, `sft` and translation constants, `analysis` constants such as the equivalence bound and the floor rule, naming templates). Templates: `configs/language/_template.yaml`, `configs/benchmark/_template.yaml`; stage-level templates remain under `configs/datagenerator/`, `configs/evals/`, `configs/training/`.
- `uv run python -m pipeline.plan [series=...] [tier=0] [format=commands stage=datagen|finetune|eval]` expands a series into every dataset, finetune and eval cell with its derived names and command. A cell's tier is the maximum of its condition, benchmark and language tiers; `tier=0` is the minimum publishable unit and tiers are bought in order, each after its go/no-go rule in `docs/methodology.md` section 11. Paste the plan into `docs/experiments.md` before the first paid run of a series.
- Never hardcode experiment parameters (language, benchmark, model, dataset repo, row counts, register, seeds, bounds) in pipeline code. New language or benchmark = one registry file plus its code in the series list (and a `tiers` entry if it must wait for a gate); new experiment = new series file plus CLI overrides, reusing the existing stage modules. Parking or promoting a benchmark, a language or a condition is a tier change in the series file, never a deletion.
- If a pipeline can't express a new experiment, extend it with new config keys, keeping old configs working.
- Register each series in `docs/experiments.md` (config path, plan counts, HF artifacts, wandb runs, spend ledger, amendments).

### Naming conventions

- HF datasets: `sanggusti/{benchmark}-{language}` with splits `train` and `test` (e.g. `sanggusti/gsm8k-jv`); derived eval splits add a suffix (`-rt`, `-nllb`, `-pro1`); the pooled training set is `sanggusti/{benchmark}-all`.
- HF models: `sanggusti/{benchmark}-{train_language}-{series}-{base}{backend_slug}-r{replicate}` (e.g. `sanggusti/gsm8k-jv-s01_language_medium-gemma3-4b-r1`; the transparent backend adds `-sft`, e.g. `sanggusti/gsm8k-id-s01_language_medium-gemma3-4b-sft-r1`).
- wandb: project `qc_native_preference_tuning`, run name `{stage}-{benchmark}-{language}-{condition}-{base}-r{replicate}` (stages: `datagen`, `finetune`, `eval`), tags for benchmark, language, condition, base role and series. Log the resolved Hydra config as the wandb run config.
- Names are produced only by `src/utils/registry.py:expand_matrix` from the series `naming` templates; no stage composes a name by hand.
- Controlled-experiment constants: same base models (ids from `client.autoscientist.list_models()`; `training_models.list()` is deprecated), same `max_iterations`, `target_win_rate`, augmentation rows, `data_format`, pinned `hyperparams`, the `sft` block (seeds, GPU, timeout; hyperparameters copied from the pinned AutoScientist recommendation), translation settings, decoding settings and the `analysis` block (primary family, equivalence bound, target detectable difference, floor and ceiling rules) across every condition of a series. Changing a constant means a new series file, not an edit to an existing one; a decision the design defers to a phase (the second benchmark, the base size) is recorded as an amendment in `docs/experiments.md`.

## Testing

Test necessary functions, not everything.

- Prefer using existing, already-tested functions over writing new code.
- New pipeline logic gets a small test following the existing style in `tests/`: no-network unit tests where possible (like `test_inspect.py`), live smoke tests that skip when the API key is absent (like `test_adaption.py`).
- Do not test third-party SDK behavior, impossible scenarios, or code that doesn't exist yet. TDD is not required; verify by running the real command when that's cheaper than a test.

## Pre-flight check (before any paid or long run)

Adaption runs, AutoScientist training, Modal GPU jobs, and eval sweeps cost money and time. Output this checklist before submitting; if an item can't be filled, stop and complete it first.

- Config: which YAML under `configs/` defines this run (path).
- Dataset verified: columns/schema confirmed (HF hub inspection or `datasets.get_status` row count).
- Model verified: base model id exists in `client.autoscientist.list_models()` (or Inspect `--model` resolves).
- Persistence: `push_to_hub` target set; job storage is ephemeral.
- Tracking: wandb run will be created with the conventions above.
- Cost/timeout: value and justification (e.g. AutoScientist `max_iterations=3` on a 3B model; Modal `timeout=` set).
- Smoke first: run with `max_rows`/`--limit` small before the full run.
- Tier and floor: the cell's tier is at or below the tier whose go/no-go rule has passed (`uv run python -m pipeline.plan tier=N` lists it), and for a finetune or a full-split translation the language cleared the Phase 2 floor rule (`analysis.floor`) on the pinned base, with the numbers in `docs/experiments.md`.

## Environment

Loaded from `.env` at repo root (via `python-dotenv`; `tests/conftest.py` loads it for tests):

- `ADAPTION_API_KEY` -> Adaption (dataset enhancement, AutoScientist finetuning)
- `LIGHTNING_API_KEY` -> Lightning AI / litai
- `LITAI_MODEL` -> default litai model id (e.g. `lightning-ai/gpt-oss-120b`)
- `HF_TOKEN` -> HuggingFace Hub (datasets, models, repos)
- `WANDB_API_KEY` -> wandb tracking
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` -> judge/baseline models for Inspect evals

Modal jobs read cloud secrets named `huggingface` and `wandb` (`modal.Secret.from_name(...)`), not the local `.env`.

## Literature first

Before building a new method or experiment: find the landmark paper, follow citations to recent improvements, read methodology sections, and extract concrete recipes ("dataset X + method Y + lr Z -> score W on benchmark V"). Validate datasets exist on HF Hub with the right schema. `docs/research/` already surveys this project's area; start there. Skip only for trivial non-code operations.

## Working style

- State assumptions; if multiple interpretations exist, present them instead of picking silently.
- Minimum code that solves the problem. No speculative abstractions or unrequested flexibility.
- Surgical changes: touch only what the task requires, match existing style, clean up only your own orphans.
- Define success criteria before multi-step work and verify each step.

## Communication

- Concise and direct. No restating the request. No em-dashes.
- Include direct URLs when referencing HF datasets/models, wandb runs, or Adaption runs.
- For errors: what went wrong, why, and the fix in progress.
- Run independent tool calls in parallel. Present options only on genuine ambiguity, otherwise act.
