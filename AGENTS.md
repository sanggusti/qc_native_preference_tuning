# AGENTS.md

## Project overview

Research codebase for **native-language preference tuning**: does the language of the finetuning dataset affect LLM performance across low-resource Indonesian languages (Indonesian `id`, Javanese `jv`, Sundanese `su`, Minangkabau `min`, other languages to be added)? Experiments cross 5 domains (medical, math, programming, science, general) x 5 languages, finetuned and evaluated per condition. The full research plan lives in `docs/research.md`; literature review in `docs/research/`.

This is a research repo, not a product. The workflow is tool-driven:

**Adaption** (dataset enhancement + finetuning) -> **HuggingFace Hub** (publish datasets/models) -> **Inspect AI** (evals) -> **wandb** (experiment tracking), with compute on **Modal** (GPU) and **Lightning AI** (data processing, evals, hosted inference).

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
| Finetuning (managed SFT loop) | Adaption AutoScientist (`client.autoscientist`) | `pipeline/training/` |
| Evaluations | Inspect AI tasks | `src/evals/tasks/` |
| Experiment tracking, run metadata | wandb | wraps every stage |
| Publishing datasets and models | HF Hub (`push_to_hub`) | after datagen / finetune |
| GPU compute (inference, custom training) | Modal | `pipeline/modal_runner/` |
| Data processing, eval compute, hosted LLM inference | Lightning AI (`litai`) | `pipeline/evals/litai_tools.py` |

Reuse existing functions before writing new ones. Reference implementations: `pipeline/datagenerator/evals_translate/mgsm_convert.py` (Adaption translate -> preference pairs -> HF push), `src/evals/tasks/multilingual_qa.py` (parameterized Inspect task), `pipeline/evals/litai_tools.py` (litai inference), `pipeline/modal_runner/sample_modal_inference.py` (Modal app + secrets).

## Experiment configuration

**Pipelines are generic; experiments are configs.** This is mandatory.

- Every experiment gets its own Hydra YAML under `configs/{datagenerator,evals,training}/`. Templates: `configs/datagenerator/translate/_template.yaml`, `configs/evals/_template.yaml`, `configs/training/autoscientist.yaml`.
- Never hardcode experiment parameters (language, domain, model, dataset repo, row counts) in pipeline code. New experiment = new config file plus CLI overrides, reusing the existing pipeline module.
- If a pipeline can't express a new experiment, extend it with new config keys, keeping old configs working.
- Register each experiment in `docs/experiments.md` (config path, HF artifacts, wandb run, status).

### Naming conventions

- HF datasets: `sanggusti/{domain}-qa-{language}` (e.g. `sanggusti/medical-qa-jv`)
- HF models: `sanggusti/{domain}-{language}-finetuned`
- wandb: project `qc_native_preference_tuning`, run name `{stage}-{domain}-{language}` (stages: `datagen`, `finetune`, `eval`), tags for domain, language, and finetune condition. Log the resolved Hydra config as the wandb run config.
- Controlled-experiment constants: same base model (from `client.training_models.list()`), same `max_iterations` and `target_win_rate` across all finetune conditions. Changing a constant means a new experiment series, not an edit to an existing config.

## Testing

Test necessary functions, not everything.

- Prefer using existing, already-tested functions over writing new code.
- New pipeline logic gets a small test following the existing style in `tests/`: no-network unit tests where possible (like `test_inspect.py`), live smoke tests that skip when the API key is absent (like `test_adaption.py`).
- Do not test third-party SDK behavior, impossible scenarios, or code that doesn't exist yet. TDD is not required; verify by running the real command when that's cheaper than a test.

## Pre-flight check (before any paid or long run)

Adaption runs, AutoScientist training, Modal GPU jobs, and eval sweeps cost money and time. Output this checklist before submitting; if an item can't be filled, stop and complete it first.

- Config: which YAML under `configs/` defines this run (path).
- Dataset verified: columns/schema confirmed (HF hub inspection or `datasets.get_status` row count).
- Model verified: base model id exists in `client.training_models.list()` (or Inspect `--model` resolves).
- Persistence: `push_to_hub` target set; job storage is ephemeral.
- Tracking: wandb run will be created with the conventions above.
- Cost/timeout: value and justification (e.g. AutoScientist `max_iterations=3` on a 3B model; Modal `timeout=` set).
- Smoke first: run with `max_rows`/`--limit` small before the full run.

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
