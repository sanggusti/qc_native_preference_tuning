# AGENTS.md

## Project overview

Research codebase for **native-language preference tuning**: holding base model, task content, and pipeline fixed, does the language used as the finetuning and eval medium change accuracy? Series S01 (`s01_language_medium`) crosses 2 benchmarks (gsm8k, medqa) x 9 languages (`en id jv su min ace ban bjn bug`) x 5 conditions (base, native, english_anchor, indonesian_anchor, pooled), finetuned and evaluated per cell. The full research plan lives in `docs/research.md`; method in `docs/methodology.md`; literature review in `docs/research/`.

This is a research repo, not a product. The workflow is tool-driven:

**Adaption** (dataset enhancement + finetuning) -> **HuggingFace Hub** (publish datasets/models) -> **Inspect AI** (evals) -> **wandb** (experiment tracking), with GPU compute on **Modal**. (Lightning AI is down; never route work to Lightning/litai.)

## Setup and commands

- Package manager is `uv`. Always use the repo venv, never system Python: `uv run <cmd>` or `.venv/bin/python`.
- Install: `uv sync` (add `--extra training --extra modal --extra dev` as needed).
- Tests: `uv run pytest` (live-API tests skip automatically when keys are missing).
- Lint: `uv run ruff check .`
- Run an Inspect eval: `uv run inspect eval src/evals/tasks/translated_benchmark.py --model [provider/model] -T benchmark=gsm8k -T language=jv -T split=test`
- Plan a series sweep: `uv run python -m pipeline.plan` (matrix) or `format=commands stage=eval filter.tier=0` (commands)
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
| GPU compute (inference, custom training, batch jobs) | Modal | `pipeline/modal_runner/` |

Reuse existing functions before writing new ones. Reference implementations: `pipeline/datagenerator/evals_translate/mgsm_convert.py` (Adaption translate -> preference pairs -> HF push), `src/evals/tasks/translated_benchmark.py` (parameterized S01 Inspect task), `src/utils/registry.py` + `pipeline/plan.py` (series expansion), `pipeline/modal_runner/sample_modal_inference.py` (Modal app + secrets).

## Experiment configuration

**Pipelines are generic; experiments are configs.** This is mandatory.

- Every experiment gets its own Hydra YAML under `configs/{datagenerator,evals,training}/`. Templates: `configs/datagenerator/translate/_template.yaml`, `configs/evals/_template.yaml`, `configs/training/autoscientist.yaml`.
- Series structure lives in registries: `configs/language/{code}.yaml`, `configs/benchmark/{name}.yaml`, `configs/series/{series}.yaml` (templates: `_template.yaml` in each group). Adding a language is one registry file plus one code in the series list; adding a benchmark is one registry file plus one name in the series list. Artifact names come only from `src/utils/registry.py`, never composed by hand.
- Never hardcode experiment parameters (language, domain, model, dataset repo, row counts) in pipeline code. New experiment = new config file plus CLI overrides, reusing the existing pipeline module.
- If a pipeline can't express a new experiment, extend it with new config keys, keeping old configs working.
- Register each experiment in `docs/experiments.md` (config path, HF artifacts, wandb run, status).

### Naming conventions

- HF datasets: `sanggusti/{benchmark}-{language}` (e.g. `sanggusti/gsm8k-jv`), pooled `sanggusti/{benchmark}-all`, derived `-rt`/`-nllb`/`-pro1` suffixes (templates in the series `naming` block)
- HF models: `sanggusti/{benchmark}-{language}-{condition}-{base}-r{replicate}`
- wandb: project `qc_native_preference_tuning`, run names `datagen-{benchmark}-{language}`, `finetune-{benchmark}-{language}-{condition}-{base}-r{replicate}`, `eval-{benchmark}-{language}-{condition}-{base}-r{replicate}`; tags for benchmark, language, condition, base role. Log the resolved Hydra config as the wandb run config.
- Controlled-experiment constants: same base models, same `max_iterations`, `target_win_rate`, augmentation rows, `data_format`, and pinned `hyperparams` across all finetune conditions (series `autoscientist` block). Changing a constant means a new experiment series, not an edit to an existing config.

## Testing

Test necessary functions, not everything.

- Prefer using existing, already-tested functions over writing new code.
- New pipeline logic gets a small test following the existing style in `tests/`: no-network unit tests where possible (like `test_inspect.py`), live smoke tests that skip when the API key is absent (like `test_adaption.py`).
- Do not test third-party SDK behavior, impossible scenarios, or code that doesn't exist yet. TDD is not required; verify by running the real command when that's cheaper than a test.

## Pre-flight check (before any paid or long run)

Adaption runs, AutoScientist training, Modal GPU jobs, and eval sweeps cost money and time. Output this checklist before submitting; if an item can't be filled, stop and complete it first.

- Config: which YAML under `configs/` defines this run (path).
- Sweep reviewed: `uv run python -m pipeline.plan` output checked for the cells this run covers.
- Dataset verified: columns/schema confirmed (HF hub inspection or `datasets.get_status` row count).
- Model verified: base model id exists in `client.autoscientist.list_models()` (or Inspect `--model` resolves).
- Persistence: `push_to_hub` target set; job storage is ephemeral.
- Tracking: wandb run will be created with the conventions above.
- Cost/timeout: value and justification (e.g. AutoScientist `max_iterations=3` on a 3B model; Modal `timeout=` set).
- Smoke first: run with `max_rows`/`--limit` small before the full run.

## Environment

Loaded from `.env` at repo root (via `python-dotenv`; `tests/conftest.py` loads it for tests):

- `ADAPTION_API_KEY` -> Adaption (dataset enhancement, AutoScientist finetuning)
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
