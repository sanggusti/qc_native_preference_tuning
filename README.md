# Native Preference Tuning

Evaluate finetuned LLMs on Indonesian low-resource language tasks using [Inspect AI](https://inspect.aisi.org.uk/) and [Adaption](https://adaptionlabs.ai/).

## Research Question

Holding base model, task content and pipeline fixed, which language of finetuning data gives the best accuracy when a model is evaluated in Indonesian or one of its regional languages (Javanese, Sundanese, Minangkabau, Acehnese, Balinese, Banjarese, Buginese): the target language itself, English, Indonesian as a pivot, or a pool of all of them? The cross-language ordering of accuracy is reported alongside with its covariates. The proposal is in [docs/research.md](docs/research.md), the pre-registered design in [docs/methodology.md](docs/methodology.md), and the literature review in [docs/research/](docs/research/README.md).

## Architecture

```
Adaption (data enhancement & translation)
    ↓
Finetuned model (Adaption AutoScientist; a transparent LoRA loop on Modal repeats two cells)
    ↓
Inspect AI (evaluation & tracing)
    ↓
Results across benchmarks × languages
```

**Benchmarks:** gsm8k (math) first; the second benchmark is decided in Phase 2 against stated criteria (medqa is the placeholder); arc, mmlu and humaneval registered for a later series (see [docs/benchmarks.md](docs/benchmarks.md))  
**Languages:** English (en, reference), Indonesian (id), Javanese (jv), Sundanese (su), Minangkabau (min) in tier 0; Acehnese (ace), Balinese (ban), Banjarese (bjn) and Buginese (bug) after the gate and the floor rule; more through one registry file each

## Quick Start

```bash
# Install (uv manages the venv; extras as needed)
uv sync
uv sync --extra training --extra modal --extra dev

# Tests (live-API tests skip when keys are missing) and lint
uv run pytest
uv run ruff check .
```

### Plan a series and run evaluations

```bash
# Expand series S01 into datasets, finetunes and eval cells (nothing is submitted)
uv run python -m pipeline.plan
uv run python -m pipeline.plan tier=0 format=commands stage=eval

# Smoke the eval task with no model and no Hub access (dataset_path is
# relative to the task file; mockllm scores 0 by design, proving the path)
uv run inspect eval src/evals/tasks/translated_benchmark.py \
    --model mockllm/model -T benchmark=gsm8k -T language=id \
    -T dataset_path=../../../tests/fixtures/gsm8k_id_sample.jsonl --limit 2

# Evaluate a translated benchmark against a real model (smoke run first)
uv run inspect eval src/evals/tasks/translated_benchmark.py \
    --model hf/sanggusti/gsm8k-jv-s01_language_medium-gemma3-4b-r1 \
    -T benchmark=gsm8k -T language=jv --limit 20
```

See [docs/reproducibility.md](docs/reproducibility.md) for the registries, naming and the stage commands.

### Translate a benchmark (needs ADAPTION_API_KEY; paid)

```bash
# Smoke first: 20 rows of the Javanese gsm8k test split, no Hub push
uv run python -m pipeline.datagenerator.translate_benchmark \
    benchmark=gsm8k language=jv split=test max_rows=20
```

## Project Structure

```
qc_native_preference_tuning/
├── configs/                   # YAML configuration
│   ├── language/              # language registry, one file per language
│   ├── benchmark/             # benchmark registry, one file per standard task
│   ├── series/                # experiment series with pinned constants
│   ├── plan.yaml              # primary config for pipeline.plan
│   ├── *_benchmark.yaml, *_finetune.yaml, *_sweep.yaml, *_kit.yaml  # stage entry configs
│   └── datagenerator/, evals/, training/   # stage-level templates
├── pipeline/                  # End-to-end pipeline runners
│   ├── plan.py                # expands a series into its matrix and commands
│   ├── covariates/            # Phase 0 covariate scripts
│   ├── datagenerator/         # translation, gate, derived splits, review kit
│   ├── evals/                 # eval sweep runners (serve + run cells)
│   ├── training/              # finetuning stages (AutoScientist, transparent LoRA on Modal)
│   └── modal_runner/          # Modal cloud launcher (NLLB, inference)
├── src/                       # Core library
│   ├── analysis/              # logs, inference, equivalence, report
│   ├── evals/tasks/           # Inspect tasks (translated_benchmark, multilingual_qa)
│   ├── models/                # Model loading utilities
│   └── utils/                 # registry loader, data helpers
├── tests/                     # Pytest test suite (no-network tests plus live smoke tests)
└── docs/                      # proposal, methodology, reproducibility, literature, diagrams
```

## Metrics

**Primary:** accuracy under a language-agnostic scorer (numeric match for math, option letter for multiple choice), paired across languages by source item, with within-language contrasts (Indonesian-pivot advantage, English-anchor advantage, gain from native tuning, regression, pooling) as the causal results, each with a difference and an equivalence verdict, and the cross-language ordering reported with its covariates.

**Covariates reported with every result:** base-model exposure (bits per byte on FLORES-200), tokenizer fertility, translation quality gate results, output-language fidelity, output tokens.

**Secondary:** model-graded quality with a fixed judge, never in the primary metric.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ADAPTION_API_KEY` | Adaption API key for dataset operations |
| `HF_TOKEN` | HuggingFace Hub for datasets and models |
| `WANDB_API_KEY` | Experiment tracking |
| `OPENAI_API_KEY` | OpenAI API key (for judge model / baseline) |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional, for Claude baseline) |

Modal jobs read cloud secrets named `huggingface` and `wandb`, not the local `.env`. Paid compute is Modal plus Adaption only.

## References

- [Inspect AI documentation](https://inspect.aisi.org.uk/)
- [Adaption SDK documentation](https://docs.adaptionlabs.ai/)

See [docs/](docs/) for detailed research notes and experiment logs.
