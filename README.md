# Native Preference Tuning

Evaluate finetuned LLMs on Indonesian low-resource language tasks using [Inspect AI](https://inspect.aisi.org.uk/) and [Adaption](https://adaptionlabs.ai/).

## Research Question

Holding base model, task content and pipeline fixed, which language of finetuning data gives the best accuracy when a model is evaluated in Indonesian or one of its regional languages (Javanese, Sundanese, Minangkabau, Acehnese): the target language itself, English, Indonesian as a pivot, or a pool of all of them? The cross-language ordering of accuracy is reported alongside with its covariates. The proposal is in [docs/research.md](docs/research.md), the pre-registered design in [docs/methodology.md](docs/methodology.md), and the literature review in [docs/research/](docs/research/README.md).

## Architecture

```
Adaption (data enhancement & translation)
    ↓
Finetuned model (Adaption AutoScientist; a transparent LoRA loop on Modal repeats two cells)
    ↓
Inspect AI (evaluation & tracing)
    ↓
Results across domains × languages
```

**Benchmarks:** gsm8k (math) first; the second benchmark is decided in Phase 2 against stated criteria (medqa is the placeholder); other standard Inspect tasks through one registry file each (see [docs/benchmarks.md](docs/benchmarks.md))  
**Languages:** English (en, reference), Indonesian (id), Javanese (jv), Sundanese (su), Minangkabau (min) in tier 0; Acehnese (ace) after the gate and the floor rule; more through one registry file each

## Quick Start

```bash
# Install core dependencies
pip install -e "."

# Install with training dependencies (torch, transformers, etc.)
pip install -e ".[training]"

# Install dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

### Plan a series and run evaluations

```bash
# Expand series S01 into datasets, finetunes and eval cells (nothing is submitted)
uv run python -m pipeline.plan
uv run python -m pipeline.plan tier=0 format=commands stage=eval

# Evaluate a translated benchmark in one language (smoke run first)
uv run inspect eval src/evals/tasks/translated_benchmark.py \
    --model hf/sanggusti/gsm8k-jv-s01_language_medium-gemma3-4b-r1 \
    -T benchmark=gsm8k -T language=jv --limit 20
```

See [docs/reproducibility.md](docs/reproducibility.md) for the registries, naming and the stage commands.

### Use Adaption for dataset enhancement

```python
from adaption import Adaption

client = Adaption()  # uses ADAPTION_API_KEY env var

# Upload and enhance a dataset
dataset_id = client.datasets.upload_file("data/my_dataset.jsonl").dataset_id
```

### Translate datasets for evaluation

```bash
python -m pipeline.datagenerator.evals_translate.mgsm_convert \
    target_language=Javanese \
    max_rows=50
```

## Project Structure

```
qc_native_preference_tuning/
├── configs/                   # YAML configuration
│   ├── language/              # language registry, one file per language
│   ├── benchmark/             # benchmark registry, one file per standard task
│   ├── series/                # experiment series with pinned constants
│   ├── plan.yaml              # primary config for pipeline.plan
│   └── datagenerator/, evals/, training/   # stage-level templates
├── pipeline/                  # End-to-end pipeline runners
│   ├── plan.py                # expands a series into its matrix and commands
│   ├── datagenerator/         # Dataset translation & enhancement
│   ├── evals/                 # LitAI evaluation tools
│   ├── training/              # finetuning stages (AutoScientist, transparent LoRA on Modal)
│   └── modal_runner/          # Modal cloud launcher
├── src/                       # Core library
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
| `OPENAI_API_KEY` | OpenAI API key (for judge model / baseline) |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional, for Claude baseline) |

## References

- [Inspect AI documentation](https://inspect.aisi.org.uk/)
- [Adaption SDK documentation](https://docs.adaptionlabs.ai/)
- A Simple Guide to Retrieval Augmented Generation, Abhinav Kimothi, Manning 2025
- Reinforcement Learning with Human Feedback, Nathan Lambert, Manning 2026

See [docs/](docs/) for detailed research notes and experiment logs.
