# Native Preference Tuning

Evaluate finetuned LLMs on Indonesian low-resource language tasks using [Inspect AI](https://inspect.aisi.org.uk/) and [Adaption](https://adaptionlabs.ai/).

## Research Questions

- Does a sub-3B model finetuned with English critique/refinement + Indonesian/Native Language output achieve comparable win-rates against large API baselines on Indonesian conversational tasks?
- Does increasing Indonesian/Native Language DPO preference pairs improve final answer quality, and where do gains saturate?

## Architecture

```
Adaption (data enhancement & translation)
    ↓
Finetuned model (via Adaption Autoscientists)
    ↓
Inspect AI (evaluation & tracing)
    ↓
Results across domains × languages
```

**Domains:** Medicine, Programming, General, Science  
**Languages:** Indonesian (id), Javanese (jv), Sundanese (su), Minangkabau (min)

## Quick Start

```bash
# Install core dependencies
pip install -e "."

# Install with training dependencies (torch, transformers, etc.)
pip install -e ".[training]"

# Install dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

### Run evaluations with Inspect AI

```bash
# Run the multilingual QA task with a specific model
inspect eval src/evals/tasks/multilingual_qa.py --model openai/gpt-4o-mini

# Run with a local model
inspect eval src/evals/tasks/multilingual_qa.py --model hf/your-finetuned-model
```

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
│   ├── minimal_config.yaml    # Main config (domains, languages, models)
│   └── datagenerator/         # Data generation configs
├── pipeline/                  # End-to-end pipeline runners
│   ├── datagenerator/         # Dataset translation & enhancement
│   ├── evals/                 # LitAI evaluation tools
│   ├── training/              # DPO / SFT training pipeline
│   └── modal_runner/          # Modal cloud launcher
├── src/                       # Core library
│   ├── evals/                 # Inspect AI evaluation tasks
│   │   └── tasks/             # Task definitions (multilingual QA, etc.)
│   ├── models/                # Model loading utilities
│   └── utils/                 # Data loading, translation helpers
├── tests/                     # Pytest test suite
└── docs/                      # Research notes & experiment logs
```

## Metrics

**Primary:** LLM-as-judge win-rate (via Inspect AI `model_graded_qa`)

**Secondary:**
- Indonesian/Native language fluency
- Factuality
- Instruction following
- Cultural/naturalness score

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
