# Native Preference Tuning

English and Native Language Preference Tuning project for small LLMs.

This repository provides end-to-end pipelines for generating preference datasets, training models with DPO/SFT, and evaluating them using an LLM-as-judge approach.

Research Questions to be answered by this project:

- Can a hybrid refinement architecture (English structural Chain-of-Thought + Indonesian native output) match the conversational accuracy of massive API-based models when using a sub-3B parameter model?
- How much synthetic Indonesian preference data (via DPO) is actually required at the final output stage to recover the signal degradation caused by translating an English-anchored critique?

See [docs/README.md](docs/README.md) for full documentation.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from configs import load_config
from pipeline.training import run_training_pipeline

config = load_config()
run_training_pipeline(config)
```
