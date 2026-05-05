# qc_native_preference_tuning

English and Native Language Preference Tuning project for small LLMs.

This repository provides end-to-end pipelines for generating preference datasets, training models with DPO/SFT, and evaluating them using an LLM-as-judge approach.

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
