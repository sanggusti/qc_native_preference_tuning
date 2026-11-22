---
name: compute
description: Run GPU or batch workloads on Modal for this project. Use when a job needs a GPU (inference on finetuned checkpoints, custom training), when processing data at scale, or when the user mentions Modal or where to run something.
---

# Compute (Modal)

Routing: **Modal** = GPU work (inference on finetuned checkpoints, NLLB-200 translation, the transparent LoRA backend, anything CUDA) and batch jobs. Managed finetuning runs on Adaption (see `finetune` skill); the transparent backend (`sft_check` condition, `pipeline/training/sft_finetune.py`) runs here with peft and trl, seeds from the series `sft` block and hyperparameters copied from the pinned AutoScientist recommendation. Lightning AI is down; never route work to Lightning/litai.

## Modal (GPU)

Reference: `pipeline/modal_runner/sample_modal_inference.py` (app `qc-native-preference`). Docs: `https://modal.com/llms.txt`.

```python
import modal

app = modal.App(name="qc-native-preference")
image = modal.Image.debian_slim().uv_pip_install("torch", "transformers", "peft", "trl", "wandb", "datasets")

@app.function(
    image=image,
    gpu="A100",                      # or "L40S", "A10G", "T4" for lighter jobs
    timeout=3600,                    # always set; justify in pre-flight
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
)
def run_inference(config_yaml: str): ...
```

- Secrets are **cloud secrets** named `huggingface` and `wandb` (not local `.env`). Manage at modal.com or `uv run modal secret list`.
- Run: `uv run modal run pipeline/modal_runner/<file>.py` (needs `--extra modal`; auth once via `modal setup`).
- Container storage is ephemeral: persist outputs via `push_to_hub` or a `modal.Volume`.
- Pass the experiment's Hydra config (path or dump) into the function; don't hardcode parameters.
- Pick the smallest GPU that fits; smoke-test with a few rows before full runs.

## Pre-flight

Modal bills by usage. Follow the AGENTS.md pre-flight checklist: config path, timeout set and justified, persistence via HF push, wandb run planned, smoke run first.
