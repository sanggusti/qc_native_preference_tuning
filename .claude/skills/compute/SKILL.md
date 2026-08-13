---
name: compute
description: Run GPU or batch workloads on Modal and Lightning AI for this project. Use when a job needs a GPU (inference on finetuned checkpoints, custom training), when processing data at scale, when calling hosted LLMs via litai, or when the user mentions Modal, Lightning, or where to run something.
---

# Compute (Modal + Lightning AI)

Routing: **Modal** = GPU work (inference on finetuned checkpoints, anything CUDA). **Lightning AI** = data processing, eval compute, and hosted LLM inference via `litai`. Managed finetuning does not need either; it runs on Adaption (see `finetune` skill).

## Modal (GPU)

Reference: `pipeline/modal_runner/sample_modal_inference.py` (app `qc-native-preference`). Docs: `https://modal.com/llms.txt`.

```python
import modal

app = modal.App(name="qc-native-preference")
image = modal.Image.debian_slim().uv_pip_install("torch", "transformers", "peft", "wandb", "datasets")

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

## Lightning AI (litai + studios)

Reference: `pipeline/evals/litai_tools.py`. Env: `LIGHTNING_API_KEY`, default model in `LITAI_MODEL` (e.g. `lightning-ai/gpt-oss-120b`).

```python
from litai import LLM
llm = LLM(model=os.environ["LITAI_MODEL"])
# repo helpers (async): inference_with_litai(question), multiturn_conversation_with_litai(question, history)
```

- Use litai hosted models for baselines and judges when an OpenAI/Anthropic key isn't preferred.
- Batch data processing and eval sweeps run in Lightning Studios: a studio is a persistent cloud machine; clone the repo, `uv sync`, set `.env`, and run the same Hydra/inspect commands as locally. Nothing in the pipeline should be studio-specific.
- Note: `litai` is imported by code/tests but not declared in `pyproject.toml`; install it in the environment that runs these helpers.

## Pre-flight

Both platforms bill by usage. Follow the AGENTS.md pre-flight checklist: config path, timeout set and justified, persistence via HF push, wandb run planned, smoke run first.
