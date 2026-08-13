---
name: finetune
description: Launch, monitor, and harvest Adaption AutoScientist finetuning runs, then publish models to HuggingFace and log to wandb. Use when training or finetuning a model (SFT/DPO/preference tuning), picking a base model, setting hyperparameters, downloading checkpoints, or when the user mentions AutoScientist or training runs.
---

# Finetuning (Adaption AutoScientist)

AutoScientist is a managed loop: it optimizes data + hyperparameters, trains, evaluates against `target_win_rate`, and iterates up to `max_iterations`. Code goes in `pipeline/training/`; per-series constants in `configs/training/autoscientist.yaml`. SDK details: `references/autoscientist.md`.

## Lifecycle

```python
from adaption import Adaption
client = Adaption()

# 0. Dataset must be ingested/adapted and >= 1,000 rows (platform minimum, all models)
# 1. Base model: omit for auto-select, or pin from the training-models list
models = client.training_models.list()
# 2. Launch (constants come from configs/training/autoscientist.yaml)
run = client.autoscientist.create(
    dataset_id=dataset_id,
    model=cfg.model,                      # or omit; resolved model returned on the run
    data_format=cfg.data_format,          # "chat" (default) or "instruction"
    max_iterations=cfg.max_iterations,    # platform default 3
    target_win_rate=cfg.target_win_rate,  # default 0.7 small / 0.8 large models
    augmentation_domain_rows=cfg.augmentation_domain_rows,
    augmentation_general_rows=cfg.augmentation_general_rows,
    idempotency_key=f"{domain}-{language}-{series}",  # safe retries while in progress
)
# 3. Monitor (default timeout 4h; raises TrainingTimeout but run continues server-side)
run = client.autoscientist.wait_for_completion(run.experiment_id)
# 4. Harvest: "succeeded" means target hit OR iterations exhausted; check best_win_rate
url = client.autoscientist.download(run.experiment_id)   # best checkpoint
```

## Hyperparameter rules

- **Don't override hyperparams.** AutoScientist derives them from model + effective dataset size. Inspect the derivation with `client.autoscientist.recommend_hyperparams(...)` instead of guessing.
- `batch_size` is model-fixed; the default `"max"` resolves it. Any other integer fails after the job is accepted.
- `lora_r` in 1..64; `lora_alpha` must be exactly 1x or 2x `lora_r` (API cross-checks).
- Rows longer than the model's training context are truncated, not rejected; DPO context limits are shorter than SFT. Pick a larger-context model for long documents.

## Controlled experiment

All conditions in a series share base model, `max_iterations`, `target_win_rate`, augmentation rows, and `data_format` (see AGENTS.md). Record the resolved values returned on the run object; omitted params resolve server-side and differ per model.

## Publish + track

- Push the downloaded checkpoint to HF as `sanggusti/{domain}-{language}-finetuned` (`HF_TOKEN`).
- AutoScientist has no wandb hook; log client-side: wandb run `finetune-{domain}-{language}` in project `qc_native_preference_tuning` with config = create() args + resolved run fields, metrics = `best_win_rate`, per-iteration diagnostics (loss, lr, grad norm) from the run object, plus HF model URL and `experiment_id`.
- Pre-flight (AGENTS.md) before creating a run; runs cost credits and hours.
