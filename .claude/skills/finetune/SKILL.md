---
name: finetune
description: Launch, monitor, and harvest Adaption AutoScientist finetuning runs, then publish models to HuggingFace and log to wandb. Use when training or finetuning a model (SFT/DPO/preference tuning), picking a base model, setting hyperparameters, downloading checkpoints, or when the user mentions AutoScientist or training runs.
---

# Finetuning (Adaption AutoScientist, plus the transparent LoRA backend)

AutoScientist is a managed loop: it optimizes data + hyperparameters, trains, evaluates against `target_win_rate`, and iterates up to `max_iterations`. Code goes in `pipeline/training/`; the series constants live in `configs/series/{series}.yaml` (`autoscientist` block) and the per-run override schema in `configs/training/autoscientist.yaml`. SDK details: `references/autoscientist.md`.

## Two backends

- `autoscientist` (default): every finetuned condition of a series. Pinned in S01 to one iteration, no augmentation, `processing_mode="raw"`, LoRA, loss on inputs, and `hyperparams` copied verbatim from one `recommend_hyperparams` call recorded in the series file. A replicate is a fresh run with a new `idempotency_key`; the SDK has no seed.
- `sft` (condition `sft_check`, tier 1): the same rows and the same hyperparameters, field by field, on a plain LoRA loop (peft + trl) on Modal with a fixed seed per replicate (`sft.seeds`). Entry point `pipeline/training/sft_finetune.py` (override schema `configs/training/sft.yaml`); models carry the `-sft` slug from `naming.backend_slugs`. It exists to answer whether the managed results are platform artifacts (methodology H6): the en and id `sft_check` cells must fall inside the replicate interval of the managed cells. Use the `compute` skill for the Modal mechanics; never change an AutoScientist constant to make the two agree.

## Lifecycle

```python
from adaption import Adaption
client = Adaption()

# 0. Dataset must be ingested/adapted and >= 1,000 rows (platform minimum, all models)
# 1. Base model: pin an id from the model list (never omit in a controlled series)
models = client.autoscientist.list_models().models
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
run = client.autoscientist.wait_for_completion(run.id)
# 4. Harvest: "succeeded" means target hit OR iterations exhausted; check best_win_rate
client.autoscientist.download(run.id).write_to_file("artifact.tar.gz")   # gzip body, not a URL
```

## Hyperparameter rules

- **In a controlled series, pin them once.** AutoScientist derives hyperparams from model + effective dataset size; call `client.autoscientist.recommend_hyperparams(...)` once on the reference dataset (S01: English gsm8k on the primary base), record the result in the series file with the dataset id and date, and pass every field on every run. Never let each run derive its own, and never guess values. Outside a controlled series, omitting them is fine.
- `batch_size` is model-fixed; the default `"max"` resolves it. Any other integer fails after the job is accepted.
- `lora_r` in 1..64; `lora_alpha` must be exactly 1x or 2x `lora_r` (API cross-checks).
- Rows longer than the model's training context are truncated, not rejected; DPO context limits are shorter than SFT. Pick a larger-context model for long documents.

## Controlled experiment

All conditions in a series share base model, `max_iterations`, `target_win_rate`, augmentation rows, `data_format`, `training_type`, `train_on_inputs`, `processing_mode` and the pinned `hyperparams` (see AGENTS.md). Record the resolved values returned on the run object and reject a run whose `best_hyperparams` differ from the submission. Buy finetunes in tier order (`uv run python -m pipeline.plan tier=N format=commands stage=finetune`); a language must have cleared the Phase 2 floor rule (`analysis.floor`) before its finetunes are submitted.

## Publish + track

- Push the downloaded checkpoint to HF under the name the planner derived from `naming.hf_model` (e.g. `sanggusti/gsm8k-jv-s01_language_medium-gemma3-4b-r1`; `-sft` slug for the transparent backend), with `HF_TOKEN`. Never compose the name by hand.
- AutoScientist has no wandb hook; log client-side: wandb run per the series naming in `configs/series/*.yaml`, project `qc_native_preference_tuning`, config = create() args + resolved run fields (`model`, `max_iterations`, `target_win_rate`, `best_hyperparams`, `iterations_completed`), metrics = `best_win_rate`, plus HF model URL and `run.id`. The run object has no per-iteration loss or learning-rate history.
- Pre-flight (AGENTS.md) before creating a run; runs cost credits and hours.
