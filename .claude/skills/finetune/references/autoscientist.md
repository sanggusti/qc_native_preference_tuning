# AutoScientist API reference (SDK 0.8.0, verified against installed package)

Docs: `https://docs.adaptionlabs.ai/autoscientist/overview/index.md` and siblings (`running-autoscientist`, `recommended-hyperparameters`, `supported-models`, `interpreting-results`, `download-the-model`, `data-augmentation`, `run-on-non-adapted-data`). Source of truth for signatures: `.venv/lib/python3.10/site-packages/adaption/resources/autoscientist.py`.

## Methods (client.autoscientist)

| Method | Purpose |
|---|---|
| `create(dataset_id=, ...)` | Start the loop. Params: `model`, `data_format` ("chat" default / "instruction"), `max_iterations` (default 3), `target_win_rate` (0-1; default 0.7 small / 0.8 large), `augmentation_domain_rows`, `augmentation_general_rows` (synthetic rows added pre-training, count toward effective size), `column_mapping` (inferred when omitted), `hyperparams` (overrides, not advised), `idempotency_key` (scoped to dataset_id, valid while run in progress), `voucher`. Returns `AutoscientistRun` with resolved values. |
| `get(experiment_id)` | Poll status + diagnostics. |
| `wait_for_completion(experiment_id, initial_interval=10, max_interval=60, backoff_factor=2, timeout=14400)` | Poll until succeeded/failed/cancelled. Raises `TrainingTimeout` after 4h default; the run continues server-side. |
| `list()` | All runs. |
| `cancel(experiment_id)` | Stop a run. |
| `download(experiment_id)` | Best checkpoint once succeeded. |
| `recommend_hyperparams(...)` | Inspect derived hyperparameters without launching. |
| `client.training_models.list()` / `autoscientist.list_models()` | Base model ids for `model=`; `list_models` includes per-model methods/limits (`model.id`, `model_size`, `methods`, `training_types`). |

## Interpreting results

- `succeeded` = hit `target_win_rate` OR exhausted `max_iterations`. Always read `best_win_rate`.
- Loop stages per iteration: data optimization -> training -> evaluation (win rate + loss/lr/grad-norm diagnostics) -> hyperparameter adjustment.

## Constraints (docs/autoscientist/supported-models)

- Minimum 1,000 rows for every model.
- `batch_size` is a per-model constant; keep the `"max"` default. Wrong values fail after acceptance.
- `lora_r` 1..64; `lora_alpha` = 1x or 2x `lora_r`.
- Training context < serving context; DPO context < SFT context; long rows truncate silently.
- Sample of supported base models (docs, Aug 2026): `Qwen/Qwen3.5-0.8B`, `meta-llama/Llama-3.2-3B-Instruct` (full FT available), `google/gemma-3-4b-it` (full), `mistralai/Mistral-7B-Instruct-v0.2`, `openai/gpt-oss-20b`, `meta-llama/Llama-3.3-70B-Instruct-Reference` (full), `openai/gpt-oss-120b`. VLM variants accept image columns. The runtime list is authoritative.

## Notes for this repo

- For the sub-3B research question, `meta-llama/Llama-3.2-3B-Instruct` and `Qwen/Qwen3.5-0.8B` are the relevant bases.
- Docs state API training is SFT; the supported-models table lists DPO context limits (preference-pairs datasets are a first-class Adaptive Data output). If a run must be DPO, verify current `create()` behavior against the live docs before assuming.
- Non-adapted (raw) datasets work via `datasets.create(source={"processing_mode": "raw", "column_mapping": {...}})` but Adaptive Data first is recommended.
