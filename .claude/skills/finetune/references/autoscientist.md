# AutoScientist API reference (SDK 0.10.0, verified against installed package)

Docs: `https://docs.adaptionlabs.ai/autoscientist/overview/index.md` and siblings (`running-autoscientist`, `recommended-hyperparameters`, `supported-models`, `interpreting-results`, `download-the-model`, `data-augmentation`, `run-on-non-adapted-data`). Source of truth for signatures: `.venv/lib/python3.10/site-packages/adaption/resources/autoscientist.py`.

## Methods (client.autoscientist)

| Method | Purpose |
|---|---|
| `create(dataset_id=, ...)` | Start the loop. Params: `model`, `data_format` ("chat" default / "instruction"), `max_iterations` (default 3), `target_win_rate` (0-1; default 0.7 small / 0.8 large), `augmentation_domain_rows`, `augmentation_general_rows` (synthetic rows added pre-training, count toward effective size), `column_mapping` (inferred when omitted), `hyperparams` (overrides, not advised), `idempotency_key` (scoped to dataset_id, valid while run in progress), `voucher`. Returns `AutoscientistRun` with resolved values. |
| `get(experiment_id)` | Poll status + diagnostics. |
| `wait_for_completion(experiment_id, initial_interval=10, max_interval=60, backoff_factor=2, timeout=14400)` | Poll until succeeded/failed/cancelled. Raises `TrainingTimeout` after 4h default; the run continues server-side. |
| `list()` | All runs. |
| `cancel(experiment_id)` | Stop a run. |
| `download(experiment_id)` | Best checkpoint once succeeded. Returns a `BinaryAPIResponse` (gzip body), not a URL: call `.write_to_file(path)`. |
| `recommend_hyperparams(...)` | Inspect derived hyperparameters without launching. |
| `autoscientist.list_models()` | Base model ids for `model=`; includes per-model `id`, `display_name`, `model_size`, `context_length`, `methods` (e.g. sft, dpo), `training_types` (lora, full). `client.training_models.list()` is a deprecated alias. |

## Interpreting results

- `succeeded` = hit `target_win_rate` OR exhausted `max_iterations`. Always read `best_win_rate`.
- Loop stages per iteration: data optimization -> training -> evaluation (win rate) -> hyperparameter adjustment.
- The run object (`AutoscientistRun`) exposes `id` (pass it as `experiment_id`), resolved `model`, `status`, `iterations_completed`, resolved `max_iterations` and `target_win_rate`, `best_win_rate`, `best_hyperparams` (what the downloadable artifact was trained with), `download_available`, `error`. It exposes no per-iteration loss, learning rate or gradient norm, and nothing about the judge that computes the win rate. Log `best_hyperparams` and compare it with what you submitted.
- Controlled series: pass every `hyperparams` field explicitly (take them from one `recommend_hyperparams` call on the reference condition), pin `training_type` (lora or full) and `train_on_inputs`, set augmentation rows to 0, and pass `model` explicitly. Consider `target_win_rate=1.0` so no condition stops early on a judge whose language behaviour is unknown.

## Constraints (docs/autoscientist/supported-models)

- Minimum 1,000 rows for every model (docs claim; not stated in the SDK).
- `batch_size` is `"max"` or an integer; the ceiling depends on model and hardware and larger values are rejected at launch. Keep `"max"`.
- `lora_r` 1..64 (docs claim); `lora_alpha` = 1x or 2x `lora_r` (SDK-checked). Other `Hyperparams` fields: `n_epochs`, `learning_rate`, `lora_dropout`, `lora_trainable_modules` (`all-linear` or a module list), `lr_scheduler_type` (linear, cosine, constant), `scheduler_num_cycles`, `min_lr_ratio`, `warmup_ratio`, `max_grad_norm`, `weight_decay`, `train_on_inputs`, `training_type` (lora, full).
- Training context < serving context; DPO context < SFT context; long rows truncate silently.
- Sample of supported base models (docs, Aug 2026): `Qwen/Qwen3.5-0.8B`, `meta-llama/Llama-3.2-3B-Instruct` (full FT available), `google/gemma-3-4b-it` (full), `mistralai/Mistral-7B-Instruct-v0.2`, `openai/gpt-oss-20b`, `meta-llama/Llama-3.3-70B-Instruct-Reference` (full), `openai/gpt-oss-120b`. VLM variants accept image columns. The runtime list is authoritative.

## Notes for this repo

- For the sub-3B research question, `meta-llama/Llama-3.2-3B-Instruct` and `Qwen/Qwen3.5-0.8B` are the relevant bases.
- Docs state API training is SFT; the supported-models table lists DPO context limits (preference-pairs datasets are a first-class Adaptive Data output). If a run must be DPO, verify current `create()` behavior against the live docs before assuming.
- Non-adapted (raw) datasets work via `datasets.upload_file(path, processing_mode="raw", column_mapping={"prompt": ..., "completion": ...})`. For the controlled language series this is the intended path: the repo assembles and translates the rows, so every condition trains on exactly the rows the repo controls.
