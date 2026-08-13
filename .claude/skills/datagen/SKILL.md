---
name: datagen
description: Create, translate, or enhance training/eval datasets with Adaption Adaptive Data and publish them to HuggingFace Hub. Use when building preference pairs, translating a benchmark into id/jv/su/min, uploading data to Adaption, running dataset adaptation/augmentation, evaluating dataset quality, or pushing datasets to HF.
---

# Dataset generation (Adaption Adaptive Data + HF)

Reference pipeline: `pipeline/datagenerator/evals_translate/mgsm_convert.py` (HF load -> Adaption translate -> preference pairs -> `save_to_disk` + `push_to_hub`). Extend it via configs, not forks. SDK details: `references/adaption-datasets.md`.

## Lifecycle

```python
from adaption import Adaption
client = Adaption()  # reads ADAPTION_API_KEY

# 1. Ingest (pick one)
ds = client.datasets.upload_file("data.jsonl", name="medical-qa-src")        # local file helper
ds = client.datasets.create_from_huggingface(url=..., files=[...])           # HF import
# 2. Wait for async ingestion (row_count appears when ready; status "failed" = error)
client.datasets.wait_for_completion(ds.dataset_id)   # or poll get_status until row_count
# 3. Run adaptation
client.datasets.run(
    ds.dataset_id,
    column_mapping={"prompt": "question", "completion": "answer"},
    training_type="preference_pairs",          # or "instruction_dataset" (default, for SFT)
    brand_controls={"blueprint": "..."},       # freeform system prompt applied to every completion
)
result = client.datasets.wait_for_completion(ds.dataset_id, timeout=1800)
# 4. Quality + export
client.datasets.get_evaluation(ds.dataset_id)  # source vs adapted quality signals
url = client.datasets.download(ds.dataset_id)  # presigned URL
```

## Key parameters

- `training_type`: `"preference_pairs"` makes Adaption generate chosen/rejected pairs for DPO from the same `column_mapping.prompt`/`completion` sources. `"instruction_dataset"` produces enhanced prompt/completion pairs for SFT. The repo also builds pairs manually (translated answer = chosen, English answer = rejected) in `mgsm_convert.py:to_preference_dataset`, and `src/prompts/maneuver.py` (`SLIGHT_FACTUAL_ERROR`) generates degraded rejected samples.
- `language_expansion`: translation/localization; output rows ~ input x (1 + sample_rate x target_count). Three-state: omitted = keep prior config, `None` = clear, object = overwrite. The MGSM pipeline instead uses a translation `blueprint`, which rewrites rows in place.
- `column_mapping.context`: list of per-row context columns (audience, reference docs). Universal prompts are web-app only; for SDK runs use a fixed prompt column instead.
- `estimate=True` on `run()`: validates and returns the credit cost **without starting the run**. Use during pre-flight.

## Gotchas

- Ingestion and runs are async; never use a dataset before `row_count` is set. `wait_for_completion` raises on timeout but the server keeps processing.
- AutoScientist later requires >= 1,000 rows; plan generation counts accordingly.
- Supported uploads: csv, json, jsonl, parquet, pdf, docx, pptx, xlsx, html, zip, txt.
- Smoke-test with `max_rows` small (Hydra override) before full runs; runs cost credits.

## Publish to HF

- Naming: `sanggusti/{domain}-qa-{language}`. Push via `Dataset.push_to_hub(repo_id, private=...)` (uses `HF_TOKEN`), as in `mgsm_convert.py:main`.
- Keep provenance columns (`question_en`, `answer_en`, `source_language`, `target_language`) so evals and audits can trace rows.
- Log the run to wandb: project `qc_native_preference_tuning`, run `datagen-{domain}-{language}`, config = resolved Hydra config, plus the HF URL and Adaption dataset_id.

## Config

Each datagen experiment = one YAML in `configs/datagenerator/translate/` (template: `_template.yaml`). Existing example: `configs/datagenerator/evals_convert/mgsm_convert.yaml`.
