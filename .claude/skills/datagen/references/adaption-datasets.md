# Adaption datasets API reference (SDK 0.8.0, verified against installed package)

Client: `from adaption import Adaption; client = Adaption()` (env `ADAPTION_API_KEY`).
Docs serve markdown: `https://docs.adaptionlabs.ai/<path>/index.md`. Verify against `.venv/lib/python3.10/site-packages/adaption/resources/datasets/datasets.py` when in doubt.

## Methods (client.datasets)

| Method | Purpose |
|---|---|
| `create(source={...})` | Register a dataset. Local file source: `{"name", "file_format"}` -> returns `upload_instructions.url` (presigned PUT); HF/Kaggle source: `{"url", "files": [...]}`. Raw mode: add `"processing_mode": "raw"` + `"column_mapping"` when data is already prompt/completion. |
| `upload.complete_by_id(dataset_id, file_size_bytes=, sha256=)` | Confirm a presigned-URL upload. |
| `upload_file(path, name=...)` | Convenience: create + PUT + complete in one call. Used by `mgsm_convert.py`. |
| `create_from_huggingface(url=, files=)` / `create_from_kaggle(...)` | Convenience importers. Kaggle needs credentials registered in app settings. |
| `get_status(dataset_id)` | Async ingestion status. Ready when `row_count` is not None; `status == "failed"` -> inspect `error_data.message`. |
| `wait_for_completion(dataset_id, timeout=...)` | Poll helper until terminal state; raises on timeout (server keeps going). Works for ingestion and runs. |
| `run(dataset_id, ...)` | Start adaptation. See parameters below. |
| `get(dataset_id)` / `list()` | Metadata. |
| `get_evaluation(dataset_id)` | Quality signals comparing source vs adapted data. |
| `download(dataset_id)` | Presigned download URL for the adapted dataset. |
| `publish(dataset_id, ...)` | Publish/share the dataset on the platform. |

## `run()` parameters

- `column_mapping`: role assignments; required for real runs. Roles: `prompt`, `completion`, `context` (list of columns), image column for VLM data.
- `training_type`: `"instruction_dataset"` (default; enhanced prompt/completion for SFT) or `"preference_pairs"` (generates chosen/rejected for DPO from the same source columns).
- `brand_controls`: `{"blueprint": str}` freeform system prompt on every generated completion, plus structured `length`, `safety_categories`, `hallucination_mitigation`.
- `language_expansion`: adds translated/localized rows. Output ~ input x (1 + sample_rate x target_count), billed on output rows. Three-state: omit = keep prior config; `None` = clear; object = overwrite.
- `recipe_specification`: dedup/rephrase/reasoning-trace recipes; omitted = backend defaults.
- `estimate=True`: validate + credit quote without starting the run.
- `job_specification`: execution parameters.

## Three ways to inject preferences (docs/guides/applying-preferences)

1. **Blueprint** (`brand_controls.blueprint`): run-wide system prompt; tone/policy/persona.
2. **Context columns**: per-row material, `column_mapping={"context": ["audience", "reference_doc"]}`.
3. **Universal prompt**: web-app only; SDK equivalent = a fixed-string prompt column.

## Ingestion formats

csv, json, jsonl, parquet, pdf, docx, pptx, xlsx, html, zip, txt.
