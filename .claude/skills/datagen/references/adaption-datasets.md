# Adaption datasets API reference (SDK 0.10.0, verified against installed package)

Client: `from adaption import Adaption; client = Adaption()` (env `ADAPTION_API_KEY`).
Docs serve markdown: `https://docs.adaptionlabs.ai/<path>/index.md`. Verify against `.venv/lib/python3.10/site-packages/adaption/resources/datasets/datasets.py` when in doubt.

## Methods (client.datasets)

| Method | Purpose |
|---|---|
| `create(source={...})` | Register a dataset. Local file source: `{"name", "file_format"}` -> returns `upload_instructions.url` (presigned PUT); HF/Kaggle source: `{"url", "files": [...]}`. Raw mode: add `"processing_mode": "raw"` + `"column_mapping"` when data is already prompt/completion. |
| `upload.complete_by_id(dataset_id, file_size_bytes=, sha256=)` | Confirm a presigned-URL upload. |
| `upload_file(path, name=, processing_mode=, column_mapping=)` | Convenience: create + PUT + complete in one call (csv, json, jsonl, parquet only). `processing_mode="raw"` with a `column_mapping` skips adaptation and makes the rows trainable as uploaded. Used by `mgsm_convert.py`. |
| `create_from_huggingface(url=, files=)` / `create_from_kaggle(...)` | Deprecated; use `create(source={"url": ..., "files": [...]})`. |
| `get_status(dataset_id)` | Status is one of pending, running, awaiting_input, succeeded, failed. Ingestion is ready when `row_count` is not None; `awaiting_input` means uploaded and waiting for `run()`. `status == "failed"` -> inspect `error_data.message`. |
| `wait_for_completion(dataset_id, timeout=...)` | Polls until succeeded or failed; `awaiting_input` is not terminal, so call it after `run()`, never right after upload (poll `row_count` for ingestion instead). Raises on timeout (server keeps going). |
| `run(dataset_id, ...)` | Start adaptation. See parameters below. |
| `get(dataset_id)` / `list()` | Metadata. |
| `get_evaluation(dataset_id)` | Quality signals comparing source vs adapted data. |
| `download(dataset_id, file_format="jsonl")` | Streams the processed rows as a binary body (`BinaryAPIResponse`): call `.write_to_file(path)`. Parquet arrives as a tar.gz of shards. Not a URL. |
| `publish(dataset_id, ...)` | Returns 501 in 0.10.0 (not implemented); push to HF with `Dataset.push_to_hub` instead. |

## `run()` parameters

- `column_mapping`: role assignments; required for real runs. Roles: `prompt`, `completion`, `context` (list of columns), image column for VLM data.
- `training_type`: `"instruction_dataset"` (default; enhanced prompt/completion for SFT) or `"preference_pairs"` (generates chosen/rejected for DPO from the same source columns).
- `brand_controls`: `{"blueprint": str}` freeform system prompt on every generated completion, plus structured `length`, `safety_categories`, `hallucination_mitigation`.
- `language_expansion`: adds translated/localized rows. Output ~ input x (1 + sample_rate x target_count), billed on output rows. Three-state: omit = keep prior config; `None` = clear; object = overwrite.
- `recipe_specification`: `version` plus `recipes={prompt_rephrase, deduplication, reasoning_traces}` booleans; omitted = backend defaults, so set all three explicitly in controlled runs.
- `job_specification`: `max_rows`, `idempotency_key`.
- `estimate=True`: validate + credit quote without starting the run (also the way to probe whether a language code is accepted by `language_expansion`: unknown codes return 400 with a sample of supported codes).
- Adapted downloads carry `enhanced_prompt` / `enhanced_completion` next to `original_*` columns; map them explicitly instead of assuming the source column names survive.

## Three ways to inject preferences (docs/guides/applying-preferences)

1. **Blueprint** (`brand_controls.blueprint`): run-wide system prompt; tone/policy/persona.
2. **Context columns**: per-row material, `column_mapping={"context": ["audience", "reference_doc"]}`.
3. **Universal prompt**: `column_mapping.universal_prompt` (requires at least one `context` column).

## Ingestion formats

csv, json, jsonl, parquet, pdf, docx, pptx, xlsx, html, zip, txt.
