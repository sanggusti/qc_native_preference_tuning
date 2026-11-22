# Reproducibility Contract (S01)

What must be logged, pinned, and preserved so that series `s01_language_medium` can be rerun and audited. Paid compute is Modal plus Adaption only.

## 1 Principles

1. Every number in `docs/experiments.md` traces to a wandb run, an HF artifact, or an Inspect log.
2. Configs define runs; CLI overrides are logged, never silent. The resolved Hydra config is stored on every wandb run.
3. Nothing paid runs without a pre-flight record (see `docs/experiments.md` section 7) and a reviewed `pipeline.plan` output.
4. Retranslation beats patching: flawed datasets get a new version, never an in-place edit.
5. Seeds are first-class: all sampling, shuffling, and mixture decisions take explicit seeds.

## 2 Environment and seeds

- Package manager `uv`; repo venv only (`uv run <cmd>`). Record `uv pip freeze` output and Python version on every run.
- Sampling seed: the series `sampling_seed` governs train/test row sampling and the pooled-mixture fill. No run invents its own seed.
- Replicates are independent runs distinguished by replicate index with a fresh `idempotency_key` each; AutoScientist exposes no training seed. Fixed training seeds apply only to the transparent SFT backend once its series block lands.
- Seeds govern: train/test row sampling, translation batch order, pooled-mixture selection, review-sample selection, bootstrap resampling in analysis.
- Modal jobs pin the image tag and record the function call id and image digest in the job log; Adaption runs record the returned run/dataset IDs.

## 3 Data versions

- HF datasets: `sanggusti/{benchmark}-{language}` (18 canonical: gsm8k/medqa x 9 languages) plus `sanggusti/{benchmark}-all` pooled (2).
- Each dataset card states: source benchmark and split, translation blueprint text, translator (Adaption dataset ID), row counts (train 1000, full test), sampling seeds, GlotLID pass rate, gate summary, schema (canonical columns), and version number.
- Versioning: `v1`, `v2`, ... on dataset revision; the series config pins the exact version or commit hash used. In-place overwrites are forbidden.
- English datasets are versioned identically (sampling is a versioned decision even without translation).

## 4 Model versions

- Base models: primary plus contrast IDs from `client.autoscientist.list_models()`, recorded in the Phase 2 ledger before any training run. Changing either starts a new series.
- HF models: `sanggusti/{benchmark}-{language}-{condition}-{base}-r{replicate}` (120 unique checkpoints; `{language}` is the train medium, `{base}` a short base-model tag).
- Each model card states: base model ID, training dataset version, AutoScientist run ID, pinned hyperparameters, `max_iterations`, `target_win_rate`, replicate index, training rows, and eval languages it serves (noting degenerate double-label cells).
- AutoScientist artifacts are downloaded and pushed to HF immediately; Adaption-side storage is not archival.

## 5 Evaluation logging

- One Inspect log per eval run (468 total), stored with the wandb eval run and linked from the ledger.
- Each eval wandb run logs: resolved config, model repo and revision, dataset repo and revision, scorer versions, 250 item IDs, per-item outputs and scores, and aggregate accuracy with Wilson interval.
- Deterministic scorers only for primary metrics (numeric match for gsm8k, letter match for medqa). Any model-graded secondary score uses a frozen judge prompt with the judge model ID and version logged.
- Modal inference jobs log image tag, GPU type, timeout, and per-run cost.

## 6 Training logging

- One wandb run per AutoScientist training run, named `finetune-{benchmark}-{language}-{condition}-{base}-r{replicate}`, tagged with benchmark, language, condition, base role, and replicate.
- Each training run logs: resolved config, dataset version, base model, all AutoScientist parameters, iteration-level win rates and checkpoints, final artifact hash, and HF push URL.
- Failed or stopped runs stay in the ledger with status and reason; they are never deleted or renamed into success.

## 7 Datagen procedure

1. Sample 1000 train rows with the series `sampling_seed` (same items across replicates; replicates vary runs, not data); translate the full test split.
2. Translate train and test rows into id, jv, su, min, ace, ban, bjn, bug via Adaption with the benchmark blueprint; en rows pass through unchanged.
3. QA gates per translated dataset: schema check, answer-key invariance (numbers, units, MCQ letters), GlotLID language-ID pass rate above threshold, gate steps 0-5 per methodology section 6.
4. Push passing datasets to HF with full cards; record URLs in the ledger. Failing datasets are retranslated as a new version.
5. Datagen wandb runs are named `datagen-{benchmark}-{language}` and log the blueprint, Adaption IDs, seeds, QA metrics, and HF URL.

## 8 Covariate provenance

E8 adjusts E1, E3, and E4 for item covariates. Every covariate value must trace to a logged computation:

| Covariate | Source | Logged at |
|---|---|---|
| character length, token count | fixed tokenizer (ID and version logged) | datagen run |
| fertility (tokens per word) | same tokenizer | datagen run |
| bits per byte | base model NLL per byte on reference text | covariate run |
| numeric density | regex counter, versioned in repo | datagen run |
| translation quality score | gate columns (NLLB agreement, back-translation, ESA) | datagen run |
| lexical similarity | NusaX identical-content-token share and chrF++ | covariate run |
| item ID, benchmark, language | dataset version | dataset card |
| replicate index | series config | training run |

Analysis code reads covariates only from these logged artifacts, never recomputed ad hoc with different settings. The analysis script version (git hash) is recorded with every results update.
