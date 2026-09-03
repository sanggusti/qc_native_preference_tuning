# Experiment registry

Every series, every paid run and every artifact is recorded here. The planner output for a series is pasted in before its first paid run; phases append their numbers and spend as they complete. Amendments to a pre-registered design are logged in section 4, never edited into `docs/methodology.md`.

## 1. Series

| Series | Config | Question | Status | Pre-registration commit |
|---|---|---|---|---|
| S01 | `configs/series/s01_language_medium.yaml` | language as the medium of finetuning and evaluation (docs/methodology.md) | designed; Phase 0 not started | to be recorded at the first paid run |
| S02 | to be created from S01 with `max_iterations: 3`, `target_win_rate: 1.0` | does the adaptive AutoScientist loop change the language ranking | planned after S01 tier 0 | |
| S03 | to be created | preference tuning with cross-lingual consistency pairs | planned | |
| S04 | to be created | additional benchmarks and the second language wave (ban, bjn, bug, mad) | planned | |

## 2. S01 plan (from `uv run python -m pipeline.plan`)

| Quantity | Tier 0 | Full series |
|---|---|---|
| Datasets (translated and derived) | 22 | 24 |
| Finetune runs | 36 | 60 |
| Evaluation runs | 138 | 231 |

Base models (provisional until Phase 2): primary `google/gemma-3-4b-it`, contrast `meta-llama/Llama-3.2-3B-Instruct`. Languages: en, id, jv, su, min, ace. Benchmarks: gsm8k, medqa. Replicates: 3 (confirmed by the Phase 4 pilot).

## 3. Ledger

One row per phase and per paid unit. Planned numbers come from the planner and the `estimate=True` responses; actuals are filled in when the phase completes.

| Phase | Unit | Planned | Actual | Artifacts and run URLs | Date |
|---|---|---|---|---|---|
| 0 | live model catalogue (`client.autoscientist.list_models()`) | | | | |
| 0 | fertility, characters per item, bits per byte per language and base | | | | |
| 0 | NusaX id-min identical-token share (leakage threshold) | | | | |
| 1 | `language_expansion` probe with `estimate=True` per code | | | | |
| 1 | FLORES-100 probe per language (Adaption and NLLB-200 chrF++) | about 600 rows | | | |
| 1 | 50-row gsm8k probe per language with the full quality columns | about 300 rows | | | |
| 1 | admitted languages and deferred languages with scores | | | | |
| 2 | 250-item base evals per candidate base and language | | | | |
| 2 | pinned primary and contrast base | | | | |
| 2 | full-split base evals (the `base` cells) | | | | |
| 3 | gsm8k test split translation and gate per language | 1,319 rows x 5 | | | |
| 3 | gsm8k train split translation and gate per language | 1,000 rows x 5 | | | |
| 3 | derived splits `-rt`, `-pro1`; human-verified subsets | | | | |
| 4 | tier 0 gsm8k finetunes (primary base) | 6 x 3 | | | |
| 4 | variance pilot; replicate policy | | | | |
| 4 | tier 0 gsm8k evals | | | | |
| 5 | medqa translation, finetunes, evals | | | | |
| 6 | tier 1 contrast-base finetunes and evals | 6 x 3 | | | |
| 7 | tier 2 pooled finetunes and anchor evals | 2 x 3 | | | |

## 4. Amendments

| Date | Series | Change | Reason | Commit |
|---|---|---|---|---|
| | | | | |

## 5. Artifacts

Datasets `sanggusti/{benchmark}-{language}` (splits train, test), derived splits `-rt`, `-nllb`, `-pro1`, pooled sets `-all`; models `sanggusti/{benchmark}-{train_language}-{series}-{base}-r{replicate}`; wandb project `qc_native_preference_tuning`. Fill in URLs as they are produced.

| Artifact | URL | Gate or run summary |
|---|---|---|
| | | |

## 6. Roadmap of stage modules

Tracked as follow-up issues; the planner prints these entry points.

| Module | Needed by | Status |
|---|---|---|
| `pipeline/covariates/fertility.py`, `bits_per_byte.py` | Phase 0 | not started |
| `pipeline/datagenerator/translate_benchmark.py` | Phase 1 | not started (reference: `evals_translate/mgsm_convert.py`, fixed for SDK 0.10.0) |
| `pipeline/datagenerator/gate.py` | Phase 1 | not started |
| `pipeline/modal_runner/nllb_translate.py` | Phase 1 | not started |
| `pipeline/evals/serve_modal.py` | Phase 2 | not started |
| `pipeline/datagenerator/derive_split.py` | Phase 3 | not started |
| `pipeline/training/autoscientist_finetune.py` | Phase 4 | not started |
| `pipeline/datagenerator/pool_benchmark.py` | Phase 7 | not started |
| `src/analysis/` | Phase 4 | not started |
| `src/evals/tasks/translated_benchmark.py` | Phase 2 | implemented; execution scoring for code benchmarks pending |

## 7. Pre-flight record for the first paid run

Before the first `datasets.run` or `autoscientist.create` of S01, this section is filled with the items of the AGENTS.md checklist: the config path, the dataset verification, the model verification from the live catalogue, the push targets, the wandb run names, the cost from `estimate=True`, and the smoke run that preceded it.
