# Experiment registry

Every series, every paid run and every artifact is recorded here. The planner output for a series is pasted in before its first paid run; phases append their numbers and spend as they complete. Amendments to a pre-registered design are logged in section 4, never edited into `docs/methodology.md`.

## 1. Series

| Series | Config | Question | Status | Pre-registration commit |
|---|---|---|---|---|
| S01 | `configs/series/s01_language_medium.yaml` | which language of finetuning data is best for each target language: native, English anchor, Indonesian pivot or pooled (docs/methodology.md) | designed and re-reviewed (docs/plan_review.md section 5); Phase 0 not started | to be recorded at the first paid run |
| S02 | to be created from S01 with `max_iterations: 3`, `target_win_rate: 1.0` | does the adaptive AutoScientist loop change the language ranking | planned after S01 tier 0 | |
| S03 | to be created | preference tuning with cross-lingual consistency pairs | planned | |
| S04 | to be created | additional benchmarks and the second language wave (ban, bjn, bug, mad) | planned | |

## 2. S01 plan (from `uv run python -m pipeline.plan`)

| Quantity | Tier 0 | Full series |
|---|---|---|
| Datasets (translated and derived) | 14 | 30 |
| Finetune runs | 15 | 66 |
| Evaluation runs | 85 | 261 |

Per tier (finetunes / evaluations): tier 0, gsm8k on en, id, jv, su, min with the native, English-anchor and Indonesian-pivot arms, 15 / 85; tier 1, the transparent-backend check on en and id (6 runs on Modal) and the contrast base on gsm8k (15), 21 / 33; tier 2, Acehnese in every lower-tier condition and the pooled model, 9 / 44; tier 3, the second benchmark (medqa as placeholder), 21 / 99.

Base models (provisional until the Phase 2 floor rule, issue #63): primary `google/gemma-3-4b-it`, contrast `meta-llama/Llama-3.2-3B-Instruct`. Languages: en, id, jv, su, min (tier 0), ace (tier 2). Benchmarks: gsm8k (tier 0); medqa parked at tier 3 pending issue #62. Replicates: 3 (to be confirmed or revised by the Phase 4 rule).

## 3. Ledger

One row per phase and per paid unit. Planned numbers come from the planner and the `estimate=True` responses; actuals are filled in when the phase completes.

| Phase | Unit | Planned | Actual | Artifacts and run URLs | Date |
|---|---|---|---|---|---|
| 0 | live model catalogue (`client.autoscientist.list_models()`) with per-size cost | | | | |
| 0 | fertility, characters per item, bits per byte per language and base | | | | |
| 0 | NusaX identical-content-token share and chrF++ of each regional language against Indonesian (leakage threshold, D2 covariate) | | | | |
| 0 | register and orthography audit of the FLORES-200 references (jv, min, ace) | | | | |
| 1 | `language_expansion` probe with `estimate=True` per code | | | | |
| 1 | FLORES-200 devtest calibration per language (Adaption and NLLB-200 chrF++, QE diagnostics) | about 5,000 rows | | | |
| 1 | 50-row gsm8k probe per language with the full quality columns | 250 rows | | | |
| 1 | admitted languages and deferred languages with scores | | | | |
| 2 | 250-item gsm8k test subset translation per admitted language | about 1,250 rows | | | |
| 2 | 250-item base evals per candidate base and language; Belebele id, jv, su probes | | | | |
| 2 | floor rule on id and min, ceiling on en; pinned primary and contrast base with size and cost (issue #63) | | | | |
| 2 | English-only finetuning probe per second-benchmark candidate; decision recorded as an amendment (issue #62) | | | | |
| 3 | gsm8k test split translation and gate per tier 0 language | 1,319 rows x 4 | | | |
| 3 | gsm8k train split translation and gate per tier 0 language | 1,000 rows x 4 | | | |
| 3 | derived splits `-rt` (NLLB-200 back-translation), `-pro1`; human-verified subsets | | | | |
| 3 | full-split base evals on the original and `-pro1` splits (the base and base_pro1 cells) | | | | |
| 4 | tier 0 gsm8k finetunes (primary base) | 5 languages x 3 replicates = 15 | | | |
| 4 | replicate rule from the pooled s_r; extra replicates if required | | | | |
| 4 | tier 0 gsm8k evals (native, english_anchor, indonesian_anchor, regression, round_trip, native_pro1) | | | | |
| 5 | sft_check runs on Modal (en, id) and their evals; H6 verdict (issue #61) | 2 x 3 | | | |
| 5 | tier 1 contrast-base finetunes and evals | 5 x 3 | | | |
| 6 | Acehnese translation, gate, finetunes and evals if admitted and above floor | test 1,319, train 1,000; 3 + 3 finetunes | | | |
| 6 | pooled finetunes and evals on gsm8k | 1 x 3 | | | |
| 7 | second benchmark: translation, finetunes, evals | | | | |

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
| `pipeline/covariates/fertility.py`, `bits_per_byte.py` | Phase 0 | not started (#49) |
| `pipeline/datagenerator/translate_benchmark.py` | Phase 1 | not started (#50; reference: `evals_translate/mgsm_convert.py`, fixed for SDK 0.10.0) |
| `pipeline/datagenerator/gate.py` | Phase 1 | not started (#51) |
| `pipeline/modal_runner/nllb_translate.py` | Phase 1 | not started (#52) |
| `pipeline/evals/serve_modal.py` | Phase 2 | not started (#55) |
| `pipeline/datagenerator/derive_split.py` | Phase 3 | not started (#53) |
| `pipeline/training/autoscientist_finetune.py` | Phase 4 | not started (#54) |
| `pipeline/training/sft_finetune.py` | Phase 5 | not started (#61) |
| `pipeline/datagenerator/pool_benchmark.py` | Phase 6 | not started (#53) |
| `src/analysis/` | Phase 4 | not started (#56; equivalence test #64) |
| `src/evals/tasks/translated_benchmark.py` | Phase 2 | implemented; execution scoring for code benchmarks pending (#57) |

Decisions pending, each recorded as an amendment when made: the second benchmark (#62, Phase 2) and the base model size against the floor rule (#63, Phase 2). Execution of Phases 0 to 2 is #59; the human review kit is #58.

## 7. Pre-flight record for the first paid run

Before the first `datasets.run` or `autoscientist.create` of S01, this section is filled with the items of the AGENTS.md checklist: the config path, the dataset verification, the model verification from the live catalogue, the push targets, the wandb run names, the cost from `estimate=True`, and the smoke run that preceded it.
