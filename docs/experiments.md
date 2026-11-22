# Experiments: S01 Ledger (`s01_language_medium`)

Registered ledger for the language-as-medium series. Design in `docs/research.md`; logging contract in `docs/reproducibility.md`.

## 1 Series overview

| Constant | Value |
|---|---|
| series id | s01_language_medium |
| benchmarks | gsm8k, medqa |
| languages | en, id, jv, su, min, ace, ban, bjn, bug |
| conditions | base, native, english_anchor, indonesian_anchor, pooled |
| train rows | 1000 per condition |
| test items | 250 per language per benchmark (primary subset) |
| replicates | 3 |
| base models | primary + contrast, TBD in Phase 2 (from `client.autoscientist.list_models()`) |
| max_iterations | 3, frozen |
| target_win_rate | 0.7, frozen |
| translator | Adaption datasets API |
| eval harness | Inspect AI, deterministic scorers |
| HF datasets | `sanggusti/{benchmark}-{language}` (18) + pooled (2) |
| HF models | `sanggusti/{benchmark}-{language}-{condition}-{base}-r{replicate}` (120 unique) |
| wandb project | `qc_native_preference_tuning` |
| estimands | E1-E9 active; E10 retired |
| hypotheses | H1 exposure ordering, H2 English anchor, H3 task moderation, H4 pooled beats native |

## 2 Phase ledgers

### Phase 0: Catalogue and covariates (no spend)

| Item | Planned | Actual | URL | Date |
|---|---|---|---|---|
| `autoscientist.list_models()` catalogue | | | | |
| fertility per language per tokenizer | | | | |
| bits per byte per language per base | | | | |
| NusaX similarity vs Indonesian | | | | |
| FLORES reference audit (jv, min, ace) | | | | |
| premise table filled | | | | |

### Phase 1: Admission probes (about 8,000 Adaption rows)

| Item | Planned | Actual | URL | Date |
|---|---|---|---|---|
| `language_expansion` estimate per code (9) | | | | |
| FLORES calibration per language (NLLB) | | | | |
| 50-row gsm8k probe + gate per language | | | | |
| admitted languages | en, id, +2 regional min | | | |
| deferred languages with gate scores | | | | |

### Phase 2: Test subsets and base selection (about 3,800 rows + GPU)

| Item | Planned | Actual | URL | Date |
|---|---|---|---|---|
| 250-item test subsets per language per benchmark | | | | |
| untuned probes on subsets + Belebele id/jv/su | | | | |
| floor rule applied (id >= 30%, en < 80%) | | | | |
| `base_models` pinned by amendment | | | | |
| pre-registration commit hash | | | | |

### Phase 3: Full datagen (20 datasets)

| Dataset | Planned rows | Actual rows | QA pass | HF URL | wandb | Date |
|---|---|---|---|---|---|---|
| gsm8k-en | 1000/full | | | | | |
| gsm8k-id | 1000/full | | | | | |
| gsm8k-jv | 1000/full | | | | | |
| gsm8k-su | 1000/full | | | | | |
| gsm8k-min | 1000/full | | | | | |
| gsm8k-ace | 1000/full | | | | | |
| gsm8k-ban | 1000/full | | | | | |
| gsm8k-bjn | 1000/full | | | | | |
| gsm8k-bug | 1000/full | | | | | |
| gsm8k-all | 1000 | | | | | |
| medqa-en | 1000/full | | | | | |
| medqa-id | 1000/full | | | | | |
| medqa-jv | 1000/full | | | | | |
| medqa-su | 1000/full | | | | | |
| medqa-min | 1000/full | | | | | |
| medqa-ace | 1000/full | | | | | |
| medqa-ban | 1000/full | | | | | |
| medqa-bjn | 1000/full | | | | | |
| medqa-bug | 1000/full | | | | | |
| medqa-all | 1000 | | | | | |

### Phase 4: Training (120 checkpoints)

| Checkpoint | Status | Adaption run ID | HF URL | wandb | Date |
|---|---|---|---|---|---|
| plumbing smoke (recommend + download + push + serve + inspect) | | | | | |
| (one row per benchmark x train medium x base role x replicate; add rows as runs launch) | | | | | |

### Phase 5: Eval (468 runs) and analysis

| Eval run | Accuracy | Wilson 95% | Inspect log | wandb | Date |
|---|---|---|---|---|---|
| (one row per benchmark x eval language x condition x base role x replicate) | | | | | |

| Analysis item | Planned | Actual | URL | Date |
|---|---|---|---|---|
| E1-E5 estimates with CIs | | | | |
| E6 translator-loss bound | | | | |
| E7 contrast-base replication | | | | |
| E8 covariate-adjusted estimates | | | | |
| E9 contamination DiD | | | | |
| H1-H4 decisions | | | | |

## 3 Artifact index

| Type | Repo / run | URL | Version |
|---|---|---|---|
| HF dataset | `sanggusti/gsm8k-en` | | |
| HF dataset | `sanggusti/gsm8k-id` | | |
| HF dataset | `sanggusti/gsm8k-jv` | | |
| HF dataset | `sanggusti/gsm8k-su` | | |
| HF dataset | `sanggusti/gsm8k-min` | | |
| HF dataset | `sanggusti/gsm8k-ace` | | |
| HF dataset | `sanggusti/gsm8k-ban` | | |
| HF dataset | `sanggusti/gsm8k-bjn` | | |
| HF dataset | `sanggusti/gsm8k-bug` | | |
| HF dataset | `sanggusti/gsm8k-all` | | |
| HF dataset | `sanggusti/medqa-en` | | |
| HF dataset | `sanggusti/medqa-id` | | |
| HF dataset | `sanggusti/medqa-jv` | | |
| HF dataset | `sanggusti/medqa-su` | | |
| HF dataset | `sanggusti/medqa-min` | | |
| HF dataset | `sanggusti/medqa-ace` | | |
| HF dataset | `sanggusti/medqa-ban` | | |
| HF dataset | `sanggusti/medqa-bjn` | | |
| HF dataset | `sanggusti/medqa-bug` | | |
| HF dataset | `sanggusti/medqa-all` | | |
| HF models | (120 rows; add as pushed) | | |
| wandb | project `qc_native_preference_tuning` | | |

## 4 Amendments

| Date | Section affected | Change | Reason | Approved by |
|---|---|---|---|---|
| | | | | |

## 5 Results summary

| Estimand | gsm8k estimate (95% CI) | medqa estimate (95% CI) | Decision |
|---|---|---|---|
| E1 native effect (per language) | | | |
| E2 exposure ordering | | | |
| E3 Delta_en (per language) | | | |
| E4 Delta_id (per language) | | | |
| E5 pooled-vs-native (per language) | | | |
| E6 translator-loss bound | | | |
| E7 replication | | | |
| E8 adjusted | | | |
| E9 contamination DiD | | | |

## 6 Costs

| Stage | Provider | Item | Cost | Date |
|---|---|---|---|---|
| datagen | Adaption | | | |
| training | Adaption | | | |
| eval | Modal | | | |
| | | **Total** | | |

## 7 Pre-flight log

Paste one completed checklist per paid run (from AGENTS.md pre-flight check):

```
Run:
Config:
Dataset verified:
Model verified:
Persistence:
Tracking:
Cost/timeout:
Smoke first:
Date:
```
