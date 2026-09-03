# Methodology: language as the medium of finetuning and evaluation

This document is the pre-registered design for experiment series S01 (`configs/series/s01_language_medium.yaml`). It fixes the question, the estimands, the conditions, the data and translation protocol, the finetuning and evaluation protocols, the analysis plan and the go/no-go rules before any paid run is submitted. Changes after the first run are recorded as amendments in `docs/experiments.md`, never edited in place. The literature that motivates each choice is in [docs/research/5_language_as_medium.md](research/5_language_as_medium.md); the evaluation of the original plan is in [docs/plan_review.md](plan_review.md); the mechanics of running the matrix are in [docs/reproducibility.md](reproducibility.md).

![Pipeline flow from configs to analysis](diagrams/pipeline_flow.png)

*Figure 1. Stages of the pipeline. Registries define the experiment; the planner derives the matrix; each stage reads its cell from the plan. Editable source: `docs/diagrams/pipeline_flow.drawio`.*

## 1. Research question and estimands

### 1.1 The question

Take one base model, one task dataset in English, one translation procedure, one finetuning recipe and one language-agnostic scorer. Translate the training and test items into a language L, finetune on the translated training items, evaluate on the translated test items. Does the accuracy depend on L, and if so, which language is the best medium for the same task?

The informal version of the question asks whether some language is a "smarter" medium for an LLM, with English anchoring and reports of token-efficient Chinese reasoning as the motivation, and with the intuition that Indonesian has a small vocabulary and a simple way to express intent. The design below treats that intuition as a set of measurable quantities rather than as an assumption.

### 1.2 Notation

Let M0 be the base model, D = (D_train, D_test) the English source items, T(D, L) the translation of D into L, F(M, D') the finetuning recipe applied to M on D', and E(M, D'') the accuracy of M on D'' under the scorer.

| Symbol | Definition | Name |
|---|---|---|
| B(L) | E(M0, T(D_test, L)) | base level |
| A(L) | E(F(M0, T(D_train, L)), T(D_test, L)) | native accuracy |
| A_en(L) | E(F(M0, D_train), T(D_test, L)) | English anchor |
| A_id(L) | E(F(M0, T(D_train, id)), T(D_test, L)) | Indonesian anchor, regional L only |
| A_all(L) | E(F(M0, pooled training set), T(D_test, L)) | pooled arm |
| RT(L) | E(F(M0, D_train), T_back(T(D_test, L))) | round trip: the L test split back-translated to English, scored with the English finetune |
| G(L) | A(L) - B(L) | gain from native tuning |
| Delta_en(L) | A_en(L) - A(L) | anchor advantage |
| Reg(L) | E(F(M0, T(D_train, L)), D_test) - A(en) | English regression after tuning in L |
| tau(L, L') | A(L) - A(L') | cross-language contrast |

### 1.3 The treatment is a bundle

Choosing L sets five things at once, and with one base model, one translator and one tokenizer they are perfectly collinear across languages:

| Component | Symbol | Set by | Manipulable here |
|---|---|---|---|
| Linguistic structure | S(L) | the language | no |
| Base-model pretraining exposure | X(L) | the base model's corpus | only by changing the base model |
| Translator competence | Q(L) | the translation model | by item-level quality, by a second translator, by human verification |
| Tokenizer fertility | Phi(L) | the base tokenizer | only by changing the base model; varies item to item |
| Surface register | R(L) | the translation blueprint | yes, within a language |

The user's question is about S(L). The literature attributes measured cross-language gaps to X, Q and Phi, and no study has isolated a residual effect of S once those are normalized (see research note, sections 2 and 3). The design therefore states, for each estimand, which components it can and cannot separate.

### 1.4 Estimands ordered by identification

Identified as causal effects of a manipulable choice, within one language, paired on items and replicated over finetuning runs:

- E1, gain from native tuning: G(L).
- E2, anchor advantages: Delta_en(L), and Delta_id(L) = A_id(L) - A(L) for regional L.
- E3, pooling advantage: A_all(L) - A(L).
- E4, English regression: Reg(L).
- E5, register effect within one language: A(jv, krama) - A(jv, ngoko) (tier 1, optional).

Identified as a paired difference of bundles, descriptive of this base model, translator and tokenizer:

- E6, the cross-language contrast tau(L, L'), and the same contrast on B and A_en. This is the user's question. It is estimable and it is reported, but it is not an effect of S(L).

Identified only up to an invariance argument:

- E7, language by base-model interaction: does the ordering of A(L) change when X(L) changes and S(L) does not? Two size-matched base models with different documented exposure are the minimum for any statement that survives the objection "that is just the corpus".

Partially identified through item-level covariates:

- E8, the residual cross-language contrast after adjusting for item-level translation quality, token count, Indonesian leakage and cap hits. These vary within a language, so their coefficients are identified; what remains still contains X(L) and S(L) together.

Descriptive only:

- E9, any regression of A(L) on language-level covariates. With six languages this is a plot with a fitted line.
- E10, any statement that a language is a "simpler" or "smarter" medium. With one language per structural profile S is not replicated, so it is not estimable. What is reported instead is the measured content of the premise: characters per item, tokens per item, bits per byte, and whether Indonesian behaves as its exposure and fertility predict.

Translation quality and tokenizer fertility are mediators of the bundle effect for the "which medium should I use" question (E6) and nuisance for the "is it the language" question (E7, E8). Both readings are reported and labelled.

## 2. Hypotheses

Each hypothesis states the direction predicted by the literature and the observation that would count against it.

- H1, exposure ordering. B(L) and A(L) are ordered by base-model exposure: en >= id > jv, su > min, ace. Against: a regional language above Indonesian on either quantity on both benchmarks.
- H2, anchor advantage on reasoning. For gsm8k, Delta_en(L) >= 0 for every regional L, and the gap widens with lower exposure. For medqa the gap is smaller because the scorer needs only an option letter. Against: Delta_en(L) < 0 with a confidence interval excluding zero for a regional language on gsm8k.
- H3, similarity to Indonesian. Among regional languages, G(L) and Delta_id(L) track lexical similarity to Indonesian: min > jv, su > ace. Against: the reverse ordering across both benchmarks.
- H4, pooled beats native. A_all(L) >= A(L) for every language at equal total rows. Against: native above pooled with an interval excluding zero in at least two languages.
- H5, translation bounds accuracy. A(en) - RT(L) is positive and increases with lower exposure; the native gap A(en) - A(L) is at least as large as the round-trip loss. Against: RT(L) near A(en) for a language whose native gap is large, which would point at the model rather than the translator.
- H6, exposure invariance. The ordering of A(L) on the contrast base model agrees with the ordering on the primary base (Kendall tau over the six languages above 0.6) and every pairwise gap keeps its sign. If it does not, the ordering is model-specific and the exposure account is preferred.
- H0, the premise. The claim that Indonesian is a simpler medium predicts that Indonesian costs fewer tokens per item than English and that A(id) is not below A(en) after adjusting for exposure. The measured fertility under current tokenizers is expected to be about 1.2 to 1.6 tokens per English token for standard written Indonesian, in which case the premise is not supported at the level of the medium.

## 3. Factors, conditions and the matrix

![Experiment matrix for series S01](diagrams/language_medium_design.png)

*Figure 2. The S01 matrix. Rows are finetuning conditions, columns are evaluation languages, cells name the estimand they feed. Editable source: `docs/diagrams/language_medium_design.drawio`.*

| Factor | Levels | Role |
|---|---|---|
| Evaluation language | en, id, jv, su, min, ace (extensible through `configs/language/`) | primary |
| Condition | base, native, english_anchor, regression, round_trip (tier 0); native_contrast, english_anchor_contrast (tier 1); indonesian_anchor, pooled (tier 2) | see 1.4 |
| Benchmark | gsm8k (math), medqa (medical); extensible through `configs/benchmark/` | task-type replication |
| Base model | primary; contrast (tier 1, gsm8k only) | E7 |
| Replicate | 3 independent finetuning runs per finetuned cell (count confirmed by the Phase 4 pilot) | run-to-run variance |

Tiers are closed under the estimands they claim, so a program that stops after any tier still yields a complete result:

| Tier | Conditions | Estimands | Finetunes | Evals |
|---|---|---|---|---|
| 0 | base, native, english_anchor, regression, round_trip | E1, E2 (Delta_en), E4, E6, the round-trip bound | 36 | 138 |
| 1 | native_contrast, english_anchor_contrast on the contrast base, gsm8k only | E7 | 18 | 33 |
| 2 | indonesian_anchor, pooled | E2 (Delta_id), E3 | 6 | 60 |

Counts come from `uv run python -m pipeline.plan` with 6 languages, 2 benchmarks and 3 replicates. Tier 0 is the minimum publishable unit.

Optional identification arms, defined but not active in S01 (each is a new language entry or condition, activated by a series amendment): jv_krama (register variant of Javanese, E5); nllb evaluation splits (second translator on the test split); a question-only arm that trains on the L question with the English rationale; a full-finetuning pair on en and jv to bound the LoRA interaction; the full off-diagonal transfer matrix.

## 4. Materials

### 4.1 Base models

Only models in `client.autoscientist.list_models()` can be finetuned; the live list is authoritative and is copied into `docs/experiments.md` at Phase 0. Selection rule for the primary base: the smallest catalogue model that (a) documents Indonesian in its pretraining languages, (b) has the lowest regional-language fertility among the candidates, and (c) clears a base accuracy floor of 30 percent on gsm8k-id and 40 percent on medqa-id (chance is 25 percent), so there is headroom in both directions. Selection rule for the contrast base: same size class and instruct status, the largest documented difference in exposure to id, jv, su, min and ace, and support for the pinned training type.

Provisional choices given the documented catalogue: primary `google/gemma-3-4b-it` (4T pretraining tokens, explicit multilingual rebalancing, a 262k-entry tokenizer described by its authors as more balanced for non-English languages); contrast `meta-llama/Llama-3.2-3B-Instruct` (eight officially supported languages, none Indonesian). `Qwen/Qwen3.5-0.8B` is the only listed model whose vendor names Javanese, Sundanese, Minangkabau, Balinese and Banjar, which makes it useful as a third base for the vendor-flag covariate, but it is a different size class and is expected to sit near floor on regional-language math; it is not one of the two identification bases. If a 3B to 4B Qwen3 model appears on the live list it replaces Gemma as the primary. If only one base model is available, E7 is dropped and the report states that exposure and language are not separable in this study.

### 4.2 Languages

| Code | Language | Family | Exposure tier | Register fixed in the blueprint | In FLORES-200 | XLM-R covered |
|---|---|---|---|---|---|---|
| en | English | Indo-European | reference | source | eng_Latn | yes |
| id | Indonesian | Malayic | medium | standard written | ind_Latn | yes |
| jv | Javanese | own branch of Malayo-Polynesian | low | ngoko | jav_Latn | yes |
| su | Sundanese | Malayo-Sumbawan | low | loma | sun_Latn | yes |
| min | Minangkabau | Malayic | very low | Agam-Tanah Datar standard | min_Latn | no |
| ace | Acehnese | Chamic | very low | standard Latin orthography | ace_Latn | no |

Registers are pinned because Javanese and Sundanese speech levels change the lexicon itself and language models are biased toward particular tiers; a translator left free will pick one silently and may mix tiers within an item. Minangkabau shares about 75 percent of its lexicon with Indonesian, so a Minangkabau translation that is really lightly modified Indonesian is a specific failure mode with its own gate check (section 6). Later languages (Balinese, Banjar, Buginese, Madurese) enter through one registry file each; Madurese is not in FLORES-200 and calibrates on NusaX instead.

### 4.3 Benchmarks

| Registry name | Domain | Train source | Test source | Scorer | Answer protocol |
|---|---|---|---|---|---|
| gsm8k | math | GSM8K train, fixed 1,000-item subset | GSM8K test, 1,319 items; the 250 MGSM items tagged as a subset | numeric match on the last number | reasoning in L, then `Answer: <integer>` in English number formatting |
| medqa | medical | MedQA USMLE train, fixed 1,000-item subset | MedQA test, 1,273 items | choice letter | `ANSWER: <letter>` |

The original plan named MGSM as the math benchmark. MGSM's value elsewhere is its human translation into ten languages, none of which is a language of this study, so for these languages MGSM offers only 250 items that would have to be machine-translated anyway. At 250 paired items the minimum detectable paired difference is about 8 accuracy points (section 10.5); the full GSM8K test gives about 3.5. The eval split is therefore the full GSM8K test with the MGSM items tagged so that the 250-item subset can be reported for comparability with published MGSM numbers.

Both scorers are language-agnostic, but each imposes a protocol that every language must satisfy identically: the choice scorer requires the literal `ANSWER:` marker (a translated marker such as `JAWABAN:` returns nothing), and the numeric scorer parses `1.500` as 1.5 and `2,5` as 25, so numerals keep English formatting in every language. These constraints are part of the bundle shared by all languages and are enforced by the gate. New benchmarks enter through `configs/benchmark/{name}.yaml`; the rule is that the scorer must be language-agnostic (numeric match, choice letter, code execution). [docs/benchmarks.md](benchmarks.md) lists which standard Inspect tasks qualify.

## 5. Data protocol

### 5.1 Item alignment and splits

Every language version of a split contains exactly the same source items, identified by a stable `source_id`, so comparisons are paired at the item level. The training split is a fixed, seeded 1,000-item subset of the English training source, identical across languages (Adaption requires at least 1,000 rows). The test split is the full English test source. Structural fields (numbers, the `####` delimiter, option order, the index of the correct option, drug names, units, code) are never translated and are verified byte-identical after translation.

### 5.2 Translation route and pinned settings

Translation uses Adaption Adaptive Data. The dedicated `language_expansion` route validates ISO 639-1 codes at request time, and Minangkabau and Acehnese have only ISO 639-3 codes, so the route is probed once with `estimate=True` for all six codes; unless all are accepted, the blueprint route (a fixed system prompt with only the language name and register substituted, as `mgsm_convert.py` does) is used for every language. Whichever route is chosen is the same for all languages. Pinned for every run: the recipe toggles `prompt_rephrase`, `deduplication` and `reasoning_traces` set explicitly to false; `brand_controls.length` fixed; `hallucination_mitigation` false (web grounding would inject language-dependent knowledge); `safety_categories` empty (filtering removes rows at rates that could differ by language); `job_specification.max_rows` and an `idempotency_key` per (benchmark, language, split). Row survival must equal the input count in every language; a language that loses rows is re-run, not trimmed.

### 5.3 Canonical schema

Every published row carries: `id` (equal to `source_id`), `question`, `choices` (multiple choice only), `target`, `instruction` (the translated instruction template, identical on every row), `instruction_en` (the English template, for the English-reasoning eval variant), `question_en`, `answer_en`, `language`, `register`, `blueprint_hash`, `translator_version`, and the quality columns of section 5.4. The Inspect task in `src/evals/tasks/translated_benchmark.py` reads exactly this schema.

### 5.4 Per-item quality columns

| Column | How computed | Used for |
|---|---|---|
| struct_ok | numbers, delimiter, option count and correct index unchanged; numerals in English format; `ANSWER:` marker present in the instruction | hard gate |
| langid_top, langid_prob | GlotLID label and confidence on the translated text, restricted to the study languages plus Indonesian, Malay and English | hard gate; leak detection |
| leak_id | share of tokens identical to the Indonesian translation of the same item | Indonesian leakage covariate (E8) |
| register_pred | honorific-level classifier or judge label for jv and su | register gate; E5 |
| qe_chrf_bt | chrF++ of a back-translation to English (same translator) against the English source | item-level translation quality |
| qe_chrf_nllb | chrF++ agreement between the Adaption translation and an NLLB-200 translation of the same item | second-opinion covariate |
| qe_cometkiwi, qe_metricx | CometKiwi and MetricX-24 quality estimation; valid as gates only for languages inside the encoders' coverage (id, jv, su); diagnostic for min and ace | item-level quality where valid |
| qe_bt_nllb | reference-based score of an NLLB-200 back-translation against the English source (coverage-safe) | item-level quality for min and ace |
| solvable_bt | a strong English model solves the back-translated item and matches the gold answer (gsm8k) or picks the gold option (medqa) | answerability |
| n_tok_primary, n_tok_contrast | tokens of the item text under each base tokenizer | fertility per item (E8) |
| human_verified, esa_score, answer_preserved | true on the fixed verified subset; error-span annotation score and a binary "answer still correct and unique" verdict from two native annotators | clean-subset analysis |

### 5.5 Derived evaluation splits

- `-rt`: the L test split back-translated to English by the same translator, published as English text with the same ids. RT(L) is evaluated with the English finetunes. If RT(L) is d points below A(en), at least d points of the native gap are chargeable to information lost in translation, since the model never sees L. The back-translation adds its own loss, so this is a bound, not a decomposition.
- `-nllb`: the test split translated by NLLB-200 (3.3B), the strongest freely available system covering all six languages. Used as a second opinion on the eval split and, if activated, as a robustness evaluation.
- `-pro1` (gsm8k only): digit re-instantiation. Numeric literals in the English source are perturbed, the answer recomputed, and the same literals substituted into every translation. English GSM8K is almost certainly in every candidate's pretraining corpus and the translations are not; if A(en) drops on the perturbed split while the other languages do not, part of the English lead was memorization. This is the one contamination control that keeps the item pairing intact.

### 5.6 Publication

Datasets are published as `sanggusti/{benchmark}-{language}` with `train` and `test` splits; derived splits use the suffixes above; the pooled training set is `sanggusti/{benchmark}-all`. The dataset card lists the gate results, the calibration numbers, the blueprint, the register, the excluded item ids and the license of the source. Translated MedQA and GSM8K carry their source licenses; NLLB outputs are CC-BY-NC.

## 6. Translation quality gate

Translation quality is correlated with the treatment (worse translations in lower-exposure languages), so it must be measured and reported per language, not only gated. The gate runs in five steps per (benchmark, language, split); thresholds are set here, before any run.

Step 0, calibration, once per language before any benchmark translation. Translate the first 100 FLORES-200 devtest sentences (fixed ids) with the exact route and blueprint the benchmarks will use, and with NLLB-200 3.3B. Score chrF++ against the human reference; the Adaption number is the reported translator quality for L. Compute the quality-estimation metrics on the human references, the NLLB outputs and deliberately corrupted references (shuffled sentences, Indonesian substituted for L, altered numbers); a metric is enabled as a gate for L only if it separates references from corruptions with AUROC above 0.8, which is expected to fail for min and ace. Calibrate the GlotLID threshold so that at least 95 percent of human references pass. Measure the identical-token share between the NusaX Indonesian and Minangkabau parallel sentences to set the leakage threshold.

Step 1, hard checks on 100 percent of rows (any failure blocks the row): `struct_ok`; GlotLID label equals L with probability above the calibrated threshold, evaluated per sentence on multi-sentence items, rejecting items labelled Indonesian, Malay or English or with more than 20 percent non-target sentences; length ratio inside the calibrated band; no refusal or meta text; chrF++ between output and English source below 0.9 (otherwise the row is untranslated).

Step 2, soft signals on 100 percent of rows, combined into a per-row risk rank: CometKiwi and MetricX-24 (gates only where calibrated); chrF++ agreement with NLLB-200; the NLLB back-translation scored against the English source; `solvable_bt`.

Step 3, an LLM error-span judge on the bottom risk decile plus a 10 percent random sample, with the English source present and few-shot examples in L. The judge triages and explains; it never accepts a row on its own for min and ace, because multilingual judges overestimate quality in low-resource languages.

Step 4, human review with error-span annotation: at least 100 items per (benchmark, language) for test splits, drawn as the 50 highest-risk rows plus 50 uniformly random rows whose ids are shared across languages; two native annotators; the agreement statistic is reported; with 100 reviewed items and no critical error found, the 95 percent upper bound on the critical-error rate is about 3 percent. Training splits get 100 random rows plus the bottom 5 percent by risk.

Step 5, admission and fallback. A language is admitted to a series when: chrF++ against the FLORES reference is at least 0.9 times NLLB-200's chrF++ on the same sentences and at least 30 absolute; chrF++ against the language's own FLORES reference exceeds chrF++ against the Indonesian reference (the leakage rule, applied to every language); GlotLID share of target-language sentences is at least 0.90 for id, jv and su and 0.80 for min and ace; structure is preserved on 100 percent of rows; register consistency is at least 0.90 for jv and su; and, on the random half of the human sample, the major-or-critical error rate is at most 5 percent and the answer-changed rate at most 2 percent. A row failing a hard check is regenerated once with the failure reason appended to the blueprint, then retranslated with NLLB-200, then excluded from every language so the parallel set stays aligned. A language failing the dataset-level rule is post-edited, or switched to NLLB-200 plus post-editing, or pivoted through Indonesian (measured on FLORES first), or deferred with its scores recorded. Reviewers may be unavailable for min and ace; those languages are then admitted provisionally, flagged in the card, scheduled last in their tier, and reported in a separate column.

## 7. Finetuning protocol

AutoScientist is a managed loop with server-side data optimization, hyperparameter adjustment and early stopping judged by an evaluator whose language behaviour is undocumented. Left free, it would optimize each language differently and the ceteris paribus assumption would fail. Every run in S01 therefore uses:

- `model` pinned to the base id returned by `recommend_hyperparams`; never omitted.
- `max_iterations = 1` and `target_win_rate = 1.0`: one training cycle, no server-side data optimization, no early stopping.
- `augmentation_domain_rows = 0` and `augmentation_general_rows = 0`: synthetic rows would be generated in a language chosen by the platform.
- `hyperparams` passed with every field, copied from one `recommend_hyperparams` call on the English gsm8k dataset of the primary base and recorded in the series file with the dataset id and date; identical across languages within a base model. `training_type = lora`; `train_on_inputs = true`, so that the translated prompt text enters the gradient (with completion-only loss on MedQA the medium would barely be learned, since the completion is a letter).
- `data_format = instruction`, explicit `column_mapping`.
- The training split uploaded with `processing_mode = raw` so the run trains on exactly the published rows.
- Equal rows (1,000) in every language as the primary budget rule, because the platform derives hyperparameters from row count; tokens per run are logged and reported. A token-matched arm is a secondary question, not a correction.
- After every run, `best_hyperparams` is compared with the submitted values and the run is rejected if they differ.
- A replicate is a fresh run with identical arguments and a new `idempotency_key`; the SDK exposes no seed. Three replicates per finetuned cell is the default; the Phase 4 pilot confirms or changes the count.
- Every run logs the resolved configuration, dataset id, row count, `best_win_rate` (as a covariate only), tokens, and the configuration commit hash to wandb.

The method is supervised finetuning on translated instruction data. The repository's earlier preference-pair construction (chosen = translated answer, rejected = English answer) teaches output-language preference rather than the task and is not used in S01. Preference optimization is series S03. The adaptive AutoScientist loop (`max_iterations = 3`, `target_win_rate = 1.0`) is series S02, run on the tier 0 cells to ask whether the platform's optimizer changes the language ranking.

## 8. Evaluation protocol

- One Inspect task, `translated_benchmark`, evaluates every cell: `-T benchmark`, `-T language`, and `-T variant` for derived splits. The solver and scorer follow the benchmark registry: `generate` with numeric match for gsm8k, `multiple_choice` with the choice scorer for medqa.
- The instruction template is translated once per language, reviewed by a human, and stored on every row; the `ANSWER:` marker and English number formatting are kept in every language.
- Decoding: temperature 0, `max_tokens` 2,048, one epoch. Output tokens per response and cap hits are logged, and accuracy at a 1,024-token cap is derived from the same logs, because tight caps swing cross-language gaps by tens of points.
- Per-sample metadata: `source_id`, `language`, `condition`, `train_language`, `base`, `replicate`, `variant`, plus the quality columns joined by id. The analysis reads Inspect logs and never re-derives these.
- Output-language fidelity: GlotLID on every response; Indonesian leakage of the response for regional languages. Accuracy is reported with and without responses produced in the wrong language.
- English regression: every finetuned model is scored on the English test split.
- English-reasoning variant (gsm8k, optional tier 2 eval): the English instruction on the translated question, on base, native and anchor models, to separate the language of evaluation from the language of reasoning.
- No LLM judge in primary metrics. Judge-based metrics may appear as secondary metrics with a fixed judge bound through `--model-role grader`.
- Finetuned checkpoints are served once per model (vLLM on Modal, or the Inspect `hf` provider on Lightning), all cells of that model run against that server, and every cell runs a 20-item smoke first.

## 9. Covariates

Per language and base model, measured in Phase 0 and reported next to every result: bits per byte of the base model on FLORES-200 devtest (tokenizer-independent, paired with a held-out slice of the project's own items in case FLORES is in pretraining); tokens per source word under the base tokenizer; corpus-side counts (MADLAD-400 clean characters, HPLT words) and the vendor language-list flag; the translator's FLORES chrF++; the gate pass rates and human error rates. Per item: the quality columns of section 5.4 and the token counts. Per response: output tokens, cap hit, output-language label, Indonesian leakage.

![Tokenizer length disparity between languages](research/figures/tokenizer_unfairness_bengali_vs_english.jpg)

*Figure 3. The same sentence tokenized in English and Bengali; the non-English version is split into many more tokens. Fertility is measured per item in this study rather than assumed. Source: Petrov et al., 2023, arXiv:2305.15425, repository assets.*

## 10. Analysis plan

### 10.1 Per cell

Accuracy with a clustered bootstrap confidence interval over items (`stderr` and `ci` with the item id as cluster in Inspect). Interval overlap is never used to compare cells.

### 10.2 Primary model

Per benchmark, an item-level logistic mixed model on the exported per-sample scores:

logit P(correct_i,r = 1) = alpha + beta_language + gamma_condition + (beta gamma)_language x condition + u_i + w_r

with u_i an item random intercept (the same items appear in every language and condition) and w_r a run random intercept over finetuning runs r = (train language, base, replicate). Contrasts of the fixed effects give E1 to E4 per language and tau(L, L') for every language pair. Per pair of cells the assumption-light check is the paired item bootstrap (B = 10,000) on d_i = y_i(cell A) - y_i(cell B), reported with McNemar's exact test on the discordant counts.

### 10.3 Families and corrections

Primary family (Holm): the six native-versus-English-anchor contrasts Delta_en(L), and the pooled language effect from the mixed model, per benchmark. Everything else (15 pairwise language contrasts per condition, G, Reg, Delta_id, round-trip bounds) is reported with Benjamini-Hochberg adjusted values and labelled exploratory.

### 10.4 Identification analyses

- E7: add base model and its interactions; report the eval-language by base-model interaction and the rank agreement (Kendall tau) between the two orderings; pre-registered reading: "exposure-consistent" if the orderings agree and every pairwise gap keeps its sign, otherwise "model-specific".
- E8: add the item-level covariates (`qe_chrf_bt`, `n_tok`, `leak_id`, cap hit) as fixed effects; report the language contrasts before and after adjustment and the share of each gap absorbed.
- Round-trip bound: A(en) - RT(L) per language with a paired interval, plotted against the native gap A(en) - A(L).
- Contamination: A(en) on the original versus the `-pro1` split, with the same contrast for every other language.
- Clean subset: the primary contrasts repeated on the human-verified subset; if the ordering differs, the full-set result is labelled translation-limited.

### 10.5 Power and the noise budget

For a paired comparison of two cells on n items with discordance rate q, the minimum detectable difference at 80 percent power and alpha 0.05 is about 2.8 x sqrt(q / n):

| n | q = 0.15 | q = 0.20 | q = 0.30 |
|---|---|---|---|
| 250 (MGSM subset) | 6.9 points | 7.9 | 9.7 |
| 1,273 (MedQA test) | 3.0 | 3.5 | 4.3 |
| 1,319 (GSM8K test) | 3.0 | 3.4 | 4.2 |

Comparing two independent accuracies instead of paired items roughly doubles the detectable difference (12.5 points at n = 250 and p = 0.5). The discordance rate is measured in the pilot and the table re-computed.

Run-to-run variance is the binding constraint. If the between-replicate standard deviation of a cell mean is s_r, the standard error of a language contrast is about sqrt(2 (s_r^2 / k + s_item^2 / n)) with k replicates, and s_r dominates once it exceeds about 1.5 points. The replicate count is set from the Phase 4 pilot: k = 3 if s_r <= 2; k = 5 if s_r is between 2 and 3.5; if s_r > 3.5 with pinned hyperparameters and one iteration, the platform is not deterministic enough for this design and the program pauses until that is resolved.

![Coverage of confidence intervals at small sample sizes](research/figures/clt_coverage_failure_small_n.png)

*Figure 4. Normal-approximation and bootstrap intervals fall well below nominal coverage at small n, while Wilson and Bayesian intervals track the target. Per-cell intervals in this study use the clustered bootstrap over items at n above 1,000 and Wilson intervals for any subset analysis below a few hundred items. Source: Bowyer et al., 2025, arXiv:2503.01747, Section 3.1 experiment.*

### 10.6 Descriptive layer

One figure per benchmark: A(L), B(L) and A_en(L) against exposure, two ways (bits per byte, and corpus counts), with fertility and translation quality as marker size and colour. Six points, no p-values. The premise about Indonesian is reported as a table: characters per item, tokens per item under each tokenizer, bits per byte, and whether id sits where exposure and fertility predict.

### 10.7 Pre-registration record

The following are fixed by this document and the series file, and the commit hash is recorded in `docs/experiments.md` before the first paid run: hypotheses H0 to H6; primary and exploratory families and corrections; the mixed model and the paired bootstrap as the primary tests; item counts and the power table; the replicate policy; the AutoScientist constants; the equal-rows budget rule; the gate thresholds and the exclusion rule (an item excluded in one language is excluded in all); decoding settings; the base-model selection rules; the go/no-go rules of section 11.

## 11. Phases, cost order and go/no-go rules

Every phase writes its numbers and its spend to `docs/experiments.md` before the next phase starts.

Phase 0, free measurements. Fix the SDK drift in the pipeline (done in this revision for the legacy converter). Record the live model catalogue. Compute fertility, characters per item and bits per byte per language for each candidate base on FLORES-200 devtest (on Lightning or Modal; the sandbox cannot reach the Hub). Measure the NusaX Indonesian-Minangkabau identical-token share. Output: the covariate table and the premise-check numbers; a candidate shortlist ordered by regional fertility. No go/no-go.

Phase 1, translation gate probes (about 1,000 Adaption rows). Probe `language_expansion` with `estimate=True`. FLORES-100 probe per language with NLLB-200 calibration. 50-row gsm8k probe per language with the full column set and a human spot check where a reader exists. Go per language on the rules of section 6. Program go: at least en, id and two regional languages admitted; otherwise revise the blueprint or route and repeat.

Phase 2, base evals and model choice (GPU hours, no credits). 250-item subsets in every admitted language for each shortlisted base; apply the selection rules of 4.1; pin the primary and contrast bases in the series file. Full-split base evals for the primary base become the `base` cells. Go: at least one base clears the floor rule; otherwise start with medqa (lower floor) or reduce to the MGSM subset with the stated power loss. This phase is publishable on its own as a benchmark-release note with covariates.

Phase 3, full gsm8k translation for admitted languages, test split first under the stricter gate, then train split; derived `-rt` and `-pro1` splits; human-verified subsets for id, jv and su. Go per language: the full-split gate matches the probe within 3 chrF++ points and structure is 100 percent.

Phase 4, tier 0 finetunes on gsm8k plus the variance pilot. Submit en first (also the plumbing smoke: artifact download, Hub push, serving, Inspect on the artifact), then id, then regional languages in gate-score order. After the en run, diff `best_hyperparams` against the submission; go only on match. Evaluate native, english_anchor, regression and round_trip cells. Pilot: replicates 2 and 3 for en, id and the lowest-gate admitted regional language; set the replicate policy from s_r. Tier 0 on gsm8k is the minimum publishable unit.

Phase 5, medqa: Phases 3 and 4 repeated. Go: the gsm8k contrasts are interpretable (s_r not above 3.5).

Phase 6, tier 1 identification: contrast-base native and anchor cells on gsm8k; jv_krama and nllb eval splits if activated. Go: at least one pairwise language contrast in A(L) on the primary base exceeds 2 s_r and the ordering is stable across replicates (Kendall tau above 0.6 over replicate orderings). If the language effects are inside replicate noise, the study reports that with this pipeline the medium does not matter beyond noise, which is a result, and tier 1 is not bought.

Phase 7, tier 2 and later series: indonesian_anchor evals, pooled finetunes, the English-reasoning variant; then S02 (adaptive AutoScientist), S03 (preference tuning with cross-lingual consistency pairs), S04 (more benchmarks and the second language wave: ban, bjn, bug, mad), each starting again at Phase 1.

## 12. Threats to validity

![Confound structure between the language and the measured accuracy](diagrams/confound_structure.png)

*Figure 5. What stands between the language and the measured accuracy, and how each factor is handled. Editable source: `docs/diagrams/confound_structure.drawio`.*

| Threat | Estimands hit | Mitigation | Residual |
|---|---|---|---|
| Pretraining exposure collinear with language | E6 to E10 | base model as a factor (E7); bits per byte and corpus counts per language; vendor flags | E7 rules out one corpus, not exposure in general |
| Translator competence collinear with language | E6, E8 | round-trip bound; NLLB second translation; item-level quality columns; human-verified subset; FLORES-calibrated gate | min and ace may be unverifiable; then descriptive only |
| Tokenizer fertility | E6 (as mediator), E8 | per-item token counts; two tokenizers through the base factor; generous cap with cap hits logged | not separable from exposure within one model |
| Register fixed silently by the translator | E6 for jv, su | register column and gate; jv_krama arm (E5) | su register not manipulated |
| Minangkabau outputs that are really Indonesian | E1, E6 for min | leakage score on data and on responses with a NusaX-calibrated threshold | reported with and without leaked responses |
| English test contamination | E6, en versus rest | digit re-instantiation split across all languages | gsm8k only; MedQA has no clean perturbation |
| AutoScientist loop and judge vary by language | all finetuned cells | one iteration, no early stop, pinned hyperparameters, `best_hyperparams` check | server nondeterminism, captured by replicates |
| Run-to-run variance larger than language effects | E1 to E7 | variance pilot; replicate count set from it; program pauses if s_r > 3.5 | cost |
| Floor effects on regional-language math | E1, E6 | base floor rule; logistic model; a language at floor contributes nothing to E7 and is reported as such | |
| Ceiling effects for English on medqa | E2, E6 | base size chosen so A(en) < 0.85 on both benchmarks | |
| Language of reasoning confounded with language of evaluation | E6 | English-reasoning prompt variant on gsm8k | |
| Culturally or terminologically bound MedQA items | E6 medqa | items whose stem depends on US-specific facts tagged; reported with and without | tagging is manual |
| Scorer protocol favours English | all | literal marker and numeral format enforced identically in every language; no judge in primary metrics | the marker itself is an English token |
| Six languages | E9, E10 | stated as descriptive | |

## 13. Reporting

The report presents, in this order: the covariate and gate tables (Phase 0 and 1, including the premise numbers); the causal within-language results E1 to E5 with intervals; the cross-language contrast E6 in three versions (full set, human-verified subset, round-trip bound alongside); the identification results E7 and E8; the descriptive exposure plots; and a one-paragraph statement of what is and is not identified. Every table names the benchmark, base model, translator, register and replicate count.

## 14. Open questions to be closed in Phase 0 and 1

1. Does Adaption's dedicated translation route accept ISO 639-3 codes (min, ace)? Probe with `estimate=True`.
2. Are submitted `hyperparams` held fixed by AutoScientist at one iteration? Verify with the `best_hyperparams` diff on the first run.
3. What are the actual fertility and bits-per-byte values for jv, su, min and ace under the candidate tokenizers? No published table exists; Phase 0 produces it.
4. What is the between-replicate standard deviation of AutoScientist runs with pinned settings? The Phase 4 pilot measures it.
5. Which honorific level and which orthography does the translator default to for jv, su, min and ace? The Phase 1 probe records it.
6. Can native reviewers be found for min and ace? If not, those languages remain descriptive in the report.
