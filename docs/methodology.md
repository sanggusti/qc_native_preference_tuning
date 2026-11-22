# Methodology: s01_language_medium

Series `s01_language_medium` tests whether the language medium of finetuning data changes task accuracy when evaluation language is held fixed. Design is parallel content across languages, equal training rows per arm, fixed base models, and pre-registered estimands E1-E9. E10 is retired.

Fixed facts: languages `en id jv su min ace ban bjn bug` (English, Indonesian, Javanese ngoko, Sundanese loma, Minangkabau, Acehnese, Balinese, Banjarese, Buginese). Benchmarks `gsm8k` (numeric match) and `medqa` (choice). `train_rows=1000` per arm. Test subsets 250 items. Replicates 3. HF datasets `sanggusti/{benchmark}-{language}`. HF models `sanggusti/{benchmark}-{language}-{condition}-{base}-r{replicate}`. wandb project `qc_native_preference_tuning`.

## 1 Question and estimands

### 1.1 Primary question

Does finetuning in the native evaluation language improve accuracy over (a) no finetuning, (b) finetuning on identical content in English, (c) finetuning on identical content in Indonesian, and (d) finetuning on a pooled multilingual mixture at equal total rows?

### 1.2 Scope

Two benchmarks: grade school math reasoning via GSM8K [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) and medical multiple choice via MedQA [arXiv:2009.13081](https://arxiv.org/abs/2009.13081). Six evaluation languages above. One series, one translation protocol, one finetuning protocol, one analysis plan.

### 1.3 Notation

Let `A(b, e, c, m, r)` be accuracy on benchmark `b`, eval language `e`, train condition `c`, base model `m`, replicate `r`. Conditions: `base`, `native`, `english_anchor`, `indonesian_anchor`, `pooled`. Native for eval language `e` means finetuned in `e`. Anchors mean finetuned in English or Indonesian on parallel content. Base means eval-only with no finetuning. All finetune arms use 1000 rows total.

### 1.4 Estimands E1-E9 and retired E10

E1 native effect: `A(native in L) - A(base)`, evaluated in L, per benchmark and base, averaged over replicates. Primary efficacy of native finetuning.

E2 exposure ordering: monotone association between E1 and pretraining exposure rank `en > id > jv > su > min, ace, ban, bjn, bug` (lowest tier grouped, ties allowed). Estimated by rank correlation of per-language E1 against exposure order plus pairwise native-effect contrasts. Tests whether lower exposure languages gain more from native finetuning.

E3 Delta_en: `Delta_en(L) = A(english_anchor) - A(native in L)`, evaluated in L. Positive values favor the English anchor. Direct native versus English-anchor contrast on parallel content.

E4 Delta_id: `Delta_id(L) = A(indonesian_anchor) - A(native in L)`, evaluated in L. Positive values favor the Indonesian anchor. Direct native versus Indonesian-anchor contrast on parallel content.

E5 pooled-vs-native: `A(pooled) - A(native in L)`, evaluated in L, at equal total rows (1000). Pooled data is the `-all` mixture defined in 5.5.

E6 translator-loss bound: upper bound on accuracy loss attributable to translation error rather than language medium. Estimated from `-rt` and `-nllb` derived splits (5.5) via the round-trip bound in 10.3.

E7 contrast-base replication: replication of the sign and rank order of E3 and E4 on the contrast base model versus the primary base, per the rules in 10.6.

E8 covariate-adjusted effect: E1, E3, and E4 after adjustment for tokenizer fertility, bits per byte, language similarity, and reference audit score (Section 9, model in 10.7).

E9 contamination difference-in-differences: memorization check comparing original versus digit re-instantiated (`-pro1`) accuracy gaps across base and finetuned models, per 10.4.

E10 RETIRED: the structural question of whether some language is intrinsically a smarter medium is not estimable with one language per structural profile in this series. It required multiple languages per profile or mechanistic interventions. Retired, not estimated. Any structural language in reports must be labeled interpretation, not an estimand. Off-diagonal native transfer cells (train in A, eval in B) are out of scope for the same reason; S01 runs native diagonally only.

## 2 Contrasts and hypotheses

### 2.1 Contrasts

`Delta_en(L)` and `Delta_id(L)` are defined in 1.4. Positive values favor the anchor; negative values favor native finetuning. All contrasts are paired by test item within benchmark and eval language.

### 2.2 Hypotheses

H1 exposure ordering: E1 is largest in the lowest exposure tier (`min`, `ace`, `ban`, `bjn`, `bug`), then `su`, `jv`, intermediate in `id`, smallest in `en`. Formally, E1 is monotone nonincreasing in exposure rank.

H2 English-anchor advantage: for low exposure eval languages on gsm8k, `Delta_en(L) > 0`. English-anchor finetuning beats native finetuning when evaluated in L, because reasoning routes through the stronger English representation (Wendler et al., 2024, [arXiv:2402.10588](https://arxiv.org/abs/2402.10588); Wu et al., 2024, [arXiv:2411.04986](https://arxiv.org/abs/2411.04986)). Indonesian-anchor comparison is E4, with no directional hypothesis.

H3 task moderation: the anchor advantage is larger on gsm8k than on medqa, where knowledge must exist in L. Formally, `Delta_en(L)` on gsm8k exceeds `Delta_en(L)` on medqa for low exposure L. Tested by the benchmark interaction on E3.

H4 pooled beats native: E5 > 0. The pooled mixture at equal total rows beats single-language native finetuning, by combining anchor strength with native interface coverage.

## 3 Conditions and arms

Five arms per benchmark, language, base, and replicate:

1. `base`: eval-only. No finetuning. Provides the E1 baseline and the contamination reference in 10.4.
2. `native`: 1000 rows in eval language L, for L in `jv su min ace ban bjn bug`. For `en` and `id` eval, native coincides with the anchor arms (degenerate cells below); no separate native-en or native-id checkpoints are trained.
3. `english_anchor`: 1000 rows of parallel English content.
4. `indonesian_anchor`: 1000 rows of parallel Indonesian content.
5. `pooled`: 1000 rows total, drawn equally from all admitted languages in the `-all` split (5.5). Equal total rows is the control: pooled must not exceed 1000 rows.

Degenerate cells: for eval language `en`, `native` and `english_anchor` name the same training medium and the same checkpoint; for eval language `id`, `native` and `indonesian_anchor` coincide. Train once, evaluate once, record the accuracy under both labels. The registry encodes this in the series `degenerate_cells` list. This dedup rule is part of the registered design, not a post hoc fix.

All arms share the same base checkpoint, the same pinned hyperparameters (Section 7), the same train/test item IDs per language, and the same decoding at eval (Section 8).

## 4 Languages and bases

### 4.1 Base selection

Candidate bases come from `client.autoscientist.list_models()`. Selection applies a fixed floor rule on the Phase 2 250-item subsets (gsm8k-id, gsm8k-en, and the second benchmark once admitted), run untuned before any finetuning:

1. `gsm8k-id >= 30 percent`. The base must show usable Indonesian math reasoning, else native effects are floor compressed.
2. English below 80 percent on every benchmark used. The base must leave headroom on English, else anchor effects are ceiling compressed.
3. Prefer smallest. Among bases passing rules 1 and 2, pick the smallest parameter count to minimize cost and leave the largest expected finetuning signal.

Pin the selected primary and contrast base IDs in the series file by amendment. Changing a base starts a new series, not a revision.

### 4.2 Language admission and pinned registers

Admitted set: `en`, `id`, `jv`, `su`, `min`, `ace`, `ban`, `bjn`, `bug`. Pinned registers, used verbatim in blueprints and instruction templates:

* `en`: neutral standard English.
* `id`: standard formal Indonesian.
* `jv`: Javanese ngoko (casual familiar register), Latin script.
* `su`: Sundanese loma (neutral everyday register), Latin script.
* `min`: standard written Minangkabau, Latin script.
* `ace`: standard written Acehnese, Latin script.
* `ban`: standard written Balinese, Latin script.
* `bjn`: standard written Banjarese, Latin script.
* `bug`: standard written Buginese, Latin script.

A language is admitted only after its datasets pass the Section 6 quality gate. If a language fails admission, its eval cells are dropped per Section 12 and the pooled mixture is rebalanced over admitted languages only, still at 1000 rows total. Register is a pinned experimental constant; do not mix registers within a language.

## 5 Translation and datasets

### 5.1 Sources and subsets

Sources: GSM8K train split for math [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) and MedQA USMLE train split for medicine [arXiv:2009.13081](https://arxiv.org/abs/2009.13081). MGSM [arXiv:2210.03057](https://arxiv.org/abs/2210.03057) is the human-translated precedent for math transfer evals but is not the training source. NLLB [arXiv:2207.04672](https://arxiv.org/abs/2207.04672) is the independent translator for the `-nllb` split.

Subsets: 1000 train rows per benchmark per language, sampled by the series `sampling_seed` with disjoint IDs; the full test split translated once. Test IDs are identical across languages (parallel test set). Train IDs are identical across languages (parallel train set). Primary evals run on a fixed 250-item test subset per benchmark. HF dataset repos: `sanggusti/{benchmark}-{language}`, with `train` and `test` splits, plus derived repos below.

### 5.2 Blueprint route

All Adaption translations use the blueprint route: `brand_controls={"blueprint": ...}` with `{language}` and `{register_instruction}` substitution from the registry (see benchmarks doc). Blueprints require: preserve all numerals in Latin digits, preserve GSM8K answer values exactly, preserve MedQA choice letters and option order, preserve units and code spans, translate only prose, use the pinned register. The `language_expansion` route is probed once with `estimate=True` per code and recorded; the default stays the blueprint route.

### 5.3 Adaption execution

SDK 0.8.0, `client.datasets`: `upload_file`, `get_status` polling until `row_count` (never `wait_for_completion` right after upload), `run` with `column_mapping` and `brand_controls.blueprint`, `get_evaluation`, `download`. Notes: `datasets.run` has NO `idempotency_key`; only `autoscientist.create` does. `datasets.download` returns a `str` URL, not a file object. `create_from_huggingface` is deprecated; prefer explicit upload. Every run uses `estimate=True` first with the estimate logged, and logs the resolved Hydra config, dataset ID, blueprint hash, and row counts to wandb.

### 5.4 Canonical schema and per-item quality columns

Every published row carries the canonical columns: `id`, `question`, `choices`, `target`, `instruction`, `instruction_en`, `question_en`, `answer_en`, `language`, `register`, `blueprint_hash`, `translator_version`.

Every translated row additionally carries quality columns: `struct_ok`, `lid_pass`, `lid_conf`, `length_ratio`, `echo_chrf` (untranslated-echo check), `nllb_agree_chrf` (agreement with the NLLB-200 second opinion), `bt_chrf` and `bt_comet` (back-translation scored against the English source), `solvable_bt` (digit re-instantiated back-translation still solves), `leak_id` (Indonesian leakage on content tokens), `borrowed_term_share`, `fallback_count`, `judge_esa` with `judge_model` (advisory LLM error-span judge), `human_verified`, `esa_score`, `answer_preserved`, `exclude_flag`, `exclude_reason`. GlotLID follows Kargaran et al., 2023, [arXiv:2310.16248](https://arxiv.org/abs/2310.16248).

### 5.5 Derived splits and estimand mapping

* `-rt` (round trip): back-translate the L test split to English with the NLLB Modal job; publish `sanggusti/{benchmark}-{language}-rt` as English text with the same ids and the canonical schema under the English instruction template. Supports E6.
* `-nllb` (independent translator): translate the English test split with NLLB-200 into L; publish `-nllb`. Supports E6 robustness: E3/E4 recomputed on NLLB test text must agree in sign with the primary result, else the claim is translator specific.
* `-pro1` (digit re-instantiation, gsm8k only): perturb the numeric literals of the English source, recompute the answer deterministically, substitute the same literals into every language by structural replacement (numbers are preserved fields); publish `-pro1` for every language including English with a manifest of the perturbations. Supports E9: fresh numbers defeat memorization.
* `-all` (pooled): concatenate the per-language train splits into `sanggusti/{benchmark}-all` at exactly `train_rows` total, split evenly across the series languages with a seeded selection; record the composition in the card. Supports E5.

### 5.6 Cards, licenses, canary, do-not-train

Each HF repo has a dataset card stating: source benchmark and license, translator and blueprint hash, register, sampling seed, split sizes, quality gate summary, known limitations. Licenses inherit from GSM8K and MedQA sources; NLLB outputs are CC-BY-NC and the card must say so. Each test split contains a canary row with a unique random string and the text `DO-NOT-TRAIN: s01 eval split`. Train configs exclude test IDs by `id` join, verified in gate Step 1.

## 6 Quality gate steps 0-5 with admission rules and thresholds

Run per benchmark per language per split before any finetuning on that data. Thresholds are fixed before any run. Failure blocks admission; see Section 12 for fallback.

* Step 0 calibration: full FLORES-200 devtest per language (NusaX when `flores_code` is null): chrF++ of Adaption and NLLB-200 outputs against references; QE metrics (CometKiwi, MetricX-24) on references, NLLB outputs, and corrupted references with AUROC per corruption type as diagnostics; GlotLID threshold calibrated so 95 percent of references pass; per-sentence rule calibrated on the min references.
* Step 1 hard checks: `struct_ok`; GlotLID per qualifying sentence (length floor, fewer than half numeric or symbol tokens, preserved fields stripped; item-level below three qualifying sentences); length-ratio band; refusal and meta-text detection; untranslated-echo check (chrF++ against the English source below 0.9). Rejections counted per language by cause.
* Step 2 risk rank from coverage-safe signals only: chrF++ agreement with NLLB-200, NLLB back-translation scored against the English source (chrF++ and COMET or MetricX reference mode), `solvable_bt` on the digit re-instantiated back-translation, LID confidence, `leak_id` on content tokens (numerals, preserved fields, proper nouns, and the allowed technical-term list excluded), `borrowed_term_share`.
* Step 3 LLM error-span judge on the bottom risk decile plus a 10 percent random sample, with the English source and few-shot examples in the target language; advisory only.
* Step 4 human review sample: 50 highest-risk plus 100 random rows with ids shared across languages, in error-span annotation format; annotations imported back into `human_verified`, `esa_score`, `answer_preserved`.
* Step 5 admission report per language against fixed rules: at least 0.9x NLLB chrF++ with a paired sentence-level bootstrap and a 30-point floor; own-reference chrF++ above Indonesian-reference chrF++; LID share at least 0.90; structure 100 percent; register consistency at least 0.90 for jv and su; human counts with their upper bounds (0 answer-changed in 100 gives 3.0 percent; 3 major-or-critical in 100 gives 7.6 percent); fallback share at most 5 percent. Item fallback ladder: regenerate once, NLLB-200 retranslation, exclude the id in every language. A JSON report goes into the dataset card.

## 7 Finetuning protocol

Managed finetuning via Adaption AutoScientist, SDK 0.8.0:

1. The first run of the series calls `recommend_hyperparams` on the English gsm8k dataset of the primary base and writes the returned fields into the series file (`hyperparams`, `hyperparams_source`).
2. Every later run passes those hyperparameters verbatim, with `max_iterations`, `target_win_rate`, augmentation rows, `data_format`, and column mapping from the series block. No per-arm tuning.
3. `data_format="instruction"`; augmentation rows are 0 (series constant; changing them starts a new series, since synthetic rows would confound the medium).
4. Fresh `idempotency_key` per replicate on `autoscientist.create`. This parameter exists only on `autoscientist.create`, not on `datasets.run`.
5. `wait_for_completion` on the run; on success compare `best_hyperparams` and `iterations_completed` with the submission and fail loudly on mismatch. Replicates 3 per cell, distinguished by independent runs. Model repos: `sanggusti/{benchmark}-{language}-{condition}-{base}-r{replicate}`. Log `best_win_rate`, iterations used, token counts per language, dataset id, config commit hash, and artifact hash to wandb. `succeeded` means hit `target_win_rate` or exhausted `max_iterations`; always report `best_win_rate`, never success alone.

## 8 Evaluation protocol

A cell is one benchmark by eval language by train condition by base by replicate. Primary evals use the 250-item test subset. Procedure:

1. 20-item smoke: every checkpoint runs `--limit 20` first. The full run is refused unless the smoke log shows parsed answers (non-zero accuracy on en and id, `ANSWER:` lines for choice tasks).
2. Full run: 250 items, greedy decoding (`temperature=0`), `max_tokens=512`, one attempt per item. Parse failures score 0 and increment the parse-failure count.
3. `instruction_variant`: `native` uses the translated `instruction`; `english_cot` reads `instruction_en`. The variant is recorded per eval row. Changing a template after Phase 1 requires a protocol amendment.
4. Scorers are language agnostic: gsm8k numeric match on the `ANSWER:` marker value; medqa choice letter match. No LLM judge in the primary path. Judge scores, if collected, are secondary and reported separately.
5. One log directory per cell under `logs/{series}/{benchmark}/{language}/{condition}/...`, with the series decoding settings and metadata. The same model provider serves base and finetuned cells.

## 9 Covariates

Collected per test item per eval language, stored with eval logs:

* Fertility: mean tokens per word under the eval base tokenizer. High fertility marks fragmentation risk (Petrov et al., 2023, [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)).
* Bits per byte (BPB): base model mean negative log likelihood per byte on the reference text. Measures pretraining exposure more directly than language labels.
* Similarity: lexical similarity of eval language text to English and Indonesian references (chrF++ via sacreBLEU, Post, 2018, [arXiv:1804.08771](https://arxiv.org/abs/1804.08771); identical-content-token share on NusaX parallel sentences, Winata et al., 2022, [arXiv:2205.15960](https://arxiv.org/abs/2205.15960)), excluding numerals, proper nouns, and preserved fields. Used only in E8.
* Reference audit: register classifier over jav_Latn references (ngoko vs krama), diacritic and orthography audit over min_Latn and ace_Latn, plus the `-pro1` manifest flags.
* Human review: `esa_score` and `answer_preserved` per item from the Section 6 step 4 review. Items failing audit are excluded from primary analysis and included in a sensitivity analysis.

## 10 Analysis

Primary family: E3 and E4 per language per benchmark on the primary base, pooled over replicates. All CIs are two sided 95 percent unless stated.

### 10.1 Per-cell intervals

Per-cell accuracy CI from the item bootstrap (Efron (1979, The Annals of Statistics)), 10,000 resamples over the 250 test items, percentile interval. Single-replicate cells additionally report the Wilson interval (Wilson (1927, Journal of the American Statistical Association)). Replicate-pooled cells report the mean over replicates with the two-level bootstrap in 10.2. Never report a point estimate without its interval.

### 10.2 Contrasts

Contrasts use paired items. Three preregistered methods, reported in this order:

1. Two-level bootstrap: resample replicates then items within replicates, 10,000 draws, percentile CI on the contrast. Primary.
2. McNemar exact test (McNemar (1947, Psychometrika)) on discordant pairs pooled over replicates, for binary win/loss significance per contrast.
3. Mixed effects logistic regression with random intercepts for item, item-by-language, and run, fixed effect for condition, via lme4 style specification (Bates et al., 2015, [arXiv:1406.5823](https://arxiv.org/abs/1406.5823)). Used for E8 and for heterogeneity checks.

Multiplicity: Holm correction (Holm (1979, Scandinavian Journal of Statistics)) over the primary family (E3, E4 per language per benchmark). Benjamini-Hochberg correction (Benjamini and Hochberg (1995, Journal of the Royal Statistical Society Series B)) for all secondary and exploratory contrasts. Report both raw and adjusted p values.

### 10.3 Round-trip bound

E6 bound: let `f` be the fraction of test items where `-rt` changes the reference answer or the base model prediction flips between `en` and `rt-en` while the reference is fixed. The translator-loss bound is `|Delta_translator| <= f`, and any `|Delta_en|` or `|Delta_id|` below `f` is labeled translator-indistinguishable. Calibrate `f` against the human review audit on the same items; report both the raw and audit-calibrated bound.

### 10.4 Contamination difference-in-differences

E9 check: for each model, contamination lift is accuracy on the original test minus accuracy on the `-pro1` re-instantiated test. The difference-in-differences is lift(finetuned) minus lift(base). A positive DiD indicates the finetune memorized benchmark content rather than learning the task. If lift is large for any cell, primary results are re-estimated on `-pro1` items as a sensitivity analysis, with the full sample results retained alongside.

### 10.5 Replicate rule from pooled s_r

Let `s_r` be the SD of cell accuracy across the 3 replicates, pooled over cells within a benchmark, with its interval. If pooled `s_r <= 2.0 points`, 3 replicates stand. If `2.0 < s_r <= 3.5`, add 2 replicates to cells in the primary family only. If `s_r > 3.5`, halt and diagnose (training instability or eval nondeterminism) before spending on more replicates. The rule is applied once per benchmark after the first full replicate set.

### 10.6 E7 sign and rank rules

E7 replicates E3 and E4 on the contrast base against the primary base:

* Sign rule: the sign of E3 (and separately E4) pooled over low exposure languages must agree between primary and contrast bases.
* Rank rule: Spearman rank correlation of per-language E3 (and E4) between the two bases must be at least 0.6 with one sided p below 0.05.
* E7 passes only if both rules pass for E3 or both pass for E4, reported separately. Anything else is labeled non-replication, not partial replication.

### 10.7 E8 covariate-adjusted model

E8 fits the 10.2 mixed model plus fixed effects for fertility, BPB, similarity, and audit score, with an exposure rank interaction. Report adjusted E1, E3, E4 with 95 percent CIs and the change from unadjusted estimates. If adjustment flips a primary sign, the headline result is the adjusted estimate and the unadjusted estimate moves to a sensitivity appendix with the covariate balance table.

### 10.8 Exposure plots and premise table

Required figures: (a) E1 versus exposure rank with per-language CIs, (b) `Delta_en` and `Delta_id` forest plots per language, (c) E5 pooled comparison, (d) BPB versus accuracy scatter by language. Premise table: one row per load-bearing premise (parallel content, equal rows, pinned hyperparams, greedy decoding, admission status), each with the artifact or log proving it. A result without its premise row is not reported as primary.

### 10.9 Identification statement

Randomization is over train/test item sampling and replicate runs, not over languages; languages are fixed strata. Causal claims are limited to the effect of train language medium holding content, rows, base, hyperparams, and decoding fixed. Cross-language level comparisons (e.g. raw accuracy in `jv` versus `su`) are descriptive, not causal, because test difficulty differs by language even with parallel content. Contamination, translator error, and tokenizer effects are measured confounders addressed by 10.3, 10.4, and 10.7, not assumed away.

## 11 Phases 0-5 with go/no-go rules

* Phase 0 scoping and covariates (no spend): record `client.autoscientist.list_models()`; run the covariate scripts (fertility, bits per byte, similarity, reference audit) for candidate bases and every S01 language; fill the Phase 0 ledger rows and the premise table. Go: catalogue recorded, covariates computed.
* Phase 1 admission probes (about 8,000 Adaption rows for nine languages): probe `language_expansion` with `estimate=True` for all nine codes; FLORES-200 devtest calibration per language with NLLB-200; 50-row gsm8k probe per language with the full gate. Go: en, id, and at least two regional languages admitted. No-go: revise the blueprint or route and repeat once; languages failing twice are dropped for this series.
* Phase 2 base selection (about 3,800 rows plus GPU hours): translate the 250-item test subsets per admitted language and benchmark under the test-split gate; run the untuned candidate bases on them and on Belebele id, jv, su; apply the 4.1 rules; pin `base_models` by amendment; record the pre-registration commit hash. Go: base pinned, amendment logged before Phase 3.
* Phase 3 full datagen: translate all train and test splits, run gate steps 0-5, complete the human review, publish derived splits. Go: every admitted language passed the gate per benchmark.
* Phase 4 finetuning: the first run of the phase is the plumbing smoke (recommend hyperparams, artifact download, Hub push, serving, Inspect on the artifact); then all arms, 3 replicates. Go per the 10.5 replicate rule. No-go: halt on `s_r > 3.5` per 10.5.
* Phase 5 eval and analysis: full 250-item evals per cell, then Sections 10 and 13. Go: premise table complete, primary family computed with Holm adjustment. No-go: missing premise rows block primary claims; report as exploratory only.

## 12 Fallback ladder, exclusion policy, cost accounting

Item fallback ladder, in order: (1) regenerate the failed items once with a corrected blueprint, (2) retranslate with NLLB-200, (3) exclude the id in every language. Escalation ladder: (1) drop the language for the affected benchmark, (2) drop the benchmark for the affected language only, (3) narrow the series to one benchmark rather than relaxing any threshold. Never skip a rung.

Exclusion policy: item exclusions require an `exclude_reason` from a fixed vocabulary (`numeric_flip`, `choice_flip`, `langid_fail`, `audit_fail`, `contamination`, `parse_template_error`). Language exclusions require a gate report. No post hoc exclusions after seeing E1-E5; exclusions after unblinding are sensitivity-only.

Cost accounting: log per-run Adaption credits, Modal GPU seconds, and wall time in wandb. Phase 4 smoke cost caps full authorization: budget equals measured smoke cost times planned runs times 1.3 contingency. Paid compute is Modal plus Adaption only; Lightning is down and is not a fallback.

## 13 Reporting order

Report in this fixed order: (1) admission table and premise table, (2) E1 native effects with CIs, (3) E3/E4 anchor contrasts with Holm adjusted tests, (4) H1-H4 verdicts, (5) E5 pooled result, (6) E6 translator bound and translator-indistinguishability flags, (7) E7 replication verdict, (8) E8 adjusted estimates with change-from-unadjusted, (9) E9 contamination difference-in-differences, (10) secondary and exploratory results with BH adjustment. Any deviation from this order in the final report requires a dated amendment note.
