# Research proposal: which language should be the medium of finetuning for Indonesian and its regional languages?

## Summary

Teams building models for Indonesian and its regional languages face one recurring choice: translate the finetuning data into the target language, keep it in English, translate it into Indonesian as a pivot, or pool the languages. The literature answers this for European and major Asian languages and for English pivots; it says nothing about Indonesian as a pivot for the languages that share much of their lexicon with it, and it measures neither translation quality nor pretraining exposure per item. This project takes the same task data, translates it into Indonesian, Javanese, Sundanese, Minangkabau and Acehnese, finetunes one base model under each option with the same pinned recipe, and evaluates every model on the same items in the target language.

The design is built so that the four options can be compared as causal contrasts within each language, paired on items and replicated over finetuning runs, and so that the cross-language ordering of accuracy can be read net of the three factors the literature says drive cross-language gaps: the base model's pretraining exposure to each language, the competence of the translator in each language, and the tokenizer's fertility. Those factors are measured per language and per item, held fixed where the tooling allows, and separated from the language where the design allows (a second base model with a different exposure profile, a round-trip translation bound, a digit re-instantiation split for contamination, and a transparent finetuning loop beside the managed one). The outputs are a practitioner answer per language with difference and equivalence verdicts, one descriptive cross-language comparison with its covariates attached, and a published set of gated, item-aligned benchmark translations for languages that have none.

Full design: [methodology.md](methodology.md). Literature: [research/5_language_as_medium.md](research/5_language_as_medium.md). Evaluation of the original plan and the second review: [plan_review.md](plan_review.md). How to run and extend it: [reproducibility.md](reproducibility.md).

## 1. Question

Holding base model, task content and pipeline fixed, which language of finetuning data gives the best accuracy when the model is evaluated in Indonesian or one of its regional languages: the target language itself, English, Indonesian as a pivot, or a pool of all of them? Does the answer depend on the language, and how much of the cross-language ordering of accuracy is explained by exposure, translation quality and tokenization?

Formally, with B(L) the untuned base accuracy in language L, A(L) the accuracy after finetuning in L, A_en(L) after finetuning in English, A_id(L) after finetuning in Indonesian and A_all(L) after finetuning on the pool, the study estimates the within-language contrasts Delta_id(L) = A_id(L) - A(L), Delta_en(L) = A_en(L) - A(L), G(L) = A(L) - B(L), A_all(L) - A(L) and the English regression Reg(L), and the cross-language contrasts tau(L, L') = A(L) - A(L'). The methodology document states which of these are identified as causal effects and which are descriptive of this base model, translator and tokenizer.

The user-level framing, whether one of these languages is a "smarter" medium and whether Indonesian's supposedly small vocabulary and simple expression of intent helps or hurts, is not estimable with one language per structural profile. It is kept as a set of measurements reported in a side table: tokens per item, characters per item, bits per byte under the base model, and the position of Indonesian relative to what its exposure and fertility predict.

## 2. Why this is worth doing

- The pivot question is open. To our knowledge, no published study asks whether Indonesian is a better pivot than English for Javanese, Sundanese, Minangkabau or Acehnese, although Minangkabau shares about three quarters of its lexicon with Indonesian and Indonesian has hundreds of times more web text than any of them. The English-pivot literature (PLUG, question alignment, cross-lingual-thought prompting) never tests a related high-resource language as the pivot.
- The closest controlled comparisons vary the language of tuning data for European and major Asian languages; none covers Indonesian regional languages, none measures translation quality and exposure per item, and none carries a translator-loss bound. Section 5.5 of the literature note shows that no language in the set other than Indonesian has a native evaluation set for medicine, code or science, and that for math only a 500-item MATH-style set exists for Javanese, Sundanese and Buginese (MATH-IDN); the translated, gated sets this study publishes are a contribution on their own.
- Practical value: teams building Indonesian and regional-language models need to know whether to translate their finetuning data, anchor to English, pivot through Indonesian, or pool. The within-language contrasts, each with an equivalence verdict, answer that directly, and a null for the English anchor on Indonesian is itself the answer "translation is not costing you accuracy".
- What is not the contribution: the ordering of accuracy by exposure (en above id above the regional languages) is predicted by every regional-language benchmark to date, and pooled tuning beating monolingual tuning is established for other families. Both are reported, neither is the headline, and the budget is ordered so that the novel contrasts are bought first.

## 3. Hypotheses

Stated with the observation that would count against each in the methodology (section 2). Every primary contrast is tested for difference and for equivalence within 3 accuracy points.

- H1, Indonesian pivot: finetuning in Indonesian is at least as accurate as native finetuning for every regional language, and better than English anchoring for the languages closest to Indonesian.
- H2, English anchor: English-anchored finetuning is at least as accurate as native finetuning on math for every non-English language; for Indonesian, Javanese and Sundanese the expected gap lies inside the equivalence bound.
- H3, pooling: one model finetuned on the pooled languages is at least as accurate as the native model in every language at equal total rows.
- H4, translator loss: the round-trip translator loss is positive for the lowest-exposure languages; the part of the native gap above it is not attributable to the translator.
- H5, exposure invariance: the language ordering is invariant to a size-matched base model with a different exposure profile; if not, the ordering is model-specific and the covariate table says which component (corpus, tokenizer, post-training) moves with it.
- H6, platform independence: a plain LoRA loop with fixed seeds reproduces the managed finetuning results on English and Indonesian within the replicate interval; if not, every managed-backend result is conditional on the platform.
- Descriptive, not tested: D1, base and native accuracy are ordered by pretraining exposure, en >= id > jv, su > min, ace; D2, among regional languages, gains track lexical similarity to Indonesian. Premise P: the intuition that Indonesian is a simpler medium is reported as measurements; under current tokenizers standard written Indonesian is expected to cost more tokens per item than English.

## 4. Design in brief

![Experiment matrix for series S01](diagrams/language_medium_design.png)

*Figure 1. Series S01. Rows are finetuning conditions, columns are evaluation languages. Every cell uses the same source items translated per language, the same base model, the same pinned finetuning recipe and a language-agnostic scorer.*

| Element | Choice |
|---|---|
| Languages | en (reference), id, jv, su, min in tier 0; ace in tier 2 after the gate and the floor rule; more through one registry file each |
| Benchmarks | gsm8k (GSM8K test, 1,319 items, MGSM subset tagged) in tier 0; the second benchmark is decided in Phase 2 against stated criteria, with medqa (MedQA test, 1,273 items) as the placeholder parked at tier 3; more through one registry file each, restricted to language-agnostic scorers |
| Training data | fixed 1,000-item subset of each benchmark's training source, identical ids in every language |
| Translation | Adaption Adaptive Data with a fixed blueprint and register sentence per language; a six-step gate with FLORES-200 and NusaX calibration, per-item quality columns, NLLB-200 as the independent second translator and back-translator, and a human error-span review |
| Finetuning | Adaption AutoScientist with the optimizer pinned: one iteration, no early stop, no augmentation, hyperparameters from one recommendation call copied into every run, raw upload of the published rows; three replicates per finetuned cell; a transparent LoRA loop with fixed seeds repeats the en and id cells |
| Evaluation | one parameterized Inspect task, numeric match or option letter, temperature 0, generous token cap, output-language fidelity and token counts logged |
| Conditions | tier 0: base, native, english_anchor, indonesian_anchor, regression, round_trip, and the base and native cells on the digit re-instantiated split; tier 1: sft_check and the native and english_anchor cells on a contrast base model; tier 2: ace and pooled; tier 3: the second benchmark |
| Analysis | paired item bootstrap and a mixed-effects logistic model with item, item-by-language and run random effects; Holm on the primary family (Delta_en and Delta_id) with equivalence verdicts; identification analyses for the base-model factor and item-level covariates |
| Size | tier 0 is 15 finetunes and 85 evaluation runs; the full series is 66 and 261 |

## 5. Contributions

1. To our knowledge, the first test of Indonesian as a finetuning pivot for its regional languages, inside an item-paired, pipeline-controlled comparison of finetuning language across six languages, with exposure, fertility and translation quality measured rather than assumed.
2. Gated, item-aligned translations of GSM8K (and of the second benchmark once chosen) into id, jv, su, min and ace, published with their gate scores, human-verified subsets and excluded-item lists, plus round-trip and NLLB-200 comparison splits.
3. Per-language covariate tables (tokenizer fertility and bits per byte under current base models) for jv, su, min and ace, which no publication reports.
4. Two eval-only controls and one training-side control that are reusable elsewhere: the round-trip bound that charges loss to the translator without the model ever seeing the language, a digit re-instantiation split that tests English contamination while keeping the item pairing, and a transparent finetuning loop that bounds what a managed finetuning service contributes to a result.
5. A config-driven pipeline in which adding a language or a standard Inspect benchmark is one file, tiers compose over conditions, benchmarks and languages, and the full sweep is derived by a planner before any paid run.

## 6. Relation to prior work

![Latent English in the middle layers of Llama-2](research/figures/latent_english_logitlens.png)

*Figure 2. Logit-lens decoding of Llama-2 on non-English prompts: middle layers favour the English token before the target-language token surfaces. English anchoring exploits this; the present study asks whether Indonesian plays the same role for its regional languages, and whether either holds for languages with near-zero exposure. Source: Wendler et al., 2024, arXiv:2402.10588.*

English-anchored reasoning (MGSM's English chain of thought, cross-lingual-thought prompting, self-translation, pivot-language instruction tuning, question alignment) is the baseline that native finetuning must beat, and it is the english_anchor condition here; the indonesian_anchor condition asks the same question with a related pivot. Translate-train studies (MathOctopus, Bactrian-X, Okapi, Aya) and the equal-budget comparisons of monolingual against multilingual tuning motivate the pooled arm. The exposure-ceiling results (a pinch of multilinguality is enough for behaviour, but pretraining exposure bounds what instruction tuning can reach) motivate treating exposure as the primary confound and the base model as a factor. Work on Indonesian regional languages (NusaX, NusaWrites, Cendol, Komodo, Sahabat-AI, IndoMMLU, LoraxBench, MATH-IDN) supplies the resources for calibration and the evidence that the regional gap is large and register-sensitive. The statistics of evaluation (paired differences, clustered errors, small-sample coverage, seed variance, contamination across languages) fix the analysis plan.

![Lexicon overlap of Indonesian regional languages with Indonesian and English](research/figures/nusawrites_lexicon_overlap.png)

*Figure 3. Lexicon overlap with Indonesian (x) and English (y) for Indonesian regional languages across Wikipedia, NusaParagraph and NusaTranslation. Overlap with Indonesian is why the pivot arm exists, why the design measures lexical similarity per language in Phase 0, and why it carries an Indonesian-leakage check for every regional language. Source: Cahyawijaya et al., 2023, arXiv:2309.10661, repository visualization.*

What none of the prior work does is hold base model, items, translator and finetuning recipe fixed across a set of closely related low-resource languages while measuring the confounds, or test the related high-resource language as the pivot. The novelty is the design and the pivot question, not a new method.

## 7. Program

| Phase | Spend | Output | Decides |
|---|---|---|---|
| 0 | none | covariate tables, premise numbers, live model catalogue with per-size cost | candidate base shortlist |
| 1 | about 5,300 translation rows | FLORES-200 devtest calibration and 50-row probes, gate scores | which languages are admitted |
| 2 | about 1,250 translation rows plus GPU hours | 250-item gsm8k subsets, base accuracy in every admitted language, Belebele probes, English-only probes for the second-benchmark candidates | base size against the floor rule; the second benchmark |
| 3 | full gsm8k translation for the tier 0 languages | gated test and train splits, derived splits, verified subsets, base cells | languages that stay admitted |
| 4 | 15 one-iteration finetunes on gsm8k | tier 0, the minimum publishable unit: H1 and H2; replicate variance | replicate count |
| 5 | 6 transparent LoRA runs and 15 contrast-base finetunes | H6 platform independence; H5 exposure identification | whether the ordering is model-specific |
| 6 | Acehnese if admitted, pooled models | tier 2: the lowest-exposure language; H3 | |
| 7 | the second benchmark; series S02 to S04 | task-type replication; adaptive optimizer; preference tuning; more benchmarks and languages | |

Series S02 lets AutoScientist optimize (three iterations) on the tier 0 cells to ask whether the platform's optimizer changes the ranking. Series S03 is preference tuning with cross-lingual consistency pairs, the original direction of this repository. Series S04 adds benchmarks (GSM8K variants, MMLU through its multilingual layout, HumanEval-XL) and the second language wave (Balinese, Banjar, Buginese, Madurese).

## 8. Limitations, ethics and what the study cannot claim

With one language per structural profile, the study cannot attribute a residual gap to the language's structure, and it does not try to. It can say which finetuning language is best for each target language under this base model and translator, how much of the cross-language gap is at most translator loss, whether the ordering survives a change of base model, whether the managed finetuning service reproduces on a transparent loop, and how much of the gap item-level translation quality and token counts explain. Every S01 effect is an effect under LoRA at the pinned rank; the optional full-finetuning pair bounds that conditionality. The two lowest-exposure languages may be unverifiable by human review, in which case they remain descriptive. The report states all of this in one paragraph next to the results.

Ethics and data statement. The medical datasets, if the second benchmark stays medical, are machine translations of exam items and the models finetuned on them are research artifacts; both carry a clinical-use disclaimer on their cards. Annotators are native speakers recruited for the error-span review, paid at a stated rate, credited as they prefer, and consent to the release of their annotations with the guideline. Publishing translated test items creates future contamination of these benchmarks; every test split embeds a canary string and the cards ask downstream users not to train on them. The translator is a managed service whose underlying model is not disclosed; run ids, dates and the blueprint are recorded so the translation step is reproducible as a procedure even if the model behind it changes. Source licenses (GSM8K, MedQA, FLORES-200, NusaX, NLLB outputs under CC-BY-NC) are carried into every derived artifact.
