# Review of the original plan and the changes made

This document records the plan as it stood when the repository was revived, assesses each element against the literature and the installed tooling, and lists the changes now encoded in `configs/`, `docs/methodology.md` and the pipeline. It exists so that a reader can see what was decided and why without reconstructing the history from commits.

## 1. The plan as stated

- Same evaluations for every language: standard tasks, translated per language, published on the Hub. MGSM and MedQA first, other standard Inspect tasks later.
- Same base model for every condition.
- Same training datasets across tasks, translated per language, enriched with the same Adaption technique.
- One finetuned model per language per task (for example model_1_minangkabau, model_1_javanese, model_1_acehnese), trained with Adaption AutoScientist.
- Goal: find which Indonesian language is the most performant medium, motivated by English anchoring and Chinese chain-of-thought efficiency claims, and by the intuition that Indonesian has a small vocabulary and a simple way to express intent.
- Reproducible for new languages through configs and sweeps.

The previous `docs/research.md` framed the study as cross-lingual transfer (finetune in one language, evaluate in the others) across five domains, 150 evaluation runs, with an LLM-judge quality score as a secondary metric, Cohen's d as an effect size and a paired bootstrap for significance.

## 2. Assessment

### 2.1 What is sound and kept

- Holding base model, task content and pipeline fixed is the right skeleton. Every published cross-language comparison that varies these at once ends up measuring the variation instead of the language.
- Translating the same items into every language gives a paired design. Pairing is what makes small differences detectable; unpaired comparisons of two accuracies at 250 items cannot resolve anything below about 12 points.
- Standard Inspect tasks with language-agnostic scorers (numeric match, option letter, code execution) are the right evaluation instrument. The scorer cannot then favour a language.
- Configs and sweeps as the reproducibility mechanism. The registries and planner now make the matrix a derived object.
- Publishing every translated dataset and every model on the Hub with provenance.

### 2.2 What was at risk

| Element | Risk | Evidence |
|---|---|---|
| "Which language is smarter" as the primary question | With one base model, one translator and one tokenizer, the language is perfectly collinear with pretraining exposure, translator competence and tokenizer fertility. The literature attributes cross-language gaps to those three and has never isolated a residual effect of the language's structure. A plain comparison of A(L) across languages would have been reported as a language effect and would not survive review. | literature note, sections 5.2 and 5.3; layer-swap study 2605.26735; exposure-ceiling results 2312.12683, 2404.11553 |
| The Indonesian premise | Indonesian has almost no inflection in any register, but its derivational morphology is productive in the standard written register the translations use, and current tokenizers spend about 1.2 to 1.6 times as many tokens on it as on English. Human languages converge on similar information rates. The premise would have been an assumption; it is now a set of measurements. | literature note, section 5.2.3 |
| MGSM as the math eval set | MGSM has no split in any language of this study, so its 250 items would be machine-translated anyway, and 250 paired items detect only differences of about 8 points. | methodology, section 10.5 |
| AutoScientist left to optimize per run | Server-side data optimization, hyperparameter adjustment and early stopping judged by an evaluator of unknown language behaviour would make each condition a different recipe. The run object does not even report per-iteration changes. | section 4 below; literature note, section 5.8 |
| Equal examples versus equal tokens | Tokens per example differ by language; the platform derives hyperparameters from row count. Unstated, this would have been an uncontrolled budget difference. | Ahia et al. 2023, 2305.13707 |
| Translation quality | An LLM translating into Acehnese or Minangkabau is weaker than into Indonesian for the same reason the base model is; unreviewed machine translation changes what a benchmark measures; translated eval items with errors can flip cross-language conclusions. There was no gate. | Global MMLU 2412.03304; 2605.24904; NLLB-200 chrF++ into ace about 37 versus id about 69 |
| Register | Javanese and Sundanese speech levels change the lexicon; a translator picks one silently and models are biased toward particular tiers. | 2502.20864; LoraxBench 2508.12459 |
| Minangkabau leakage | About 75 percent lexical overlap with Indonesian; a "Minangkabau" translation can be lightly modified Indonesian and a Minangkabau model can answer in Indonesian while being scored correct. | Koto and Koto 2020, 2009.09309 |
| Contamination asymmetry | English GSM8K and MedQA are in pretraining corpora; the translations are not. The English lead would be partly memorization, and text-overlap checks do not see cross-language contamination. | GSM1k 2405.00332; 2406.13236 |
| Token caps | Tight output caps swing native-versus-English gaps by tens of points. | 2608.04160 |
| Preference pairs as built | chosen = translated answer, rejected = English answer teaches output-language preference, not the task. | `mgsm_convert.py`, `to_preference_dataset` |
| LLM-judge quality score | Multilingual judges overestimate quality in low-resource languages and flip preferences under language switching. | 2607.02235; 2606.14278 |
| Cohen's d | Ill-defined for Bernoulli item scores; the raw accuracy difference with its interval is the effect size. | statistics report |
| Seeds | AutoScientist exposes no seed; run-to-run variance on 1,000-row finetunes is often larger than the effects of interest. | 2002.06305; 2503.07329 |
| Naming | `sanggusti/{domain}-qa-{language}` and `{domain}-{language}-finetuned` cannot name a second benchmark in a domain, a replicate, a base model or a derived split. | AGENTS.md |
| Toolchain drift | The skill references were written for Adaption SDK 0.8.0; the installed SDK is 0.10.0 and the legacy pipeline would have failed at download and silently fallen back to English text on a column-name mismatch. | section 4 below |

### 2.3 Changes made

| Change | Where | Why |
|---|---|---|
| The cross-language ordering is reported as a descriptive comparison of bundles; within-language contrasts (gain, anchor advantage, regression, pooling) are the primary causal results | methodology 1.4 | only these are identified with one translator and one tokenizer |
| Base model becomes a factor: a primary base plus a size-matched contrast base with a different exposure profile, on gsm8k, in tier 1 | series file `base_models`; methodology 4.1 | the minimum for any statement that survives "that is just the corpus" |
| gsm8k replaces mgsm: the full GSM8K test is translated, the MGSM items are tagged | `configs/benchmark/gsm8k.yaml` | power: 3.4 versus 8 points minimum detectable difference |
| Round-trip condition and `-rt` splits, back-translated by NLLB-200 | series file; methodology 5.5 | bounds translator loss from above with an independent back-translator and the model never seeing the language |
| Digit re-instantiation `-pro1` split for gsm8k, evaluated for base and native models | series file; methodology 5.5 | contamination control that keeps the item pairing |
| Locale-tolerant numeric scorer alongside the strict one | `translated_benchmark.py`; methodology 4.3 | an untuned model answering in the Indonesian numeral convention is not scored wrong for formatting |
| Register sentence substituted into every blueprint | `configs/language/*.yaml`; `configs/benchmark/*.yaml` | the pinned register reaches the translator |
| Pooled and Indonesian-anchor arms | series file, tier 2 | the literature's strongest prediction (pooled beats native) and the regional-pivot question |
| AutoScientist pinned: one iteration, no early stop, no augmentation, hyperparameters from one recommendation copied into every run, LoRA, loss on inputs, raw upload | series file `autoscientist`; methodology 7 | ceteris paribus |
| Replicates instead of seeds, count set by a variance pilot | series file `replicates`; methodology 10.5 | the SDK has no seed; variance must be measured |
| Translation gate with calibration, per-item quality columns, second translator, human error-span review, admission thresholds and fallbacks | methodology 5.4 and 6 | translation quality is correlated with the treatment |
| Register pinned per language; leakage check for Minangkabau | `configs/language/*.yaml`; methodology 4.2, 6 | silent register choice and Indonesian leakage |
| Answer protocol enforced identically: literal `ANSWER:` marker, English numerals | methodology 4.3 | the installed scorers require it |
| Generous token cap with cap hits logged; accuracy at two caps | series file `generation`; methodology 8 | cap effects |
| No LLM judge in primary metrics; Cohen's d dropped; paired item bootstrap and a mixed-effects model with item and run random effects; Holm on a small primary family | methodology 10 | statistics of paired binary outcomes |
| Cost-ordered phases with go/no-go rules and a minimum publishable unit | methodology 11 | a partial budget still yields a complete result |
| Naming with benchmark, series, base and replicate; derived-split suffixes | series file `naming`; AGENTS.md | the old scheme could not name the matrix |
| Preference tuning moved to series S03; adaptive optimizer to S02 | methodology 7 | keep the primary series interpretable |
| The premise about Indonesian turned into measured quantities | methodology 1.4, 2, 10.6 | no design assumption rests on it |

### 2.4 What was deliberately not done

- The full off-diagonal transfer matrix (finetune in L, evaluate in every L') is defined as a condition but not active. Each off-diagonal cell is an evaluation only, so it can be bought later without new finetunes; it does not bear on the primary question.
- Five domains at once. Two benchmarks with different scorer types (numeric, letter) give the task-type replication; the registry makes a third benchmark one file.
- A native-script arm. All languages are evaluated in Latin script; current models are near zero on Javanese and Sundanese native scripts and FLORES uses Latin for these languages.
- Human translation of full sets. The gate and the verified subsets are the affordable substitute; where reviewers are unavailable (Minangkabau, Acehnese), the language stays descriptive.

## 3. Reproducibility for new languages

The plan asked for the study to extend to other languages with little effort. That is now: one registry file per language (`configs/language/{code}.yaml`), the code added to the series list, and the planner derives datasets, finetunes, evaluations, names and commands. Later Indonesian languages (Balinese, Banjar, Buginese, Madurese) differ only in their calibration references (Madurese is not in FLORES-200 and uses NusaX) and in which quality-estimation metrics can be enabled. The same holds for benchmarks: one registry file, with the rule that the scorer is language-agnostic. [reproducibility.md](reproducibility.md) has the steps.

## 4. Toolchain audit

The installed Adaption SDK is 0.10.0; the skills and the legacy pipeline were written against 0.8.0. Drift found by reading the installed package, and what was done about it:

| Item | Installed behaviour | Action |
|---|---|---|
| `datasets.download` | returns a binary body (`BinaryAPIResponse`), not a URL; parquet arrives as a tar of shards | `mgsm_convert.py` now writes the body to a file; skills updated |
| Dataset status | gains `awaiting_input`, which is not terminal, so `wait_for_completion` right after upload blocks until timeout | ingestion is polled on `row_count`; skills updated |
| Status errors | the field is `error_data`, not `error`, so the old check never fired | fixed in the converter |
| Adapted download columns | carry `enhanced_prompt` and `enhanced_completion`; the old `.get(question, original)` fallback would have silently written English text as the translation | the converter maps the enhanced columns and raises on anything else; tests added |
| `training_models.list()` | deprecated alias of `autoscientist.list_models()` | AGENTS.md and skills updated |
| Run object | `run.id`, not `run.experiment_id`; no per-iteration loss, learning rate or gradient norm; `best_hyperparams` is what the artifact was trained with | skills updated; the design compares `best_hyperparams` with the submission |
| `autoscientist.download` | binary gzip body | skills updated |
| `datasets.publish` | returns 501 | keep `push_to_hub` |
| `processing_mode="raw"` | upload path that skips server-side augmentation | adopted for finetuning data |
| Hyperparameters | fifteen fields; `training_type` and `train_on_inputs` are otherwise decided by the platform | pinned in the series file |
| `language_expansion` | validates ISO 639-1 codes; min and ace have only ISO 639-3 codes | probe with `estimate=True`; blueprint route as the default |
| Inspect `match(numeric=True)` | parses `1.500` as 1.5 and `2,5` as 25 | English numeral formatting enforced by the gate |
| Inspect `parse_answers` | requires the literal `ANSWER:` marker | kept in every translated instruction |

The legacy converter (`pipeline/datagenerator/evals_translate/mgsm_convert.py`) remains as the reference for the Adaption calls; the generic stage modules that replace it are listed in [reproducibility.md](reproducibility.md) and tracked in [experiments.md](experiments.md).

## 5. Second review: is this a good research topic?

A critical re-read of the design after it was merged, asking not whether it is sound but whether it is worth running. The verdict and the changes it produced.

### 5.1 Verdict

The study is a good applied and resource topic and a weak topic for the question as originally phrased. Section 1.4 of the methodology already concedes that a "smarter medium" claim is not estimable with one language per structural profile, and no six-language design can make it so. What the study can answer is the practitioner's choice for each target language, translate, anchor to English, pivot through Indonesian, or pool, and it can publish the gated translations and covariate tables that nobody has for these languages. The first version led with the question it retires; this version leads with the question it can answer.

### 5.2 Findings

| Finding | Evidence | Consequence |
|---|---|---|
| The exposure ordering (H1 in the first version) is already established | IndoMMLU, NusaX, Belebele, every regional-language benchmark to date | demoted to a descriptive prediction (D1); no budget is ordered around it |
| The English-anchor advantage has a known direction and an unknown size; the expected size for id, jv and su is about three points, below the pre-registered detectable difference at three replicates | literature note 5.1.3; methodology 10.5 | an equivalence bound of 3 points is pre-registered so a null is a claim; "equivalent" is the stated most likely outcome for those languages |
| Pooled beats native is established for other families | MathOctopus; equal-budget Alpaca; the 220-run study | kept as replication, bought last (tier 2) |
| The Indonesian pivot is the one contrast specific to this family that no prior paper answers, and it costs evaluations only because the Indonesian native model exists in tier 0 | literature note 5.1.2 and 5.3; the planner | promoted to tier 0 and to the primary family; H1 in the revised numbering |
| Floor effects: a 4B LoRA model on Acehnese math, and on option-letter medicine in any regional language, will sit near floor, so two of six languages and one of two benchmarks buy little information at full price | NLLB chrF++ into ace about 37; base near chance on medqa in regional languages | ace parked at tier 2 behind the gate and the floor rule; medqa parked at tier 3 behind a Phase 2 decision; the floor rule extended to min with size as the lever |
| The managed finetuning service is the treatment: no seed, undisclosed optimizer, undocumented evaluator | SDK audit, section 4 | a transparent LoRA loop with fixed seeds repeats the en and id cells (sft_check, H6); a deviation makes every managed result conditional on the platform |
| MedQA teaches the answer format, not the medium: a thousand letter-answer rows, a base near chance in regional languages, and a borrowing policy that keeps the terms Indonesian or Latin | methodology 4.3 | second benchmark decided in Phase 2 against the criteria of benchmarks.md section 4, with an English-only finetuning probe that costs no translation |
| Cost against information: sixty finetunes, over two hundred evaluations, roughly fifteen thousand translation rows and about three thousand human annotations, mostly to confirm expected orderings | experiments.md ledger | tiers reordered so the novel contrasts are bought first; the minimum publishable unit is 15 finetunes and 85 evaluations |
| The two lowest-exposure languages are unverifiable where it matters most (no quality-estimation coverage, possibly no reviewers) | literature note 5.6 | stated as a limitation; the causal study is effectively en, id, jv, su and min |

### 5.3 Changes made

| Change | Where |
|---|---|
| Question reframed as the practitioner choice; the "simpler medium" premise kept as a measured side table | research.md; methodology 1.1; README.md; docs/README.md |
| Hypotheses reordered by novelty: pivot, anchor, pooling; then translator loss, exposure invariance, platform independence; exposure ordering and similarity demoted to descriptive | methodology 2 |
| Equivalence verdicts (TOST, 3 points) on every primary contrast; the bound is a series constant | methodology 2, 10.3, 10.5; series file `analysis` |
| Primary family is Delta_en plus Delta_id, Holm per benchmark | methodology 10.3 |
| Tiers compose over condition, benchmark and language through the series `tiers` block; indonesian_anchor to tier 0, ace to tier 2, medqa to tier 3 | series file; `src/utils/registry.py`; `pipeline/plan.py` |
| Training backend as a condition key; the `sft` backend with fixed seeds, the `-sft` model slug, the `sft_check` condition in tier 1 | series file; registry; planner; `configs/training/sft.yaml`; methodology 7 |
| Floor rule on id and min with size as the lever; ceiling on en | methodology 4.1; series file `analysis.floor` and `analysis.ceiling` |
| Phases reordered: gsm8k tier 0 first (H1, H2), then the platform and contrast-base checks, then Acehnese and pooling, then the second benchmark | methodology 11; research.md 7; experiments.md |
| Second-benchmark criteria and candidates | benchmarks.md 4 |

### 5.4 What the second review did not change

The item pairing, the pinned recipe, the gate, the round-trip bound, the contamination split, the replicate rule and the pre-registration discipline all stand; they are stronger than most papers in this area. The registry-and-planner structure absorbed every change above without a new stage module: parking a benchmark or a language is one line in the series file.
