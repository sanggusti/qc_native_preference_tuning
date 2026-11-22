# 5. Language as Medium

Survey supporting series S01 (`s01_language_medium`): holding base model, task content, and pipeline fixed, does the language used as the finetuning and eval medium change accuracy? Companion documents: [1](1_intro_problem_statements.md), [2](2_preliminaries_fundamentals.md), [3](3_contexts_methodologies.md), [4](4_datasets_evaluation.md).

## 1 The question

Most multilingual results vary language and content together: different test sets, different training corpora, different tokenizers per language. S01 instead fixes parallel content (identical items translated into en, id, jv, su, min, ace, ban, bjn, bug) and varies only the medium of training and evaluation. The estimands E1-E9 in `docs/research.md` are all contrasts on this fixed-content grid.

Two benchmarks carry the question: GSM8K for reasoning (Cobbe et al., 2021, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)) and MedQA for knowledge-heavy MCQ (Jin et al., 2020, [arXiv:2009.13081](https://arxiv.org/abs/2009.13081)). MGSM provides the parallel-math template and the 250-item precedent (Shi et al., 2022, [arXiv:2210.03057](https://arxiv.org/abs/2210.03057)). A standing confound for any translated benchmark is Western-centric content (Singh et al., 2024, [arXiv:2412.03304](https://arxiv.org/abs/2412.03304)); S01 mitigates it by keeping content identical across media so the confound cancels in within-item contrasts, and by bounding translator loss explicitly (E6).

## 2 Prior work asking the same question

Translate-train versus anchor is the direct precedent. MathOctopus translates math instruction data into target languages and shows multilingual SFT beats English-only SFT for multilingual math (Chen et al., 2023, [arXiv:2310.20246](https://arxiv.org/abs/2310.20246)). QAlign then beats translate-train by 11-16 points on MGSM/MSVAMP by aligning questions to English and doing reasoning-side tuning only in English, sidestepping noisy translated chain-of-thought (Zhu et al., 2024, [arXiv:2401.07817](https://arxiv.org/abs/2401.07817)). The S01 native versus english_anchor contrast (E3) is this comparison restated as a registered estimand at fixed data budget.

Adjacent anchor methods: PLUG trains pivot-first generation (English draft, then target response) with large instruction-following gains (Zhang et al., 2023, [arXiv:2311.08711](https://arxiv.org/abs/2311.08711)); xCoT uses code-switched tuning with random pivoting (Chai et al., 2024, [arXiv:2401.07037](https://arxiv.org/abs/2401.07037)); MAPO converts cross-lingual reasoning consistency into DPO preference pairs without target-language annotation (She et al., 2024, [arXiv:2401.06838](https://arxiv.org/abs/2401.06838)). Zero-training anchors set the floor any trained anchor must beat: XLT prompting gains 10+ points on arithmetic reasoning (Huang et al., 2023, [arXiv:2305.07004](https://arxiv.org/abs/2305.07004)), and self-translate shows the capability is already inside the model but not elicited by native prompts (Etxaniz et al., 2023, [arXiv:2308.01223](https://arxiv.org/abs/2308.01223)). S01 bounds translator loss with round-trip and independent-translator splits (E6).

## 3 Language of thought

If models reason in a latent English-biased concept space, English-medium training should dominate native training for reasoning tasks even when evaluation is native. Evidence for an internal pivot: intermediate layers decode the English token variant before the target-language token (Wendler et al., 2024, [arXiv:2402.10588](https://arxiv.org/abs/2402.10588)). The MWork account describes a sandwich structure in which early and late layers handle language-specific form while middle layers compute over shared concepts (Zhao et al., 2024, [arXiv:2402.18815](https://arxiv.org/abs/2402.18815)). Document 2 surveys these mechanisms in full; for S01 they motivate H2 directionally (anchor advantage on gsm8k) and predict a smaller or reversed effect on medqa, where the middle layers cannot supply knowledge the model never saw in any language.

## 4 Language difficulty

Medium effects have a mechanical component: the same content costs more tokens, more compute, and longer contexts in some languages than others. Tokenizer fertility (tokens per word or per byte) is systematically worse for low-resource languages, raising cost and latency at identical content length and shrinking effective context (Petrov et al., 2023, [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)). S01 logs per-item fertility and character length as covariates for E8 so that a medium gap attributable to tokenization can be separated from a gap attributable to reasoning or knowledge. Benchmark translation must also preserve numeric and symbolic content exactly; formatting noise in translated math is the mechanism QAlign identifies behind translate-train failure, and the S01 datagen QA gates target it directly.

## 5 Indonesian regional resources

### 5.1 NusaX

NusaX is a parallel sentiment and emotion corpus covering Indonesian plus ten regional languages, including all seven S01 regional languages (jv, su, min, ace, ban, bjn, bug) (Winata et al., 2022, [arXiv:2205.15960](https://arxiv.org/abs/2205.15960)). Role in S01: parallel sentences for the lexical similarity covariate and the FLORES fallback where `flores_code` is null, not evaluation (sentiment labels do not transfer to math or medical QA). Related corpora: NusaWrites for underrepresented-language text (Cahyawijaya et al., 2023, [arXiv:2309.10661](https://arxiv.org/abs/2309.10661)) and Cendol for Indonesian instruction tuning (Cahyawijaya et al., 2024, [arXiv:2404.06138](https://arxiv.org/abs/2404.06138)).

### 5.2 FLORES-200

FLORES-200 is the massively parallel translation benchmark released with NLLB, the basis for calibration and Belebele passages (NLLB Team, 2022, [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)). Per-language availability for Minangkabau, Acehnese, Balinese, Banjarese, and Buginese must be verified before use; where a code is missing, S01 falls back to NusaX. Role in S01: calibration set for translator quality checks, not a training source.

### 5.3 Belebele

Belebele is a parallel reading-comprehension benchmark in 122 language variants, covering Indonesian, Javanese, and Sundanese but not the lower-resource S01 languages (Bandarkar et al., 2023, [arXiv:2308.16884](https://arxiv.org/abs/2308.16884)). Role in S01: an optional held-out comprehension check for id/jv/su; it cannot cover the rest, which is part of why S01 translates its own benchmarks.

### 5.4 GlotLID

GlotLID is a language identifier covering 2000+ languages with emphasis on low-resource coverage (Kargaran et al., 2023, [arXiv:2310.16248](https://arxiv.org/abs/2310.16248)). Role in S01: automated language-ID gate on every translated row (reject or flag rows the classifier does not attribute to the target language).

### 5.5 NLLB coverage

NLLB-200 provides translation models and parallel data across 200 languages (NLLB Team, 2022, [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)). S01 assumes coverage of ind, jav, sun, min, ace, ban, bjn, and bug; every code must be verified against the released NLLB-200 language list in the #52 smoke run before the gate depends on it, with NusaX and adaptation-only columns as fallback where a code is missing. Role in S01: second-opinion translator, back-translator for the E6 bound, and the `-nllb` split producer.

### 5.6 Translation quality: MQM and error-span annotation

S01 needs translation quality as a measured covariate, not an assumption. The MQM framework (Freitag et al., 2021, [arXiv:2104.14478](https://arxiv.org/abs/2104.14478)) provides a standard error taxonomy (accuracy, fluency, terminology, style) with severity weighting; ESA simplifies expert annotation to error-span marking with an overall score (Kocmi et al., 2024, [arXiv:2406.11580](https://arxiv.org/abs/2406.11580)). S01 procedure: ESA-style spot annotation on a fixed random sample per translated dataset (minimum 50 items), scored by a strong judge model with a frozen prompt, plus sacreBLEU against back-translation as a cheap automatic diagnostic (Post, 2018, [arXiv:1804.08771](https://arxiv.org/abs/1804.08771)). Reference-free quality estimation removes the back-translation step (CometKiwi, Rei et al., 2022, [arXiv:2209.06243](https://arxiv.org/abs/2209.06243); MetricX-24, Juraska et al., 2024, [arXiv:2410.03983](https://arxiv.org/abs/2410.03983); COMET, Rei et al., 2020, [arXiv:2009.09025](https://arxiv.org/abs/2009.09025)). Per-item quality scores feed E8 as covariates; datasets below the registered MQM threshold are retranslated, not patched.

### 5.7 Statistics for small multilingual contrasts

S01 test sets are 250 items per language per benchmark, and most claims are paired within-item contrasts. The registered toolkit:

- **Item bootstrap CIs** for accuracy differences, resampling items with replacement (Efron, 1979, Annals of Statistics).
- **Wilson score intervals** for single-condition accuracies (Wilson, 1927, Journal of the American Statistical Association).
- **McNemar test** for paired condition contrasts on identical items (McNemar, 1947, Psychometrika).
- **Mixed-effects logistic regression** with random intercepts for items (and replicates), fit with lme4 (Bates et al., 2015, Journal of Statistical Software), for covariate-adjusted E8 estimates.
- **Holm correction** for the small family of registered contrasts, **Benjamini-Hochberg** for exploratory per-language scans (Holm, 1979, Scandinavian Journal of Statistics; Benjamini and Hochberg, 1995, Journal of the Royal Statistical Society Series B).
- **TOST equivalence testing** when the claim is a null or near-null difference (for example, pooled matching native within a margin), since a non-significant p-value alone does not establish equivalence (Schuirmann, 1987, Journal of Pharmacokinetics and Biopharmaceutics).

Replicate variance (3 training runs) is reported as a variance component, not averaged away silently: run sensitivity is itself a finding about low-resource finetuning stability.

## References

- Cobbe et al., 2021. Training Verifiers to Solve Math Word Problems. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- Jin et al., 2020. What Disease Does This Patient Have? (MedQA). [arXiv:2009.13081](https://arxiv.org/abs/2009.13081)
- Shi et al., 2022. Language Models are Multilingual Chain-of-Thought Reasoners (MGSM). [arXiv:2210.03057](https://arxiv.org/abs/2210.03057)
- Singh et al., 2024. Global MMLU. [arXiv:2412.03304](https://arxiv.org/abs/2412.03304)
- Chen et al., 2023. Breaking Language Barriers in Multilingual Mathematical Reasoning (MathOctopus). [arXiv:2310.20246](https://arxiv.org/abs/2310.20246)
- Zhu et al., 2024. Question Translation Training (QAlign). [arXiv:2401.07817](https://arxiv.org/abs/2401.07817)
- Zhang et al., 2023. PLUG: Pivot Language Guided Generation. [arXiv:2311.08711](https://arxiv.org/abs/2311.08711)
- Chai et al., 2024. xCoT: Cross-lingual Instruction Tuning for Cross-lingual CoT. [arXiv:2401.07037](https://arxiv.org/abs/2401.07037)
- She et al., 2024. MAPO: Multilingual-Alignment-as-Preference Optimization. [arXiv:2401.06838](https://arxiv.org/abs/2401.06838)
- Huang et al., 2023. Cross-Lingual-Thought Prompting (XLT). [arXiv:2305.07004](https://arxiv.org/abs/2305.07004)
- Etxaniz et al., 2023. Do Multilingual Language Models Think Better in English? [arXiv:2308.01223](https://arxiv.org/abs/2308.01223)
- Wendler et al., 2024. Do Llamas Work in English? [arXiv:2402.10588](https://arxiv.org/abs/2402.10588)
- Zhao et al., 2024. How do Large Language Models Handle Multilingualism? (MWork). [arXiv:2402.18815](https://arxiv.org/abs/2402.18815)
- Petrov et al., 2023. Language Model Tokenizers Introduce Unfairness Between Languages. [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)
- Winata et al., 2022. NusaX. [arXiv:2205.15960](https://arxiv.org/abs/2205.15960)
- Cahyawijaya et al., 2023. NusaWrites. [arXiv:2309.10661](https://arxiv.org/abs/2309.10661)
- Cahyawijaya et al., 2024. Cendol. [arXiv:2404.06138](https://arxiv.org/abs/2404.06138)
- Bandarkar et al., 2023. Belebele. [arXiv:2308.16884](https://arxiv.org/abs/2308.16884)
- Kargaran et al., 2023. GlotLID. [arXiv:2310.16248](https://arxiv.org/abs/2310.16248)
- NLLB Team, 2022. No Language Left Behind. [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)
- Freitag et al., 2021. MQM experts and error taxonomy. [arXiv:2104.14478](https://arxiv.org/abs/2104.14478)
- Kocmi et al., 2024. ESA error-span annotation. [arXiv:2406.11580](https://arxiv.org/abs/2406.11580)
- Post, 2018. sacreBLEU. [arXiv:1804.08771](https://arxiv.org/abs/1804.08771)
- Rei et al., 2022. CometKiwi. [arXiv:2209.06243](https://arxiv.org/abs/2209.06243)
- Juraska et al., 2024. MetricX-24. [arXiv:2410.03983](https://arxiv.org/abs/2410.03983)
- Rei et al., 2020. COMET. [arXiv:2009.09025](https://arxiv.org/abs/2009.09025)
