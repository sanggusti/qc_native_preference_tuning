# Research Plan: Language as Medium (Series S01)

## 1 The question

Holding base model, task content, and pipeline fixed, does the language used as the finetuning and eval medium change accuracy?

Concretely: take one benchmark, translate identical items into nine languages, finetune separate copies of one base model in different training media, and evaluate each copy in each language. If accuracy differs across media at fixed content, language itself is a causal factor in finetuning outcomes, not just a property of the test set.

This replaces the earlier domain-sweep framing (5 domains x 5 languages) with a controlled two-benchmark design that isolates the medium effect. Series id: `s01_language_medium`.

## 2 Design summary

| Factor | S01 value |
|---|---|
| Benchmarks | gsm8k (math reasoning), medqa (medical MCQ) |
| Languages | en, id, jv, su, min, ace, ban, bjn, bug |
| Train medium conditions | base, native, english_anchor, indonesian_anchor, pooled |
| Train rows per condition | 1000 |
| Test items per language per benchmark | 250 |
| Replicates (independent runs) | 3 |
| Base models | primary + contrast, fixed for all conditions (pinned in Phase 2) |
| Training hyperparameters | pinned from `recommend_hyperparams`, identical across conditions |
| Translator | Adaption datasets API, one blueprint per benchmark |
| Eval harness | Inspect AI, deterministic scorers |
| Tracking | wandb project `qc_native_preference_tuning` |

Benchmark sources: GSM8K (Cobbe et al., 2021, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)) and MedQA (Jin et al., 2020, [arXiv:2009.13081](https://arxiv.org/abs/2009.13081)). The 250-item test size follows the MGSM precedent of 250 parallel items per language (Shi et al., 2022, [arXiv:2210.03057](https://arxiv.org/abs/2210.03057)). Cross-lingual medical degradation is documented by XlingEval (Jin et al., 2023, [arXiv:2310.13132](https://arxiv.org/abs/2310.13132)) and MMedBench (Qiu et al., 2024, [arXiv:2402.13963](https://arxiv.org/abs/2402.13963)).

## 3 Languages and benchmarks

| Code | Language | ISO 639-3 | Script in S01 | Resource level |
|---|---|---|---|---|
| en | English | eng | Latin | high (anchor) |
| id | Indonesian | ind | Latin | medium (regional pivot) |
| jv | Javanese | jav | Latin | low |
| su | Sundanese | sun | Latin | low |
| min | Minangkabau | min | Latin | low |
| ace | Acehnese | ace | Latin | very low |
| ban | Balinese | ban | Latin | very low |
| bjn | Banjarese | bjn | Latin | very low |
| bug | Buginese | bug | Latin | very low |

All nine are written in Latin script in S01. Javanese, Sundanese, Minangkabau, Acehnese, Balinese, Banjarese, and Buginese are Austronesian languages with large speaker populations and thin LLM training representation.

| Benchmark | Task | Source split | Scorer | Why selected |
|---|---|---|---|---|
| gsm8k | grade-school math, numeric answer | GSM8K train/test | numeric exact match | reasoning-heavy; multilingual precedent in MGSM |
| medqa | USMLE-style 4-choice MCQ | MedQA train/test | letter exact match | knowledge-heavy; clinical relevance for the region |

Translated items keep numbers, units, drug names, and answer keys invariant; only the natural-language carrier changes. Translation quality is spot-checked with MQM-style error-span annotation (see research note 5, section 5.6).

## 4 Conditions and design matrix

For each eval language L, five condition labels are evaluated in L:

| Condition | Train medium | Answers |
|---|---|---|
| base | none (base model, no finetuning) | what the base model already does in L |
| native | L itself (1000 rows in L) | E1: the native finetuning effect |
| english_anchor | English (1000 rows in en) | E3: English-anchor transfer into L |
| indonesian_anchor | Indonesian (1000 rows in id) | E4: regional-pivot transfer into L |
| pooled | stratified mix of all six languages, 1000 rows total | E5: pooled versus native at fixed data budget |

Native runs diagonally (train L, eval L); anchors, pooled, and base are evaluated in every language. Off-diagonal native transfer cells are out of scope in S01.

Degenerate cells: for eval language en, `native` and `english_anchor` name the same training medium and the same checkpoint; for eval language id, `native` and `indonesian_anchor` coincide. Train once, evaluate once, record the accuracy under both labels. This dedup rule is part of the registered design, not a post hoc fix.

Counts per benchmark per base role per replicate: 10 unique trained checkpoints (en, id, jv, su, min, ace, ban, bjn, bug, pooled) plus base; 45 eval runs (9 languages x 5 condition labels). Series totals across both benchmarks, both base roles, and 3 replicates: 20 HF datasets, 120 unique trained checkpoints, 468 eval runs, 117,000 primary item inferences (468 x 250).

## 5 Hypotheses

- **H1 (exposure ordering):** native finetuning gains are monotone nonincreasing in pretraining exposure (largest in the min/ace/ban/bjn/bug tier, smallest in en). Prior evidence: MGSM resource ordering, Kew et al. ceiling observation (Kew et al., 2023, [arXiv:2312.12683](https://arxiv.org/abs/2312.12683)).
- **H2 (English anchor):** for gsm8k in low-resource L, english_anchor meets or beats native, because reasoning routes through the model's English-biased concept space. Prior evidence: QAlign over translate-train (Zhu et al., 2024, [arXiv:2401.07817](https://arxiv.org/abs/2401.07817)), PLUG (Zhang et al., 2023, [arXiv:2311.08711](https://arxiv.org/abs/2311.08711)).
- **H3 (task moderation):** the anchor advantage (H2) is larger for gsm8k than for medqa, where knowledge must exist in L. Prior evidence: XlingEval degradation on medical correctness, recoverable only with domain plus language adaptation.
- **H4 (pooled beats native):** pooled meets or beats native in low-resource L at the same 1000-row budget, via cross-lingual transfer within the mixture. Prior evidence: multilingual mixtures matching monolingual tuning with far fewer per-language examples (Shaham et al., 2024, [arXiv:2401.01854](https://arxiv.org/abs/2401.01854); Muennighoff et al., 2022, [arXiv:2211.01786](https://arxiv.org/abs/2211.01786)).

## 6 Estimands

| ID | Name | Definition | Decides |
|---|---|---|---|
| E1 | native effect | acc(native, L) minus acc(base, L) | does native finetuning help in L |
| E2 | exposure ordering | rank correlation of per-language E1 with exposure order | H1 |
| E3 | Delta_en | acc(english_anchor, L) minus acc(native, L) | H2 per language |
| E4 | Delta_id | acc(indonesian_anchor, L) minus acc(native, L) | regional pivot value |
| E5 | pooled-vs-native | acc(pooled, L) minus acc(native, L) | H4 |
| E6 | translator-loss bound | round-trip bound from -rt/-nllb splits | how much of any gap is translation artifact |
| E7 | contrast-base replication | E3/E4 sign and rank replicated on the contrast base | robustness across bases |
| E8 | covariate-adjusted | E1/E3/E4 re-estimated adjusting for fertility, BPB, similarity, audit | confound check |
| E9 | contamination DiD | (orig minus pro1) gap differenced across base and finetuned | memorization check |
| E10 | retired | structural smarter-medium question, one language per profile | kept in registry to prevent silent reintroduction |

## 7 Pipeline stages

1. **Datagen:** translate 1000 train + full test rows per benchmark into id, jv, su, min, ace, ban, bjn, bug via Adaption; en stays source. Push 20 datasets to `sanggusti/{benchmark}-{language}` plus pooled. Log fertility, language-ID pass rate, and gate metrics per dataset.
2. **Train:** one AutoScientist run per (benchmark, train medium, base role, replicate) with pinned hyperparameters (120 runs). Push checkpoints to `sanggusti/{benchmark}-{language}-{condition}-{base}-r{replicate}`.
3. **Eval:** Inspect AI tasks per (benchmark, eval language, condition, base role, replicate) (468 runs); deterministic scorers; 250 items each. GPU inference on Modal where needed.
4. **Analysis:** estimate E1-E9 with item bootstrap CIs, Wilson intervals, McNemar paired contrasts, mixed-effects models, and Holm/BH multiplicity control (methods in note 5, section 5.7). Results go to `docs/experiments.md` section 5.

Paid compute is Modal plus Adaption only.

## 8 Repository structure (target)

```
qc_native_preference_tuning/
  configs/
    language/
      _template.yaml en.yaml id.yaml jv.yaml su.yaml min.yaml ace.yaml
      ban.yaml bjn.yaml bug.yaml
    benchmark/
      _template.yaml gsm8k.yaml medqa.yaml
    series/
      s01_language_medium.yaml
    plan.yaml
  pipeline/
    plan.py                        # series planner: matrix and stage commands
    datagenerator/
      translate_benchmark.py       # Adaption translate -> gate -> HF push (#50)
    training/
      autoscientist_finetune.py    # AutoScientist wrapper, pinned constants (#54)
  src/
    utils/
      registry.py                  # languages, benchmarks, series, cell expansion
    evals/
      tasks/
        translated_benchmark.py    # parameterized Inspect task (benchmark, language)
  docs/
    research.md                    # this document
    methodology.md                 # method detail
    benchmarks.md                  # benchmark protocol and registry rules
    reproducibility.md             # logging and provenance contract
    plan_review.md                 # review record of the original plan
    experiments.md                 # S01 ledger
    research/                      # literature notes 1-5
```

## 9 Related work

The literature program is in `docs/research/`: notes 1-4 cover degradation evidence, multilingual internals, method families, and datasets. Note 5 (`5_language_as_medium.md`) is the dedicated survey for this series: prior work asking the same medium question, language of thought, language difficulty, Indonesian regional resources, translation quality, and small-contrast statistics.

## References

- Cobbe et al., 2021. Training Verifiers to Solve Math Word Problems. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- Jin et al., 2020. What Disease Does This Patient Have? (MedQA). [arXiv:2009.13081](https://arxiv.org/abs/2009.13081)
- Shi et al., 2022. Language Models are Multilingual Chain-of-Thought Reasoners (MGSM). [arXiv:2210.03057](https://arxiv.org/abs/2210.03057)
- Jin et al., 2023. Better to Ask in English (XlingEval). [arXiv:2310.13132](https://arxiv.org/abs/2310.13132)
- Qiu et al., 2024. Towards Building Multilingual Language Model for Medicine. [arXiv:2402.13963](https://arxiv.org/abs/2402.13963)
- Zhu et al., 2024. Question Translation Training (QAlign). [arXiv:2401.07817](https://arxiv.org/abs/2401.07817)
- Zhang et al., 2023. PLUG: Pivot Language Guided Generation. [arXiv:2311.08711](https://arxiv.org/abs/2311.08711)
- Shaham et al., 2024. Multilingual Instruction Tuning With Just a Pinch of Multilinguality. [arXiv:2401.01854](https://arxiv.org/abs/2401.01854)
- Muennighoff et al., 2022. Crosslingual Generalization through Multitask Finetuning (xP3). [arXiv:2211.01786](https://arxiv.org/abs/2211.01786)
- Kew et al., 2023. Turning English-centric LLMs Into Polyglots. [arXiv:2312.12683](https://arxiv.org/abs/2312.12683)
