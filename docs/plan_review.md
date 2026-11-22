# Plan review: s01_language_medium

Review of the original research plan against literature, SDK reality, and platform status. Companion to docs/methodology.md (design) and docs/benchmarks.md (protocol).

## 1 Original plan summary

The original plan asked whether finetuning language affects accuracy across low resource Indonesian languages, crossing 5 domains (medical, math, programming, science, general) with 5 languages (`en id jv su min`) for 25 datasets and 150 eval runs. Data flow was Adaption translation, AutoScientist finetuning, Inspect evals, HF publishing, wandb tracking, on Modal and Lightning compute. Controls were same base, parallel content, equal rows, fixed `max_iterations` and `target_win_rate`. Metrics were accuracy plus model-graded quality with paired bootstrap and Cohen d. The literature survey covered Indonesian eval gaps, multilingual mechanics, and dataset catalogs.

## 2 Findings

1. Scope exceeded the quality budget. Five domains times five languages with per-domain blueprints made translation audit infeasible; the plan had no per-item quality columns, no admission thresholds, and no translator-loss bound, so language effects and translation errors were confounded.
2. Estimands were implicit. The plan named cross-lingual transfer, domain sensitivity, and saturation but defined no paired contrasts (`Delta_en`, `Delta_id`), no pooled control at equal rows, and no replication rule, leaving H1-H4 untestable as stated.
3. Anchoring was underspecified. English-anchor and Indonesian-anchor arms existed as cells but had no register pinning, no instruction template control, and no prompt-level anchor baselines (XLT [arXiv:2305.07004](https://arxiv.org/abs/2305.07004), self-translate [arXiv:2308.01223](https://arxiv.org/abs/2308.01223)), despite QAlign [arXiv:2401.07817](https://arxiv.org/abs/2401.07817), PLUG [arXiv:2311.08711](https://arxiv.org/abs/2311.08711), and MAPO [arXiv:2401.06838](https://arxiv.org/abs/2401.06838) showing anchors are the comparison to beat.
4. Statistics were underpowered and uncorrected. Paired bootstrap plus Cohen d with no family definition, no item and replicate two-level structure, and no multiplicity control would inflate false positives across the run matrix.
5. Contamination and tokenizer confounds were unmeasured. Small perturbations of benchmark items move accuracy substantially (GSM-Symbolic, Mirzadeh et al., 2024, [arXiv:2410.05229](https://arxiv.org/abs/2410.05229)), and fertility gaps change effective difficulty (Petrov et al., 2023, [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)), yet the plan collected neither overlap screens nor fertility/BPB covariates.
6. Platform assumptions were stale. The plan assumed Lightning for eval compute and current Adaption SDK semantics; installed SDK is 0.8.0 with narrower return types, and Lightning is down (Section 4).

## 3 Robustness changes made

1. Narrowed series to `s01_language_medium`: 2 benchmarks (`gsm8k`, `medqa`), 9 languages (`en id jv su min ace ban bjn bug`), fixed `train_rows=1000`, 250-item tests, 3 replicates, primary plus contrast base. Depth over breadth: full quality gates and bounds on two benchmarks beat thin coverage on five.
2. Pre-registered E1-E9 with retired E10, paired `Delta_en`/`Delta_id` contrasts, and H1-H4 with directional verdicts. Added E5 pooled at equal total rows, E6 translator-loss bound from `-rt`/`-nllb`, E7 sign plus rank replication on the contrast base, E8 covariate adjustment, E9 contamination difference-in-differences on `-pro1`.
3. Pinned registers (`jv` ngoko, `su` loma, others standard), `{language}` plus `{register_instruction}` blueprint substitution, `ANSWER:` marker with Latin-only numerals, language-agnostic scorers, and recorded `instruction_variant`.
4. Added quality gate Steps 0-5 with numeric thresholds, per-item quality columns, `-rt`/`-nllb`/`-pro1`/`-all` derived splits, dataset cards with canary and do-not-train, and a contamination difference-in-differences check.
5. Replaced loose stats with item bootstrap plus Wilson per cell (Wilson (1927, Journal of the American Statistical Association); Efron (1979, The Annals of Statistics)), two-level bootstrap plus McNemar plus mixed model for contrasts (McNemar (1947, Psychometrika)), Holm over the primary family and BH elsewhere (Holm (1979, Scandinavian Journal of Statistics); Benjamini and Hochberg (1995, Journal of the Royal Statistical Society Series B)), and interval guidance per Brown, Cai and DasGupta (2001, Statistical Science). Mixed models follow lme4 [arXiv:1406.5823](https://arxiv.org/abs/1406.5823).
6. Added base floor rule (gsm8k-id at least 30 percent, English below 80 percent, prefer smallest), 20-item smoke gate, Phase 0-5 go/no-go rules, fallback ladders with no skipped rungs, exclusion vocabulary, and cost accounting.
7. Added fertility, BPB, NusaX/X similarity via sacreBLEU [arXiv:1804.08771](https://arxiv.org/abs/1804.08771), NusaX context [arXiv:2205.15960](https://arxiv.org/abs/2205.15960), reference audit, and human review covariates. Deferred the third benchmark until translation validity is proven.
8. Made experiments config-driven: language, benchmark, and series registries plus `pipeline.plan`, so the full sweep is reviewable before any paid run.

## 4 SDK and platform notes

Installed Adaption SDK is 0.8.0. Verified against `adaption/resources/datasets/datasets.py` and `adaption/resources/autoscientist.py`:

* `datasets.run` has NO `idempotency_key`. Only `autoscientist.create` accepts `idempotency_key`, scoped to `dataset_id` while the run is in progress. Do not pass it to dataset runs.
* `datasets.download` returns `str` (presigned URL). Only `autoscientist.download` returns `BinaryAPIResponse` with `write_to_file`. Dataset download code must fetch the URL; model download code uses `write_to_file`.
* `create_from_huggingface` is deprecated. Prefer explicit `upload_file` plus `run` for provenance.
* AutoScientist: pin via `recommend_hyperparams` first, then pass explicit hyperparams to every `create`. Minimum 1000 rows. `batch_size` stays at the per-model default. `succeeded` means hit `target_win_rate` or exhausted `max_iterations`; always read `best_win_rate`.
* Paid compute is Modal plus Adaption only. Lightning is down and is not available for eval compute or hosted inference. All GPU and batch work routes to Modal; all adaptation and managed finetuning routes to Adaption. No Lightning fallback in this series.

## 5 Residual risks

1. Translation quality in the lowest tier (`min`, `ace`, `ban`, `bjn`, `bug`) may fail admission twice, shrinking the low exposure tail and weakening H1 and H3. Mitigation: `-nllb` cross-check and early Phase 1 exit for weak languages.
2. Even pinned AutoScientist runs may vary across replicates (`s_r > 3.5` halt in 10.5). Mitigation: replicate rule with diagnose-before-scale.
3. Contamination difference-in-differences has low power on 250 items if re-instantiation barely moves accuracy. Mitigation: `-pro1` re-estimation is pre-registered regardless of significance.
4. Single contrast base for E7 if only one base clears the floor rule. Mitigation: E7 falls back to cross-replicate sign stability, labeled as weakened replication.
5. Cost overrun if Phase 4 smoke underestimates full-run price. Mitigation: 1.3 contingency multiplier and per-run credit logging with a hard Phase 4 cap.
