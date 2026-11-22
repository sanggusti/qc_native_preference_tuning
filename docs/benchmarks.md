# Benchmarks: s01_language_medium

Two active benchmarks: `gsm8k` (numeric match) and `medqa` (choice). This file defines the protocol, registry, procedure, and roadmap. Methodology estimands and analysis live in docs/methodology.md.

## 1 Benchmark protocol

Response format: every model output must end with a line `ANSWER: <value>`. For `gsm8k`, `<value>` is the final number in English numerals (Latin digits 0-9, optional decimal point and leading minus). For `medqa`, `<value>` is one choice letter A, B, C, or D. Instruction templates state this format in the eval language with the pinned register, and include one minimal example that uses the `ANSWER:` marker but no task content overlapping test items.

English numerals are mandatory in all languages: scorers accept only Latin digits for `gsm8k`. Localized digit strings score 0 and increment the parse-failure count, so numeral localization shows up as a measured error, not silent mis-scoring.

Language-agnostic scorers: `gsm8k` extracts the last `ANSWER:` line and compares numeric values with tolerance 1e-6 after stripping commas and whitespace. `medqa` extracts the last `ANSWER:` letter and compares case insensitively to the reference letter. No semantic judge, no language specific parsing. Optional LLM-judge scores are secondary only and never overturn scorer verdicts.

References: GSM8K [arXiv:2110.14168](https://arxiv.org/abs/2110.14168), MedQA [arXiv:2009.13081](https://arxiv.org/abs/2009.13081), MGSM translation precedent [arXiv:2210.03057](https://arxiv.org/abs/2210.03057), MMedC/MMedBench context [arXiv:2402.13963](https://arxiv.org/abs/2402.13963), XlingEval healthcare eval design [arXiv:2310.13132](https://arxiv.org/abs/2310.13132).

## 2 Registry schema and blueprint rules

Benchmarks are registry entries under `configs/benchmark/{name}.yaml` following `_template.yaml`, loaded by `src/utils/registry.py`. Schema keys: `name`, `source_dataset`, `source_config`, `source_split_train`, `source_split_test`, `record_fields`, `preserved_fields`, `scorer` (`numeric` | `choice` | `code_exec`), `scorer_language_agnostic`, `instruction_template_key`, `blueprint`. Languages live under `configs/language/{code}.yaml` with `register` and `register_instruction`. A series under `configs/series/{series}.yaml` lists its languages, benchmarks, and conditions; `expand_matrix` derives every train and eval cell and every artifact name from those three files. Adding a language is one registry file plus one code in the series list; adding a benchmark is one registry file plus one name in the series list.

Blueprint substitution: translation blueprints contain exactly two variables, `{language}` and `{register_instruction}`. At run time both are substituted from the registry; no other free text varies by language. Example skeleton: `Translate the following {benchmark} item into {language}. Register: {register_instruction}. Preserve ...`.

Preserved fields, enforced by gate Step 2: `gsm8k`: final numeric answer, all intermediate numerals, units strings, person and place names transliterated consistently. `medqa`: question stem meaning, all four options, option order, correct choice index, drug and anatomy terms (translate prose, keep standard Latin terms on first mention with translation in parentheses). Anything not in preserved fields may be localized for fluency within the pinned register.

## 3 Evaluation procedure

Splits: per benchmark per language, `train` with 1000 rows sampled by the series `sampling_seed`, `test` with the full translated test split; parallel IDs across languages. Derived repos `test_rt`, `test_nllb`, `audit_pro1` as defined in methodology 5.5. Train and test are disjoint by `id`; the join check runs before every finetune and every eval.

250-item subsets: fixed IDs sampled once by seed. All primary evals for a benchmark and language use the same 250 IDs. The 20-item smoke subset is a fixed prefix of the 250, never a fresh sample.

Decoding: greedy, `temperature=0`, `max_tokens=512` from the series decoding block, stop at end of sequence or after the `ANSWER:` line plus 16 tokens. One attempt per item. Parse failures score 0.

Log dirs: one directory per cell from the series `log_template`, `logs/{series}/{benchmark}/{eval_language}/{condition}/{base}-r{replicate}/`, containing the Inspect log, `config_snapshot.yaml`, and `parse_report.json`. wandb mirrors summary metrics; logs are the source of truth.

## 4 Roadmap

Active in S01: `gsm8k`, `medqa`. Candidates for later series, in priority order: `arc` (ARC [arXiv:1803.05457](https://arxiv.org/abs/1803.05457)), `mmlu` (MMLU [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)), `humaneval` (HumanEval [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)), `commonsense_qa` (CommonsenseQA [arXiv:1811.00937](https://arxiv.org/abs/1811.00937)), `gpqa` (GPQA [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)). Related multilingual references for later design: Belebele [arXiv:2308.16884](https://arxiv.org/abs/2308.16884), Global-MMLU [arXiv:2412.03304](https://arxiv.org/abs/2412.03304), MMLU-ProX [arXiv:2503.10497](https://arxiv.org/abs/2503.10497), INCLUDE [arXiv:2411.19799](https://arxiv.org/abs/2411.19799), IrokoBench [arXiv:2406.03368](https://arxiv.org/abs/2406.03368), M3Exam [arXiv:2306.05179](https://arxiv.org/abs/2306.05179), HumanEval-XL [arXiv:2402.16694](https://arxiv.org/abs/2402.16694).

A new benchmark joins a series through one registry file plus the task wiring for its scorer type; no planner changes are needed. No third benchmark work starts in S01.
