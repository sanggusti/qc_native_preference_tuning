# Standard Inspect benchmarks and their translated variants

The study reuses standard tasks from `inspect_evals` (installed version 0.19.0, Inspect AI 0.3.262) with translated datasets swapped in. This document records, per task, what the package does, whether the scorer is language-agnostic, and what a translated variant needs. It is the checklist for adding a benchmark to `configs/benchmark/`. The rule from `docs/methodology.md`: a benchmark enters a series only if its scorer does not depend on the language of the answer.

## 1. Two protocol constraints imposed by the scorers

Verified by executing the installed code:

- Letter-choice scoring (`multiple_choice` solver, `choice` scorer) extracts the answer with a case-insensitive regex on the literal `ANSWER:` marker. `Answer: b.` and `answer: B` are accepted; `JAWABAN: B` returns nothing. Translated instructions keep the marker.
- Numeric matching (`match(numeric=True)`) strips currency symbols and thousands separators written with commas, and reads a period as the decimal point: `Jawaban: 1.500` against target 1500 scores wrong (parsed as 1.5), `2,5` against 2.5 scores wrong (parsed as 25), and `Rp18` yields no number. Translated items and translated instructions keep English number formatting, and the gate checks it.

Both constraints are shared by every language, so they do not favour one language over another; they do mean that the marker itself is an English token in every prompt, which the methodology lists as a residual threat. Because an untuned model may still answer in the Indonesian numeral convention, the task scores numeric benchmarks twice: with the strict scorer and with `numeric_match_locale`, which rewrites `1.500` to 1500 against integer targets and `2,5` to 2.5 against fractional targets before matching. The locale-tolerant score is the primary metric; the strict score is kept for comparability with published numbers.

## 2. Inventory

Legend for the scorer column: agnostic (numeric, letter or execution), English-dependent (pattern on English words, English article stripping, English rule checkers), model-graded (English rubric; usable only as a secondary metric with a fixed judge).

| Task | Data source | Solver and template | Scorer | Translate | Translated variant |
|---|---|---|---|---|---|
| mgsm | per-language TSV from the simple-evals mirror, 11 languages, none in this study | `generate`; per-language CoT instruction with a native "Answer:" label | numeric, agnostic | question; instruction once per language | reuse the CSV loader with a local TSV; add instructions for id, jv, su, min, ace; carry an explicit source id for pairing |
| gsm8k | `openai/gsm8k` | `prompt_template` with an English math template; optional English few-shot from train | numeric, agnostic | question, answer (keep `####`); template | reuse `record_to_sample`; set `fewshot=0` unless a translated few-shot pool exists. Used as the math benchmark in S01 |
| math | MATH-lighteval | template; optional few-shot | model-graded equivalence, plus exact match and sympy | problem; template | prefer the exact and sympy scorers; unit words in English are stripped by the normalizer but translated unit words are not |
| mmlu (0-shot, 5-shot) | `cais/mmlu`; non-English through `openai/MMMLU` | `multiple_choice`, English template; 5-shot uses English dev examples | letter, agnostic | question, choices; template | publish in the MMMLU column layout and reuse `record_to_sample_mmmlu`; 0-shot only unless a translated dev set exists |
| mmlu_pro | TIGER-Lab/MMLU-Pro | `multiple_choice` with a CoT template | letter, agnostic | question, options; template | native `question_id` supports pairing |
| arc (easy, challenge) | allenai/ai2_arc | `multiple_choice` default template | letter, agnostic | question, choices.text | keep the nested `choices` layout and `answerKey`; native id supports pairing |
| gpqa | simple-evals CSV | `multiple_choice`, CoT; choices shuffled | letter, agnostic | question, four answers; template | CSV loader with the same headers; seed the shuffle |
| medqa | bigbio/med_qa (script loader) | `multiple_choice` with an English template | letter, agnostic | question, choices; template | bypass the script loader, reuse `record_to_sample`; the translated correct option must equal one translated choice exactly. Used as the medical benchmark in S01 |
| pubmedqa | qiaojin/PubMedQA | `multiple_choice`, yes/no/maybe | letter, agnostic | context, question, the three option words; labels | wrapper translates the "Context:" and "Question:" labels |
| healthbench | simple-evals JSONL | `generate` | model-graded per rubric item | conversation; rubrics | expensive; rubric language decision needed; secondary only |
| humaneval | openai/openai_humaneval | `generate` with an English instruction | execution, agnostic | docstring inside the prompt; instruction | best case: only natural language changes, tests and code are fixed; Docker sandbox required |
| mbpp | google-research-datasets/mbpp | English few-shot examples hard-wired | execution, agnostic | prompt text; template and few-shot | wrapper must rebuild the prompt with translated few-shot |
| ifeval | google/IFEval | `generate` | English rule checkers | prompts, constraint keywords | only language-neutral constraint types transfer; secondary |
| hellaswag | Rowan/hellaswag | English system message plus `multiple_choice` | letter, agnostic | context, endings; system message | thin wrapper |
| winogrande | allenai/winogrande | template with a `[BLANK]` token | letter, agnostic | sentence, options; label | pronoun-gap items translate poorly; low priority |
| commonsense_qa | tau/commonsense_qa | `multiple_choice` | letter, agnostic | question, five choices | thin wrapper |
| piqa | ybisk/piqa (script loader) | `multiple_choice` | letter, agnostic | goal, two solutions | bypass the script loader |
| boolq | google/boolq | inline English template | pattern on Yes/No | passage, question | keep English Yes/No as the answer protocol, or swap the pattern |
| truthfulqa | truthfulqa/truthful_qa | `multiple_choice` | letter, agnostic | question, targets | many items are culture-bound; validity uncertain |
| squad | rajpurkar/squad_v2 | English system message | token F1 with English article removal; literal `unanswerable` | context, question, spans | spans must be extracted from the translated context; article removal is a no-op for these languages |
| drop | EleutherAI/drop | English few-shot system prompt | token F1 | passage, question, answers | numeric subsets are safest |
| race_h | ehovy/race | `multiple_choice` | letter, agnostic; clustered by article | article, question, options | long passages; expensive to translate |
| bbh | Joschka/big_bench_hard | few-shot prompts stored in the dataset | letter for MC subsets; English words for binary subsets | question, choices, few-shot | MC subsets only; skip word-sorting and other English-specific subsets |
| musr | TAUR-Lab/MuSR | English system prompt plus `multiple_choice` | letter, agnostic | narrative, question, choices | long narratives |
| simpleqa | simple-evals CSV | `generate` | model-graded with an English rubric | question | entity names should not be translated; secondary |
| agieval | AGIEval JSONL, English tasks | `multiple_choice` or cloze | letter for MC; model-graded for cloze with the model under test as grader | passage, question, options | MC only; replace the cloze grader with exact match |
| paws | google-research-datasets/paws | English template | substring test on Yes/No | sentence pairs | fragile scorer; swap for a pattern or choice protocol |
| lingoly | ambean/lingOly | English problem sheet | JSON exact match | instruction sheet | not a natural fit |
| onet | matichon/thai-onet-m6-exam | English system message, `multiple_choice` | letter, agnostic | already Thai | the package's only non-English-native exam task; the pattern to copy for an Indonesian exam set |
| sciknoweval | hicai-zju/SciKnowEval | per-item system prompt | mixed; MC agnostic, generation English-tooling dependent | question, per-item prompt | MC and classification subset only |
| chembench | jablonkagroup/ChemBench | templates with `[ANSWER]` tags | letter or numeric with tolerance, agnostic | question text | tag protocol is script-neutral |
| lab_bench | futurehouse/lab-bench | CoT `multiple_choice` | letter with an English abstain string | question, choices, abstain string | text subsets only; pass the translated abstain string |
| frontierscience | openai/frontierscience | `generate` | model-graded (olympic) and rubric (research) | problem | olympic items only; always pass an explicit grader |
| aime2024 | Maxwell-Jia/AIME_2024 | template | numeric, agnostic | problem; template | 30 items; several epochs needed |
| livebench | livebench/* | English instructions inside the prompt | task-specific English parsers | | not a translation target |

## 3. Mechanisms used for sweeps and reproducibility

- `inspect eval-set --log-dir logs/{series}/{benchmark}/{language}` retries and resumes completed samples; one log directory per cell.
- `-T name=value` sets task arguments; resolved arguments are stored in the log and appear as `task_arg_*` columns in `evals_df`.
- `--model-role grader=...` binds a fixed judge for any secondary model-graded metric; scorers must fetch it with `get_model(role="grader", required=True)`, because an unbound role otherwise falls back to the model under test.
- `stderr(cluster="...")` and `ci(method="bootstrap", cluster="...")` give clustered errors over a metadata key; the task clusters by `source_id`.
- `--temperature 0`, `--max-tokens`, `--seed` (honoured by the `hf` and `vllm` providers), `--epochs`, `--limit`, `--sample-id`.
- Providers for finetuned checkpoints: `hf/<repo>` in process, `vllm/<repo>` against a local server, or `openai-api/<service>/<model>` for a hosted endpoint; the harness is identical across conditions.
- `inspect_ai.analysis.samples_df` exports per-sample scores and `metadata_*` columns for the paired analysis outside Inspect.
- Task and eval metadata merge into the log; sample metadata is available to solvers, scorers and templates.

## 4. The second benchmark and the roadmap

S01 runs gsm8k in tier 0. MedQA, the original second benchmark, is parked at tier 3 through the series `tiers` block because a thousand rows of letter answers teach the answer format rather than the task, the base sits near chance in the regional languages, and the borrowing policy keeps the medical terms Indonesian or Latin so the medium is barely varied. Phase 2 decides the second benchmark against five criteria, recorded as an amendment in `docs/experiments.md`:

1. Finetuning on 1,000 rows moves the primary base by at least the target detectable difference (`analysis.target_mde_points`) in English, measured with an English-only probe that costs no translation: finetune on the English train subset with the pinned recipe and evaluate on the English test split. A benchmark that does not move in English cannot show a medium effect either.
2. The scorer is language-agnostic (section 1).
3. The untuned base is above chance on the Indonesian and Minangkabau 250-item subsets (the floor rule of the methodology, section 4.1).
4. Items are not culture-bound; a tagged subset of US-specific stems is acceptable, a majority is not.
5. The answer is produced by generation, so the medium enters the output, or the task carries a reasoning trace before the letter.

Candidates in the order they are probed: medqa as it stands; arc (challenge, letter but with a reasoning trace under the CoT template); a drop numeric subset (generative, numeric scorer); humaneval with translated docstrings (execution scorer; needs the code solver-scorer pair in `translated_benchmark.py`); a math subset under the exact and sympy scorers. Beyond the second benchmark, the next candidates in order of how little they need translated and how clean the scorer is: mmlu (0-shot through the MMMLU layout), commonsense_qa, gpqa. Each enters through one file under `configs/benchmark/` and a solver-scorer pair in `translated_benchmark.py` if the type is new (execution scoring is the one pair not yet implemented).
