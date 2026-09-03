# Reproducibility: registries, the planner and sweeps

Pipelines are generic; experiments are configs. This document describes the registries that define an experiment, the planner that derives the matrix, the naming rules, the commands per stage, and the steps for adding a language or a benchmark. What exists in the repository today and what remains to be implemented is stated explicitly in section 8.

## 1. Principle

Every value that could differ between two experiments lives in a YAML file under `configs/`. Pipeline code reads the registry and never hardcodes a language, a benchmark, a model, a repository id or a row count. A new experiment is a new config plus command-line overrides, reusing the existing stage modules. If a stage cannot express an experiment, it gets a new config key, and old configs keep working.

## 2. Registries

### 2.1 Languages: `configs/language/{code}.yaml`

| Field | Meaning |
|---|---|
| `code` | short code used in repository names, wandb tags and series lists (`jv`) |
| `iso_639_3` | ISO 639-3 code (`jav`) |
| `name` | English name, substituted verbatim into translation blueprints |
| `native_name` | endonym |
| `flores_code` | FLORES-200 and NLLB-200 code (`jav_Latn`); null if the language is not covered (Madurese), in which case NusaX is the calibration reference |
| `script` | script used for every dataset in the study; Latin throughout |
| `family` | genealogical placement, for interpretation |
| `resource_tier` | high, medium, low, very_low |
| `translation_register` | the register pinned for all translations (`ngoko`, `loma`, `formal`, `standard`) |
| `register_notes` | why that register, and which alternatives exist |
| `register_instruction` | the sentence substituted into every translation blueprint through `{register_instruction}` |
| `nllb_supported` | NLLB-200 can act as the second translator |
| `xlmr_covered` | the language is in XLM-R's pretraining set, so encoder-based quality estimation is inside coverage |

Template: `configs/language/_template.yaml`. Entries today: en, id, jv, su, min, ace.

### 2.2 Benchmarks: `configs/benchmark/{name}.yaml`

| Field | Meaning |
|---|---|
| `name`, `domain` | registry key (also the Hub repository stem) and domain label |
| `inspect_reference` | the `inspect_evals` task this mirrors |
| `scorer`, `scorer_language_agnostic` | `numeric_match`, `choice`, `code_exec`, `exact_match`; must be language-agnostic |
| `eval_source`, `train_source` | Hub repository, config, split and field mapping of the English sources; `min_rows` for the training split; optional `subsets` to tag |
| `translate.fields` | columns rewritten by the translator |
| `translate.preserve` | what the blueprint must keep verbatim |
| `translate.blueprint` | the translation instruction with `{language}` and `{register_instruction}` substituted (a registry test fails if either placeholder is missing) |
| `translate.prompt_template_key` | the instruction template that is translated once per language |
| `translate.variants` | derived eval-only splits (`rt`, `nllb`, `pro1`) |

Template: `configs/benchmark/_template.yaml`. Entries today: gsm8k, medqa.

### 2.3 Series: `configs/series/{series}.yaml`

A series is one controlled experiment. It names the base models (`primary`, `contrast`), the languages, the benchmarks, the conditions with their tiers and restrictions, the active conditions, the replicates, the pinned AutoScientist and translation settings, the decoding settings, the covariates, and the naming templates. Changing any constant means a new series file; the old one stays as the record of what was run.

Condition keys: `train` (`null`, `same`, `other`, `all`, or a language code), `tier`, `description`, and optionally `eval_languages`, `exclude_eval_languages`, `benchmarks`, `bases`, `eval_variant`.

## 3. The planner

```bash
uv run python -m pipeline.plan                          # matrix table for the default series
uv run python -m pipeline.plan tier=0                   # the minimum publishable unit only
uv run python -m pipeline.plan format=commands stage=datagen
uv run python -m pipeline.plan format=commands stage=finetune tier=0
uv run python -m pipeline.plan format=commands stage=eval
uv run python -m pipeline.plan series=s02_adaptive      # another series file
```

`pipeline/plan.py` loads the series through `src/utils/registry.py`, validates every referenced language, benchmark, base role and condition, expands the cells, de-duplicates cells that coincide (the English anchor evaluated in English is the native English cell), and prints counts per tier, per condition, and the command for every dataset, finetune and evaluation. Nothing in the planner submits work. The printed plan is pasted into `docs/experiments.md` before the first paid run of a series.

For series S01 the planner reports 30 datasets (10 translated train and test repositories, 2 English source repositories, 10 round-trip splits, 6 digit re-instantiated splits, 2 pooled training sets), 60 finetunes and 255 evaluation runs, of which tier 0 is 36 finetunes and 162 evaluations.

## 4. Naming

| Artifact | Template | Example |
|---|---|---|
| Dataset | `sanggusti/{benchmark}-{language}` with splits `train` and `test` | `sanggusti/gsm8k-jv` |
| Derived eval split | `sanggusti/{benchmark}-{language}-{variant}` | `sanggusti/gsm8k-jv-rt` |
| Pooled training set | `sanggusti/{benchmark}-all` | `sanggusti/medqa-all` |
| Model | `sanggusti/{benchmark}-{train_language}-{series}-{base}-r{replicate}` | `sanggusti/gsm8k-jv-s01_language_medium-gemma3-4b-r1` |
| wandb project | `qc_native_preference_tuning` | |
| wandb run | `{stage}-{benchmark}-{language}-{condition}-{base}-r{replicate}` | `eval-gsm8k-jv-native-gemma3-4b-r1` |
| wandb tags | benchmark, language, condition, base role, series | |

Base slugs are declared in the series file (`naming.base_slugs`). Names are produced only by `expand_matrix`; no stage composes a name by hand.

## 5. Stage commands

The planner prints one command per unit of work. The eval command is runnable today for any split whose Hub repository exists; the datagen and finetune modules are on the roadmap (section 8) and the printed commands are their specification.

```bash
# datagen (planned module): one Adaption translation run per benchmark x language
uv run python -m pipeline.datagenerator.translate_benchmark benchmark=gsm8k language=jv \
    series=s01_language_medium push_to_hub=sanggusti/gsm8k-jv

# derived eval splits (planned module): round trip, NLLB-200, digit re-instantiation
uv run python -m pipeline.datagenerator.derive_split benchmark=gsm8k language=jv variant=rt \
    series=s01_language_medium push_to_hub=sanggusti/gsm8k-jv-rt

# finetune (planned module): one AutoScientist run per benchmark x base x train language x replicate
uv run python -m pipeline.training.autoscientist_finetune series=s01_language_medium \
    benchmark=gsm8k base=primary language=jv replicate=1 \
    hub_model_id=sanggusti/gsm8k-jv-s01_language_medium-gemma3-4b-r1

# eval (runnable): one Inspect run per cell; base cells use the same hf/ provider as finetuned cells
uv run inspect eval src/evals/tasks/translated_benchmark.py --model hf/sanggusti/gsm8k-jv-s01_language_medium-gemma3-4b-r1 \
    -T benchmark=gsm8k -T language=jv -T dataset_repo=sanggusti/gsm8k-jv \
    --temperature 0.0 --max-tokens 2048 \
    --metadata condition=native --metadata train_language=jv --metadata base=primary \
    --metadata replicate=1 --metadata series=s01_language_medium

# derived split (round trip): the planner adds -T variant and the suffixed repo
uv run inspect eval src/evals/tasks/translated_benchmark.py --model hf/sanggusti/gsm8k-en-s01_language_medium-gemma3-4b-r1 \
    -T benchmark=gsm8k -T language=jv -T variant=rt -T dataset_repo=sanggusti/gsm8k-jv-rt ...

# smoke first, always
uv run inspect eval src/evals/tasks/translated_benchmark.py --model ... -T benchmark=gsm8k -T language=jv --limit 20
```

Sweeps: a stage module that takes Hydra overrides can be swept with multirun, for example `uv run python -m pipeline.datagenerator.translate_benchmark -m language=id,jv,su,min,ace benchmark=gsm8k,medqa`. Evaluation sweeps run through `inspect eval-set --log-dir logs/{series}/{benchmark}/{language}` so that retries resume from completed samples and every log directory maps to one cell.

## 6. The evaluation task

`src/evals/tasks/translated_benchmark.py` is one parameterized task for every benchmark and language. It reads the benchmark registry to choose the solver and scorers (for numeric benchmarks `generate` with Inspect's strict numeric match and a locale-tolerant numeric match that rewrites Indonesian-style numerals first; for multiple choice `multiple_choice` with the choice scorer), loads the Hub repository the planner resolved from the series naming (`-T dataset_repo`, with `-T variant` appended for derived splits; a template is the fallback, and a local JSONL serves smoke tests and unit tests), takes the translated instruction template from the dataset, clusters the standard error by `source_id`, and records benchmark, language, variant and dataset in the log metadata. Confidence intervals for the analysis are computed from the exported per-sample scores in `src/analysis/`, not inside Inspect. Model roles for any secondary judge are bound with `--model-role grader=...` and fetched with `required=True`. Tests in `tests/test_translated_benchmark.py` run without network on fixtures under `tests/fixtures/`.

Canonical dataset schema consumed by the task: `id` (equal to `source_id`), `question`, `choices` (multiple choice only), `target`, `instruction`, `instruction_en`, `question_en`, `answer_en`, `language`, `register`, `blueprint_hash`, `translator_version`, and the per-item quality columns listed in `docs/methodology.md` section 5.4.

## 7. Adding a language or a benchmark

Adding a language:

1. Copy `configs/language/_template.yaml` to `configs/language/{code}.yaml` and fill every field; decide the register and record why.
2. Add the code to `languages` in the series file (or start a new series if the constants change).
3. Run `uv run pytest tests/test_registry.py` and `uv run python -m pipeline.plan`; the new cells, names and commands appear.
4. Run Phase 0 and Phase 1 of `docs/methodology.md` for the language (covariates, FLORES or NusaX calibration, probe translation, gate). Record the results in `docs/experiments.md`.
5. Only then submit translation, finetuning and evaluation for the language, in tier order.

Adding a benchmark:

1. Confirm the scorer is language-agnostic; `docs/benchmarks.md` lists which standard Inspect tasks qualify and what each needs translated.
2. Copy `configs/benchmark/_template.yaml` to `configs/benchmark/{name}.yaml`; set sources, field mapping, the blueprint, the preserved fields and the instruction template key.
3. If the scorer type is new, extend `translated_benchmark.py` with the solver and scorer pair and add a fixture test.
4. Add the name to `benchmarks` in the series file; run the tests and the planner.
5. Translate the instruction template once per language, have it reviewed, and store it on every row.

## 8. What exists and what remains

Implemented in this revision:

- Registries for languages, benchmarks and series; the loader and matrix expansion in `src/utils/registry.py`; the planner `pipeline/plan.py`; tests.
- The evaluation task `src/evals/tasks/translated_benchmark.py` with fixture tests.
- The legacy MGSM converter fixed for Adaption SDK 0.10.0 (`mgsm_convert.py`) with stub-client tests.
- Diagrams under `docs/diagrams/` (draw.io sources, SVG and PNG renders, and the generator and renderer scripts).

Remaining stage modules, in the order the phases need them (tracked in `docs/experiments.md`):

| Module | Purpose |
|---|---|
| `pipeline/covariates/fertility.py`, `bits_per_byte.py` | per-language, per-item token counts and base-model bits per byte on FLORES-200 devtest (runs on Lightning or Modal) |
| `pipeline/datagenerator/translate_benchmark.py` | registry-driven translation: fixed item subset, blueprint route with pinned settings, canonical schema, quality columns, gate, Hub push, raw-mode upload |
| `pipeline/datagenerator/gate.py` | structural checks, GlotLID, leakage, register, chrF++ against references, thresholds, report into the dataset card |
| `pipeline/modal_runner/nllb_translate.py` | NLLB-200 3.3B on Modal: FLORES calibration translations, second-opinion translations, back-translations for the quality columns and the round-trip split (Phase 1) |
| `pipeline/datagenerator/derive_split.py` | round-trip, NLLB-200 and digit re-instantiation splits, calling the NLLB job |
| `pipeline/datagenerator/pool_benchmark.py` | the pooled training set at equal total rows |
| `pipeline/training/autoscientist_finetune.py` | pinned create call, `best_hyperparams` check, artifact download, Hub push, wandb logging |
| `pipeline/evals/serve_modal.py` | one vLLM server per checkpoint, Inspect through the OpenAI-compatible provider, smoke wrapper |
| `src/analysis/` | log reader, paired bootstrap, mixed models, contrast tables with corrections, figures, the identification statement |

## 9. Environment and tracking

Secrets come from `.env` (`ADAPTION_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, `LIGHTNING_API_KEY`, judge keys). Modal jobs read the cloud secrets named `huggingface` and `wandb`. Every stage logs its resolved configuration as the wandb run config, the Hub URLs it produced, the Adaption ids it used, and the git commit of the configs. The pre-flight checklist in `AGENTS.md` applies before every paid run.
