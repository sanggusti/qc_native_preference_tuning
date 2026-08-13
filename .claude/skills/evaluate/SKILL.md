---
name: evaluate
description: Write and run Inspect AI evaluations for this project, including LLM-as-judge scoring, eval sweeps across domains/languages, viewing logs, and logging results to wandb. Use when creating eval tasks, running inspect eval, comparing models, setting up judges/win-rates, or analyzing eval results.
---

# Evaluation (Inspect AI)

Tasks live in `src/evals/tasks/`; the reference is `multilingual_qa.py` (parameterized `@task`, `model_graded_qa` scorer, domain/language metadata). Eval configs live in `configs/evals/` (template `_template.yaml`). Inspect docs serve markdown: `https://inspect.aisi.org.uk/<page>.html.md` (index `/llms.txt`).

## Task pattern (keep tasks parameterized)

```python
@task
def multilingual_qa(dataset_path=None, language="id", domain="general") -> Task:
    return Task(
        dataset=json_dataset(dataset_path, sample_fields={"input": "question", "target": "answer"}),
        # or: from inspect_ai.dataset import hf_dataset -> hf_dataset(f"sanggusti/{domain}-qa-{language}", ...)
        solver=[generate()],
        scorer=model_graded_qa(),
        metadata={"domain": domain, "language": language},
    )
```

Task params are the experiment surface. Never fork a task per experiment; add parameters with defaults.

## Running

```bash
# Single run: -T sets task params
uv run inspect eval src/evals/tasks/multilingual_qa.py \
  --model openai/gpt-4o-mini -T language=jv -T domain=medicine

# Config-driven (per-experiment YAML with task params)
uv run inspect eval src/evals/tasks/multilingual_qa.py \
  --model ... --task-config configs/evals/medicine_jv.yaml

# Smoke first
... --limit 5
```

- `--run-config` bundles task + model + generation settings in one file. `--model-role grader=...` / task `model_roles` pin the judge model separately from the model under eval.
- Logs land in `logs/inspect/` (`minimal_config.yaml` convention; set `INSPECT_LOG_DIR` or `--log-dir`). Inspect with `uv run inspect view`.
- Finetuned checkpoints: eval HF models via `--model hf/sanggusti/{domain}-{language}-finetuned` (needs `[training]` extra + GPU; use the `compute` skill to run on Modal/Lightning) or serve them and use an OpenAI-compatible endpoint.

## Judging (primary metric = pairwise win rate)

- Built-ins: `model_graded_qa()` / `model_graded_fact()`; customize `template`, `instructions`, `grader_model`, multi-judge voting by passing a list of models.
- Project judge prompts: `src/prompts/judge.py` (rubric + pairwise essay judges), `src/prompts/medical_tasks.py` (medical extraction + privacy judges).
- **English-bias pitfall** (docs/research/4): LLM judges favor English/high-resource outputs. Grade with explicit rubrics, instruct the judge in the target language where feasible, and keep an English regression check (GSM8K/MMLU subset) per docs/research.md.
- Full matrix: 6 finetune conditions x 5 eval languages x 5 domains = 150 runs; drive it as a loop over `configs/evals/*.yaml`, one wandb run per eval.

## Results -> wandb

Read scores from the eval log (`inspect_ai.log.read_eval_log(path)` or `inspect_ai.analysis` dataframes) and log to wandb: project `qc_native_preference_tuning`, run `eval-{domain}-{language}`, tags include the finetune condition; log accuracy/win-rate plus the log file path and HF model/dataset URLs.
