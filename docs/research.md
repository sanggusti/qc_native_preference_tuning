# Research Plan: Cross-Lingual Fine-Tuning Evaluation for Low-Resource Indonesian Languages

## 1. Research Question

**Primary question:** Does the language of a fine-tuning dataset affect model performance when evaluated across low-resource Indonesian languages?

Specifically: if a model is fine-tuned on domain-specific tasks (medical, programming, science, general knowledge, mathematics) in one language (e.g., Javanese), does it perform comparably when evaluated on those same tasks in other low-resource languages (Sundanese, Minangkabau)?

### Sub-questions

1. **Cross-lingual transfer.** Does fine-tuning in one low-resource Indonesian language (e.g., Javanese) transfer better to other low-resource Indonesian languages than fine-tuning in English or Indonesian?
2. **Domain sensitivity.** Does the cross-lingual performance gap vary across task domains? For example, do medical QA and math reasoning show different transfer patterns?
3. **Data saturation.** At what training set size do same-language fine-tuning gains plateau?

## 2. Target Languages

| Language | ISO 639-3 | Speakers (approx.) | Script | Resource Level |
|----------|-----------|---------------------|--------|----------------|
| English | eng | 1.5B | Latin | High |
| Indonesian | ind | 200M | Latin | Medium |
| Javanese | jav | 82M | Latin/Javanese | Low |
| Sundanese | sun | 42M | Latin/Sundanese | Low |
| Minangkabau | min | 5.5M | Latin | Low |

Javanese, Sundanese, and Minangkabau are all Austronesian languages spoken across the Indonesian archipelago. They share some lexical overlap with Indonesian but have distinct grammars and vocabularies. All three are underrepresented in LLM training data despite having millions of native speakers.

## 3. Evaluation Domains and Source Datasets

| Domain | Source Dataset | Format | Selection Rationale |
|--------|---------------|--------|---------------------|
| Medical | MedQA-USMLE | MCQ (4-choice) | Structured accuracy eval, widely benchmarked |
| Mathematics | MGSM | Open-ended numerical | Existing translation pipeline in this repo (`mgsm_convert.py`) |
| Programming | HumanEval / MBPP | Code generation | Tests whether prompt language affects code output quality |
| Science | ARC / SciQ | MCQ | General science reasoning, grade-school to college level |
| General | MMLU-subset / TyDiQA | MCQ / extractive QA | Broad knowledge coverage |

Each dataset is translated from English into Indonesian, Javanese, Sundanese, and Minangkabau using the Adaption datasets API (see Section 5). The translated versions are split 80/20 into train and test sets.

Total dataset variants: 5 domains x 5 languages = 25 datasets, all published on HuggingFace Hub.

## 4. Experiment Design

### 4.1 Independent Variable

Language of the fine-tuning dataset: {English, Indonesian, Javanese, Sundanese, Minangkabau, None (base model)}

### 4.2 Dependent Variable

Accuracy / quality score on domain evaluation tasks, measured per evaluation language via Inspect AI.

### 4.3 Controls

- Same base model across all conditions (selected from Adaption's `training_models.list()`)
- Same domain content across languages (parallel translations)
- Same number of training examples per condition
- AutoScientist handles hyperparameter optimization per run, but `max_iterations` and `target_win_rate` are held constant

### 4.4 Experiment Matrix (per domain)

| Fine-tune Language | Eval: eng | Eval: ind | Eval: jav | Eval: sun | Eval: min |
|---|---|---|---|---|---|
| None (base) | x | x | x | x | x |
| English | x | x | x | x | x |
| Indonesian | x | x | x | x | x |
| Javanese | x | x | x | x | x |
| Sundanese | x | x | x | x | x |
| Minangkabau | x | x | x | x | x |

Per domain: 6 model variants x 5 evaluation languages = 30 evaluation runs.

Across all 5 domains: **150 total evaluation runs**.

### 4.5 Metrics

- **Primary:** Accuracy (exact match for MCQ, numerical match for math, pass@1 for code)
- **Secondary:** Model-graded quality score via Inspect AI's `model_graded_qa` scorer (assesses medical accuracy, language fluency, cultural appropriateness)
- **Statistical:** Paired bootstrap test for significance, effect size (Cohen's d)

## 5. Toolchain

### 5.1 Data Translation: Adaption Datasets API

The existing `mgsm_convert.py` in this repository demonstrates the pattern:

```python
from adaption import Adaption

client = Adaption()
# Upload source dataset
dataset_id = client.datasets.upload_file(path, name="medical-qa-en").dataset_id
# Wait for processing
client.datasets.wait_for_completion(dataset_id)
# Run translation
client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "question", "completion": "answer"},
    brand_controls={"blueprint": "Translate into Javanese. Preserve medical terms and numbers."},
)
# Download translated dataset
result = client.datasets.download(dataset_id)
```

The `datasets.run()` method supports `language_expansion` and `brand_controls` parameters for controlling translation behavior. Each domain dataset gets a domain-specific blueprint (e.g., "Preserve medical terminology" for medical, "Keep code blocks unchanged" for programming).

### 5.2 Fine-Tuning: Adaption AutoScientist

AutoScientist automates the full fine-tuning loop: data augmentation, hyperparameter selection, iterative training with checkpoint evaluation, and model artifact export.

```python
from adaption import Adaption

client = Adaption()

# List available base models
models = client.training_models.list()

# Start AutoScientist training
run = client.autoscientist.create(
    dataset_id="<translated-dataset-id>",
    model="<base-model-from-list>",
    data_format="instruction",
    column_mapping={"prompt": "question", "completion": "answer"},
    max_iterations=3,
    target_win_rate=0.7,
    augmentation_domain_rows=500,  # synthetic domain-targeted augmentation
)

# Wait for completion
result = client.autoscientist.wait_for_completion(run.id)

# Download best model artifact
artifact = client.autoscientist.download(run.id)
```

Key parameters:
- `model`: Base model ID from `training_models.list()`. AutoScientist selects one automatically if omitted.
- `max_iterations`: Number of training cycles (default 3). Each iteration evaluates against a validation set and can generate targeted augmentation data.
- `target_win_rate`: Early stopping threshold (default 0.7 for small models, 0.8 for larger).
- `augmentation_domain_rows` / `augmentation_general_rows`: Number of synthetic rows to generate before training.
- `hyperparams`: Optional overrides for lora_r, lora_alpha, learning_rate, n_epochs, batch_size, etc. Not recommended since AutoScientist derives these.

### 5.3 Evaluation: Inspect AI

Inspect AI is the UK AI Safety Institute's open-source Python framework for LLM evaluations. It provides a task/solver/scorer architecture with built-in support for model-graded evaluation, sandboxed code execution, and logging.

```python
from inspect_ai import task, Task
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import exact_match, model_graded_qa
from inspect_ai.solver import generate

@task
def medical_qa(language: str = "eng"):
    return Task(
        dataset=hf_dataset(f"sanggusti/medical-qa-{language}", split="test"),
        solver=generate(),
        scorer=[exact_match(), model_graded_qa()],
    )
```

Domain-specific scorer mapping:
- Medical, Science, General: `exact_match` for MCQ
- Mathematics: `exact_match` on numerical answer extraction
- Programming: code execution sandbox via Inspect's Docker integration

### 5.4 Artifact Publishing: HuggingFace Hub

All outputs are published to HuggingFace:
- Datasets: `sanggusti/{domain}-qa-{language}` (e.g., `sanggusti/medical-qa-javanese`)
- Models: `sanggusti/{domain}-{language}-finetuned` (e.g., `sanggusti/medical-javanese-finetuned`)

## 6. Phases and Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| 1. Foundation | Week 1-2 | Inspect AI setup, Adaption SDK verified, stale code removed |
| 2. Data Curation | Week 2-5 | 25 translated datasets on HF Hub, translation quality validated |
| 3. Eval Framework | Week 4-6 | Inspect AI tasks for all 5 domains, baseline results collected |
| 4. Fine-Tuning | Week 6-8 | 25 AutoScientist training runs, model artifacts on HF Hub |
| 5. Analysis | Week 8-10 | 150 eval runs, statistical analysis, experiment report |

## 7. Related Work

### Low-Resource Indonesian Language Evaluation

**IndoSafety** (arXiv:2506.02573): The first safety evaluation dataset covering Javanese, Sundanese, and Minangkabau. Shows that existing Indonesian LLMs produce unsafe outputs in local language settings, and fine-tuning on safety data improves performance while preserving task accuracy.

**BHASA** (arXiv:2309.06085): Southeast Asian LLM evaluation suite. One of the first studies evaluating ChatGPT on Indonesian sentiment analysis and machine translation between English and Indonesian/Javanese/Sundanese.

**Culturally-Nuanced Story Generation** (arXiv:2502.12932): Used QLoRA to fine-tune Sahabat.AI for Javanese and Sundanese story generation. Found that machine translation baselines outperform LLM-generated datasets on ROUGE-L and METEOR for Javanese.

### Low-Resource Language LLM Evaluation Frameworks

**LLM Probe** (arXiv:2603.29517): Framework for evaluating LLMs on low-resource languages. References GlotEval (massively multilingual evaluation) and Eka-Eval (Indian languages).

**Inspect AI** (UK AISI): Open-source Python framework with 100+ pre-built benchmarks, task/solver/scorer architecture, sandboxed execution, and community contribution system.

### Fine-Tuning for Low-Resource Languages

**Basque Study** (arXiv:2506.07597): Systematic study of instruction-tuning strategies combining continued pretraining, synthetic data, and translated instructions. Shows that even latest instruction-tuned models show degraded capabilities on low-resource languages.

**LoRA Challenges** (CHiPSAL 2025): Documents challenges in adapting multilingual LLMs to low-resource languages using LoRA PEFT tuning, including catastrophic forgetting and overfitting on limited datasets.

**Dictionary-Guided Fine-Tuning with RL** (arXiv:2508.19481): Explores reinforcement learning alongside dictionary-based methods for extremely low-resource translation (Spanish to Wayuunaiki).

### Key Finding from Literature

No existing benchmark combines low-resource Indonesian language evaluation with multi-domain fine-tuning analysis. This research fills that gap by systematically testing cross-lingual transfer across 5 task domains.

## 8. Repository Structure (Target)

```
qc_native_preference_tuning/
  configs/
    datagenerator/
      translate/
        medical.yaml
        maths.yaml
        programming.yaml
        science.yaml
        general.yaml
    evals/
      medical_qa.yaml
      maths.yaml
      ...
    training/
      autoscientist.yaml
  pipeline/
    datagenerator/
      translate/
        domain_convert.py       # Reusable Adaption translation pipeline
    training/
      autoscientist_finetune.py # AutoScientist training wrapper
  src/
    evals/
      tasks/
        medical_qa.py           # Inspect AI task definitions
        maths.py
        programming.py
        science.py
        general.py
  docs/
    research.md                 # This document
    experiments.md              # Results (filled after Phase 5)
    multilingual_tasks.md       # Benchmark survey (existing)
```
