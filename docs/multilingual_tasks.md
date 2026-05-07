# Multilingual Evaluation tasks

- **Mathematical Reasoning**: MGSM, PolyMATH, XOR QA
  - MGSM: 11 languages (bn, de, en, es, ja, fr, ru, sw, te, th, zh); needs translation/replication for low-resource languages
  - PolyMATH: English only

- **Cross-Lingual Knowledge Transfer and Translation**: WMT24PP (google/wmt24pp)

- **Multilingual Instruction Following and Chat**: MultiIF (facebook/Multi-IF)
  - Languages: Chinese, English, French, Hindi, Italian, Portuguese, Russian, Spanish
  - Needs translation to Indonesian languages for evaluations

## Key Multilingual Benchmarks

- **General Knowledge & Chat**: Global-MMLU-Lite, MMLU-ProX, INCLUDE, MultiIF, and ECLeKTic (evaluating cross-lingual knowledge transfer)
- **Math**: MGSM (Multilingual Grade School Math) and PolyMATH
- **Translation & Reading Comprehension**: WMT24++, FLoRes, XQuAD, and XOR QA (IndicGenBench specifically tests generation capabilities in Indic languages)

## Evaluation Metrics (How Performance is Scored)

- **Accuracy / Pass@1**: For multiple-choice questions (like Global-MMLU-Lite, MMLU-ProX) or deterministic math answers (MGSM), the primary metric is straight accuracy—whether the model's final generated answer exactly matches the ground truth.

- **CHaRacter-level F-score (chrF)**: Because word boundaries and tokenization vary wildly across different scripts (e.g., Chinese characters vs. Arabic script), models like Gemma 3 use the CHaRacter-level F-score instead of word-level metrics. This is the standard metric applied to FLoRes, WMT24++, XQuAD, XOR QA, and IndicGenBench to precisely measure translation and generation quality.

- **Threshold-Based Real-World Usability**: For multilingual visual tasks like OCR, developers plot accuracy percentages per language. For instance, Qwen3-VL developers set an evaluation property of 70% accuracy as the threshold to determine if the model has "strong and usable" support for a specific language in real-world scenarios.
