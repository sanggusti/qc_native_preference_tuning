# Research Notes: Native Language Finetuning

Literature review supporting the native preference tuning project. Sources are arXiv papers, with conference venues (NeurIPS, ICLR, ACL, EMNLP) noted per reference. Figures under `figures/` are extracted from the cited papers' arXiv HTML renders and credited in captions.

1. [Introduction and Problem Statements](1_intro_problem_statements.md): motivation, research questions RQ1-RQ5, evidence of cross-lingual performance degradation by domain (general, math, medicine, code, physics), project problem statement and hypotheses.
2. [Preliminaries and Fundamentals](2_preliminaries_fundamentals.md): terminology, tokenization effects, encoder-era fundamentals, and the internal mechanics of multilingual LLMs (latent English, MWork, language neurons, semantic hub).
3. [Contexts and Available Methodologies](3_contexts_methodologies.md): continued pretraining, vocabulary extension, multilingual instruction tuning, English-anchored (pivot) training, cross-lingual preference optimization, modular methods, and the anchor-vs-native decision table.
4. [Available Datasets and Evaluation Tasks](4_datasets_evaluation.md): pretraining corpora, instruction/preference datasets, benchmarks per task family, evaluation pitfalls, and a suggested evaluation matrix.
