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

- **Accuracy / Pass@1**: For multiple-choice questions (like Global-MMLU-Lite, MMLU-ProX) or deterministic math answers (MGSM), the primary metric is straight accuracy, whether the model's final generated answer exactly matches the ground truth.

- **CHaRacter-level F-score (chrF)**: Because word boundaries and tokenization vary wildly across different scripts (e.g., Chinese characters vs. Arabic script), models like Gemma 3 use the CHaRacter-level F-score instead of word-level metrics. This is the standard metric applied to FLoRes, WMT24++, XQuAD, XOR QA, and IndicGenBench to precisely measure translation and generation quality.

- **Threshold-Based Real-World Usability**: For multilingual visual tasks like OCR, developers plot accuracy percentages per language. For instance, Qwen3-VL developers set an evaluation property of 70% accuracy as the threshold to determine if the model has "strong and usable" support for a specific language in real-world scenarios.

## Qwen3-VL Datasets

### Pre-training Datasets

- **Image-Caption & Interleaved Data**: Web-sourced multilingual (predominantly Chinese-English) image-text pairs
  - Interleaved text-image sequences sourced from global websites and book-scale datasets

- **Knowledge & Grounding**: An entity-centric world knowledge dataset spanning categories like landmarks, animals, plants, and vehicles
  - Box-based and point-based grounding datasets including COCO, Objects365, OpenImages, RefCOCO/+/g, and PixMo

- **OCR & Document Parsing**: 30 million in-house multilingual OCR samples and 3 million PDFs from Common Crawl
  - Synthetic HTML and Markdown layout corpora

- **STEM & 3D Spatial Understanding**: Synthetic geometric diagrams (K-12 and undergraduate level) and long Chain-of-Thought (CoT) problem-solving data
  - 3D visual grounding datasets (unified via Omni3D) and 2D spatial relationship/affordance annotations

- **Code & Video**: Qwen3/Qwen3-Coder text-only data, UI screenshots to HTML/CSS/SVG, flowcharts, and diagrams
  - Diverse video sources (instructional, cinematic, egocentric) with dense caption synthesis

- **Agent Data**: Multi-step GUI trajectories (desktop, mobile, web), function calling interactions, and online search datasets

### Post-training Datasets (SFT & RL)

- **SFT**: ~1.2 million samples (1/3 text-only, 2/3 image-text and video-text), incorporating multilingual dialogue and multi-turn interactions
  - A curated Long-CoT "cold start" dataset for thinking models focusing on STEM, agentic workflows, math, and code

- **RL**: Verifiable reasoning datasets (math, coding, logical reasoning, visual grounding, visual puzzles)
  - General RL data covers OCR, VQA, image captioning, and document parsing
  - ~130K multi-turn agentic interactions generated through distillation of visual agents ("Thinking with Images")

### Evaluations

- **Reasoning & STEM**: MMMU, MMMU-Pro, MathVision, MathVista, We-Math, MathVerse, DynaMath, Math-VR, LogicVista, VisualPuzzles, ZeroBench, VisuLogic, VLM Blind

- **General VQA & Docs**: MMBench, RealWorldQA, MMStar, SimpleVQA, DocVQA, InfoVQA, AI2D, ChartQA, OCRBench, CC-OCR, OmniDocBench, CharXiv, MMLongBench-Doc

- **Grounding & Spatial**: RefCOCO/+/g, CountBench, ODinW-13, ARKitScenes, Hypersim, SUNRGBD, ERQA, VSI-Bench, EmbSpatialBench, RefSpatialBench, RoboSpatialHome

- **Multi-Image & Video**: BLINK, MUIRBENCH, MVBench, Video-MME, MLVUM, LVBench, Charades-STA, VideoMMMU, MMVU

- **Agent & Coding**: ScreenSpot Pro, OSWorld, AndroidWorld, Design2Code, ChartMimic, UniSVG, V*, HRBench

## Olmo 3

### Pre-training, Mid-training, & Long-Context Datasets

- **Dolma 3 Mix** (Pre-training 6T tokens): Common Crawl (76.1%), olmOCR science PDFs (13.6%), Stack-Edu rebalanced GitHub code (6.89%), FineMath 3+ (2.56%), arXiv (0.86%), and Wikipedia & Wikibooks (0.04%)

- **Dolma 3 Dolmino Mix** (Mid-training 100B tokens):
  - **Math**: TinyMATH, CraneMath, MegaMatt, Dolmino Math
  - **Code**: StackEdu (FIM), CraneCode
  - **QA**: Reddit To Flashcards, Wiki To RCQA, Nemotron Synth QA
  - **Thinking Traces**: Meta-Reasoning, Program-Verifiable, OMR Rewrite FullThoughts, and reasoning traces distilled from QWQ, Gemini, Llama Nemotron, and OpenThoughts
  - **General**: STEM-Heavy Crawl, olmOCR high-quality subset, Tulu 3 SFT, Dolmino 1 Flan

- **Dolma 3 Longmino Mix** (Long-Context): olmOCR PDFs up to 1M tokens, synthetic CWE/REX tasks, combined with midtraining data

### Post-training Datasets (SFT, DPO, RL)

- **Dolci Think SFT & Instruct SFT**: OpenThoughts 3+, SYNTHETIC-2, Nemotron Code, Python Algorithms, WildChat, OpenAssistant, CoCoNot, WildGuardMix, WildJailbreak, Aya, TableGPT

- **Instruct-Specific**: SimFC (simulated function calling), ScienceQA, Web Search QA, FLAN, DaringAnteater, UltraFeedback

- **Dolci Think RL & RL-Zero**: IF-RLVR, Open-Reasoner-Zero, DAPO-Math, AceReason-Math, Polaris, KlearReasoner, OMEGA, AceCoder, Tulu 3, WildChat

### Evaluations

- **OlmoBaseEval Suites**: MMLU, ARC, CSQA, HellaSwag, WinoGrande, SocialIQA, PiQA, CoQA, DROP, Jeopardy, NaturalQs, SQuAD, SciQ, QASPER, Basic Skills, DBQA, ProtocolQA, Lambada, MedMCQA, MedQA, SciRIFF, GSM8K, MATH, HumanEval, MBPP, DS 1000, Deepseek LeetCode, MultiPL-E, LBPP, Deepmind Math, BigBench Hard

- **Long-Context & Chat**: RULER, HELMET, AlpacaEval, IFEval, IFBench, PopQA, GPQA, ZebraLogic, AGI Eval, LiveCodeBench, SimpleQA, LitQA2, BFCL

- **Safety**: DoAnythingNow (DAN), HarmBench, TrustLLM-JailbreakTrigger, WildJailbreak, WildGuard-Test, XSTest, BBQ, StrongReject, Toxigen, WMDP

## Kimi-VL

### Pre-training Datasets

- **Caption & Interleaved Data**: Open-source captions (e.g., LAION), in-house captions, textbooks, webpages, and datasets like Obelics

- **OCR & Knowledge**: In-house multilingual, dense, web, and handwritten OCR datasets. Knowledge data capturing geometry diagrams and academic papers

- **Video & Agent**: Web-scale open-source and in-house videos. Agent grounding data utilizing Desktop, Mobile, and Web screenshots, plus Aguvis (computer-use trajectories with synthesized CoT)

- **Pure Text**: The Moonlight language model corpus

### Post-training Datasets

- **Vanilla SFT**: Human-annotated multimodal instruction data. Rejection sampling was heavily utilized for domains where automated verification is viable (visual coding, visual reasoning, math/science)

- **Reasoning Data (Long-CoT & RL)**: A curated "warmup" dataset featuring verified reasoning paths (planning, evaluation, reflection, exploration). Advanced Long-CoT data synthesized by performing rejection sampling using Kimi k1.5 on math and domain-specific VQA tasks

### Evaluations

- **Vision & Reasoning**: MMMU, VideoMMMU, MMVU, MMBench, MMStar, MMVet, RealWorldQA, AI2D, BLINK, MathVista, MathVision, InfoVQA, OCRBench

- **Agent & OS Grounding**: ScreenSpot V2, ScreenSpot Pro, OSWorld, WindowsAgentArena

- **Long-form & Video**: MMLongBench-Doc, Video-MME, MLVU, LongVideoBench, EgoSchema, VSI-Bench, TOMATO
