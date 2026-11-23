# 6. Model Landscape (November 2026)

Survey of current open models for two project roles: the gate judge (error-span
annotation, served on one Modal L40S 48GB via vLLM) and finetuning bases
(3B-12B, instruct, LoRA-friendly, some Indonesian coverage). Every id below
returned HTTP 200 from the Hub API in November 2026; quotes are from official
model cards fetched in the same pass. VRAM fits are computed (bf16 ≈ 2 bytes
per parameter plus overhead) and still require the smoke run.

## 1 Judge decision

Pinned judge: `google/gemma-4-12B-it` on L40S (methodology section 6). 12B
dense, 256K context, Apache-2.0, ungated, vLLM-supported, 1.5M downloads,
140+ pretraining languages with 35+ supported out of the box, system prompt
and function-calling support. bf16 fits one L40S natively, which keeps the
quality gate free of quantization effects.

Challengers, in order, with why each loses today:

1. `Qwen/Qwen3.5-9B` — newest small Qwen post-trained model (thinking
   disable-able for clean JSON), 201 languages, 262K context, Apache-2.0,
   bf16 fits even an A10G. Loses on reasoning depth for span annotation
   (9B vs 12B) at judge-batch sizes where the cost gap is cents; first
   fallback if Gemma 4 disappoints on the 50-sample probe.
2. `mistralai/Mistral-Small-3.2-24B-Instruct-2506` — the only judge candidate
   with card-declared `id` and `ms` support plus explicit instruction-following
   gains, Apache-2.0, vLLM-native. Loses on serving: 24B needs FP8 on one L40S,
   and quantized span quality is unverified. Revisit with an FP8 smoke test.
3. `openai/gpt-oss-20b` — 21B/3.6B-active MoE, Apache-2.0, structured outputs,
   fits in 16GB (MXFP4). Loses on two counts: the card makes no multilingual
   claims (verify with a pilot first), and the harmony prompt format adds
   harness friction.

Explicitly rejected for judging: Qwen2.5-7B (superseded by two generations),
27B+ dense models (no bf16 fit), 671B-class MoEs (API scale only).

## 2 Base shortlist and the catalogue constraint

Live Adaption catalogue, November 2026 (16 models, `list_models`): the small
end is `google/gemma-3-4b-pt`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3.5-2B`,
`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` — all base/pretrained variants. The
smallest instruct model on the platform is `openai/gpt-oss-20b`. Neither
provisionally pinned base exists there (`google/gemma-3-4b-it`,
`meta-llama/Llama-3.2-3B-Instruct`).

Consequences for the Phase 2 decision (issue #63): same-size instruct
primary+contrast pairs are unavailable on the managed backend as listed.
Either the floor rule adapts to base variants (untuned probes cannot rely on
instruction following), Adaption is asked to list small instruct models, or
the transparent backend carries instruct models the managed loop cannot.
This note records the Hub-side shortlist; the decision stays with #63.

Hub-side base candidates (3B-12B, instruct, open license):

| Hub id | Params | Ctx | License | Indonesian evidence | Fit |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | 4B | 262K | Apache-2.0 | 201-language claim | newest arch, size-matched anchor |
| `Qwen/Qwen3.5-9B` | 9B | 262K | Apache-2.0 | 201-language claim | larger anchor |
| `google/gemma-4-E4B-it` | 8B tot / 4.5B eff | 128K | Apache-2.0 | 140+ family claim | current-gen, no gate |
| `aisingapore/Qwen-SEA-LION-v5-9B` | 9.7B | 262K | MIT | SEA post-training incl. Indonesian (no jv/su/min) | exposure-contrast base |
| `microsoft/Phi-4-mini-instruct` | 3.8B | 128K | MIT | multilingual label + MMMLU table | cheapest LoRA run |
| `Qwen/Qwen3-8B` | 8.2B | 32K | Apache-2.0 | 100+ languages claim | fallback, short context |

Rejected for bases: OLMo-3-7B (card says English-only), phi-4 14B (card
disclaims multilingual use), Ministral-8B (research-only license),
GLM-4-9B (zh/en only, custom license), Llama-4-Scout (explicit Indonesian
but 109B total and gated; revisit on 2xGPU), Llama-3.2-3B (no official
Indonesian; kept only as a size-matched control candidate).

## 3 Family notes

**Qwen.** Naming changed: Qwen3.5+ dropped `-Instruct` (plain repos are
post-trained; `-Base` is the base). Current small post-trained models:
Qwen3.5-4B/9B (201 languages, 262K, Apache-2.0, thinking disable-able).
Qwen3.0 dense (8B/14B) keeps 32K native context. MoE lines (30B/35B total)
need quant serving. Qwen2.5-7B is superseded for new work.

**Cohere.** Org renamed to `CohereLabs`. Aya Expanse 8B and Command R7B name
Indonesian explicitly but carry CC-BY-NC plus gating (Expanse also 8K
context). Command A+ 2026 is Apache-2.0 and names Indonesian/Malay but is
219B total. No Cohere winner for either role; tiny-aya-global (3.3B, 2026)
is too weak for judging and NC-gated for bases.

**Google Gemma.** Gemma 4 is Apache-2.0 and ungated (unlike Gemma 3's custom
license and manual gate): 12B instruct (judge), E4B (base), 26B-A4B MoE
(judge stretch in fp8), 31B (quant-only). Shared family card claims 140+
pretraining languages; per-language Indonesian lists are unverified.

**Allen AI2.** OLMo-3-7B-Instruct is English-only per its card; OLMo-3.1
exists only at 32B. No fit for Indonesian work; revisit if a multilingual
OLMo line appears.

**Meta Llama.** Llama-3.2-3B and 3.3-70B list 8 supported languages without
Indonesian. Llama-4-Scout names Indonesian explicitly but is 109B total and
gated: a 2xGPU revisit, not a current candidate.

**Mistral.** Mistral-Small-3.2-24B names `id`/`ms`, otherwise as section 1.
Ministral-8B is research-licensed without `id`: excluded.

**Others.** DeepSeek V3/V4 flagships are 671B-class (API scale); the
8B R1 distill inherits Qwen3 multilinguality unverified. gpt-oss-20b runs
in 16GB with structured outputs but no language claims. GLM small lines
are zh/en-only. Phi-4-mini is the cheapest multilingual base.

**Southeast Asia 2026 check.** Only SEA-LION v5 (November 2026, MIT,
Indonesian+Malay, no Javanese/Sundanese/Minangkabau) supersedes a 2024
baseline. Sailor2, SeaLLMs-v3, Cendol and Sahabat-AI have no 2026
successor as of this survey.

## 4 What stays unverified

Hub metadata cannot confirm: Indonesian (not family-level) judge accuracy
for any candidate (needs the 50-sample probe against human scores),
AutoScientist catalogue presence beyond the snapshot above, L40S vLLM
serving for each pick (needs the smoke run), or LoRA quality on
Gemma-4-E4B per-layer embeddings. All four are Phase 0/1 measurements,
not assumptions.
