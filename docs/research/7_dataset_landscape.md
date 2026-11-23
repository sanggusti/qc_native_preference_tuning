# 7. Dataset Landscape (November 2026)

Audit of every external dataset the design depends on: pinned sources are
verified live (Hub API plus streamed rows under current `datasets`), stale or
broken ones are fixed below, and candidates are assessed for Phase 2. All Hub
ids, schemas, counts and licenses in sections 1-3 were fetched in November
2026; anything else is marked UNVERIFIED.

## 1 Pinned benchmark sources

| Source | Status | Evidence |
|---|---|---|
| openai/gsm8k | Keep, canonical | MIT, parquet; train 7473, test 1319; `####` format intact |
| bigbio/med_qa | Keep with fix | parquet; 4-option config `med_qa_en_4options_bigbio_qa` verified (train 10178, test 1273); the default config has 5 options and does not match the choice scorer, so the registry pins the 4-option config |
| allenai/ai2_arc ARC-Challenge | Keep | CC-BY-SA-4.0; train 1119, test 1172; stable `id`; oldest touch but no newer ARC source exists |
| cais/mmlu | Keep for train | MIT; `all` config verified (question/subject/choices/answer); test 14042, auxiliary_train 99842; registry maps cais columns, translated rows keep the MMMLU layout |
| openai/openai_humaneval | Keep for eval | MIT, test 164, no train split (registry marks eval-only, min_rows 0) |
| evalplus/humanevalplus | Eval source | Apache-2.0; same 164 task_ids as HumanEval with strictly stronger tests; the registry eval_source |
| openai/MMMLU | Reference only | MIT; ID_ID confirmed but no jv/su/min, machine-translated, no stable ids; superseded for Indonesian eval by Global-MMLU |
| CohereLabs/Global-MMLU | Indonesian MMLU eval | Apache-2.0; id test 14042 with stable sample_ids, human-verified; Lite (400 rows) for smoke runs |
| juletxara/mgsm | Evals only | CC-BY-SA-4.0 community mirror (official data is GitHub-only, mirror fidelity UNVERIFIED); en test 250 usable for id tagging; train is 8 exemplars, never finetune training data |

## 2 Candidates assessed

Add as harder general eval: TIGER-Lab/MMLU-Pro (MIT, test 12032, 10 options;
needs the 10-letter target path). Add as large medical-train option:
openlifescienceai/medmcqa (Apache-2.0, train 182822; Indian-exam source, a
noted domain shift from USMLE). Add as second code eval: evalplus/mbppplus
(Apache-2.0, test 378, test-only). Do not adopt: GSM-Symbolic for training
or translation (CC-BY-NC-ND bars derivatives; our -pro1 reimplements the
idea on GSM8K instead), MMedBench and CareQA (no Indonesian), ARC-AGI-2
(no language dimension), MMLU-Plus (no canonical dataset exists), BBH
except via the Joschka parquet mirror with per-task scorer work in Phase 2.

## 3 Multilingual resources

| Dependency | Status |
|---|---|
| facebook/flores | Gated (access request required); all 9 codes plus Arabic-script variants confirmed present. Code falls back to the original release tarball (layout verified: `flores200_dataset/{dev,devtest}/{code}.{dev,devtest}`, 1012 lines) when Hub access fails |
| openlanguagedata/flores_plus | Sanctioned successor but also gated; contents beyond file listing UNVERIFIED |
| indonlp/NusaX-MT, NusaX-senti | Legacy scripts unloadable under datasets>=4; similarity reads the published test.csv from GitHub raw instead (12 parallel columns incl. all 9 project codes) |
| mteb/NusaXBitextMining, mteb/NusaX-senti | Parquet mirrors (eng-X pairs; senti per-language splits) as backup sources |
| facebook/belebele | Loads; id/jv/su only, min/ace/ban/bjn/bug confirmed missing; 900 rows each |
| GlotLID | Pin `cis-lmu/glotlid` `model_v4_small.ftz` (35MB); labels parse as `__label__<iso3>_<Script>`; bjn (F1 0.935) and ind (precision 0.724) are verified weak spots needing elevated thresholds |
| facebook/nllb-200-3.3B | Full coverage of all 9 codes verified in the card language list; keep |
| SEACrowd hub | Script-based, unloadable; catalogue only |
| Aya Collection | Loads (parquet) but instruction data without Buginese; side resource |
| Cendol/NusaWrites | No 2026 release; legacy scripts or unverified mirrors only |
| Newer ban/bjn/bug/min/ace parallel data | None found; NusaX (2022) remains the only human parallel corpus covering all five (Minangkabau CSV mirror exists but looks MT-generated, UNVERIFIED) |

## 4 What stays unverified

FLORES+ contents (gated), MGSM mirror fidelity against official data, Joschka
BBH per-task scorer details, and any 2026 SEA corpus released after this
audit. All four are Phase 2 checks, not assumptions.
