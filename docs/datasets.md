# Dataset coverage report

Finished datasets and coverage for the S01 data phase. Updated by the agent
whenever a dataset run completes: verify rows, gate verdict and Hub repo, then
update the tables below and the ledger in `docs/experiments.md` section 3.

Last updated: 2026-11-27 (all 8 gsm8k languages complete, gated,
leak-regens substituted and re-pushed; 6 admitted, bjn/bug provisional).

## Summary

| Set | Coverage | Rows | Status |
|---|---|---|---|
| FLORES-200 calibration | 9/9 languages (8 translated + en source) | 8,096 translated | complete, gated inputs |
| gsm8k train (1000 subset) | 8/8 complete (6 admitted, bjn/bug provisional) | 8,000 | done |
| gsm8k test (1319) | 8/8 complete (6 admitted, bjn/bug provisional) | 10,552 | done |
| medqa | parked (tier 3, issue #62) | 0 | not started |
| NLLB second-opinion set | pending Modal work | 0 | not started |

## FLORES-200 calibration (complete)

Full devtest (1,012 sentences) translated with the pinned blueprint route.
Local JSON only; no Hub push agreed for calibration sets yet. All wandb runs
backfilled (`datagen-flores-calibration-{lang}`).

| Lang | Rows | Adaption dataset id | Server display name | Status |
|---|---|---|---|---|
| id | 1012 | cadb6a23-fe21-457d-8c08-f5cfd0ed7bc3 | general_text_echo | verified id |
| jv | 1012 | 407e0307-a91a-4b82-bac7-94ee592652a7 | identity_text_pairs | verified jv |
| su | 1012 | 432a5c09-cf0e-4f2a-bbce-1a509338b645 | identity_text_pairs | verified su |
| min | 1012 | 9ebd2033-1761-41ea-ad84-6be7691a2120 | identity_text_pairs | verified min |
| ace | 1012 | bd216607-e028-4697-9fd8-ff7fe3a92dc1 | identity_text_pairs | verified ace |
| ban | 1012 | 4857aa7c-3fb7-46de-b77d-32a0da69a83a | identity_text_pairs | verified ban |
| bjn | 1012 | 7edd018f-1218-42f4-ae18-a44908802c4e | identical_sentence_pairs | verified bjn |
| bug | 1012 | a12b39ae-d939-4036-9cc5-38d79d3c1644 | factual_statements_corpus | verified bug |

Note: the server reuses display names from a small pool, so names repeat.
Dataset ids are the only reliable handle. Files: `data/calibration/`.

## gsm8k translation (in progress)

Seeded train subset (1,000, seed 47) plus full test (1,319) per language,
pushed to Hub as `sanggusti/math-qa-{lang}`.

| Lang | Train | Test | Hub repo | Gate | Status |
|---|---|---|---|---|---|
| jv | 1000 | 1319 | [math-qa-jv](https://huggingface.co/datasets/sanggusti/math-qa-jv) | lid 0.94/0.94, green | 2 leaks regenned, re-pushed |
| id | 1000 | 1319 | [math-qa-id](https://huggingface.co/datasets/sanggusti/math-qa-id) | lid 0.94/0.93, green | 0 leaks; 16 regenned, 3 repaired |
| su | 1000 | 1319 | [math-qa-su](https://huggingface.co/datasets/sanggusti/math-qa-su) | lid 0.98/0.98, green | 1 leak regenned, re-pushed |
| min | 1000 | 1319 | [math-qa-min](https://huggingface.co/datasets/sanggusti/math-qa-min) | lid 0.91/0.91, green | 0 leaks; test rerun after timeout |
| ace | 1000 | 1319 | [math-qa-ace](https://huggingface.co/datasets/sanggusti/math-qa-ace) | lid 0.95/0.96, green | 1 leak regenned, re-pushed |
| ban | 1000 | 1319 | [math-qa-ban](https://huggingface.co/datasets/sanggusti/math-qa-ban) | lid 0.93/0.93, green | 1 leak regenned, re-pushed |
| bjn | 1000 | 1319 | [math-qa-bjn](https://huggingface.co/datasets/sanggusti/math-qa-bjn) | lid 0.67/0.68, PROVISIONAL | 4 leaks regenned, re-pushed |
| bug | 1000 | 1319 | [math-qa-bug](https://huggingface.co/datasets/sanggusti/math-qa-bug) | lid 0.43/0.43, PROVISIONAL | 8 leaks regenned, re-pushed |

All Hub repos hold the final clean data (17 leak-regen rows substituted
across jv/su/bjn/ace/ban/bug and re-pushed). Local `data/translated/` files
match. Gate reports in `data/gate/` are current per split.

Provisional languages (tier-2 path, user-approved 2026-11-27): bjn translator
matches human references within 8 points (0.68 vs 0.76; the 0.90 bar is
unachievable for Banjar by anyone). bug translator underperforms references
(0.35 vs 0.84; emits pure Indonesian on hard sentences). Both ship flagged;
contamination is measured by the leak_id covariate when the NLLB set lands.

## Probes

| Probe | Rows | Result |
|---|---|---|
| id FLORES smoke (50) | 50/50 | clean Indonesian, no poisoning |
| gsm8k jv probe (50) | 50/50 | clean after leakage fix, gate green |
| gsm8k jv mini (5) | 5/5 | clean, instruction marker kept |

## Credit spend (estimates)

No billing endpoint exists; numbers are estimate responses and unit math
(~0.011 credits/row at full-run volume). Actual balance comes from the
Adaption dashboard (1687 at start).

| Item | Estimate |
|---|---|
| 8 FLORES calibrations (1012 rows each) | ~168 |
| Probes and mini-probes | ~10 |
| Failed full attempts (marker/numeral, ~7 splits) | ~150 |
| Contaminated full batch (4 languages, discarded) | ~190 |
| Clean batch, 8 languages (regen + repair incl.) | ~400 |
| Leak regens (17 rows, 5 datasets) | ~5 |
| Total spend (dashboard: 1130 used of 1687) | ~915 estimated, ~215 unaccounted |
| Remaining headroom | ~557 |

## Incidents

1. Solution leakage (Nov 2026). Translated questions contained full worked
   solutions (length ratio 2.2x). Cause: per-column runs exposed the sibling
   column and whole-item blueprints asked for it. Fix: neutral filler column,
   translate-only blueprints (PR #89).
2. Marker pin backfire (Nov 2026). Pinning `ANSWER:` in item blueprints primed
   solutions into ~20% of jv/su questions. Fix: instruction translates on its
   own 1-row dataset with the pinned blueprint; items never mention `ANSWER:`.
3. Source typo ghost numeral. gsm8k test-00278 contains `9for Marco`, which the
   numeral check demanded in translations. Fix: letter-boundary numeral regex.
4. Display-name reuse. The server renames datasets from a small pool
   (`identity_text_pairs` used 5 times). Track by dataset id only.

## Follow-ups

- Probe outputs overwrite full-split files (same path). Probe runs should write
  `gsm8k-{lang}-{split}-rows{N}.jsonl` instead.
- Row regen covers genuine defects; NLLB fallback and exclusion rungs need the
  Modal NLLB set first.
- medqa translation awaits the issue #62 second-benchmark decision.
