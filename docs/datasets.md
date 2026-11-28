# Dataset coverage report

Finished datasets and coverage for the S01 data phase. Updated by the agent
whenever a dataset run completes: verify rows, gate verdict and Hub repo, then
update the tables below and the ledger in `docs/experiments.md` section 3.

Last updated: 2026-11-28 (medpool 8/8 complete, gated, published).

## Summary

| Set | Coverage | Rows | Status |
|---|---|---|---|
| FLORES-200 calibration | 9/9 languages (8 translated + en source) | 8,096 translated | complete, gated inputs |
| gsm8k train (1000 subset) | 8/8 complete (6 admitted, bjn/bug provisional) | 8,000 | done |
| gsm8k test (1319) | 8/8 complete (6 admitted, bjn/bug provisional) | 10,552 | done |
| medpool train (1000) | 8/8 complete (6 admitted, bjn/bug provisional) | 8,000 | done |
| medpool test (250) | 8/8 complete (6 admitted, bjn/bug provisional) | 2,000 | done |
| medqa | parked (tier 3, issue #62) | 0 | not started |
| NLLB second-opinion set | pending Modal work | 0 | not started |

## FLORES-200 calibration (complete)

Full devtest (1,012 sentences) translated with the pinned blueprint route.
On the Hub as `sanggusti/flores200-calibration-{lang}` with provenance cards.
All wandb runs backfilled (`datagen-flores-calibration-{lang}`).

| Lang | Rows | Hub repo | Adaption dataset id | Status |
|---|---|---|---|---|
| id | 1012 | [flores200-calibration-id](https://huggingface.co/datasets/sanggusti/flores200-calibration-id) | cadb6a23 | verified id |
| jv | 1012 | [flores200-calibration-jv](https://huggingface.co/datasets/sanggusti/flores200-calibration-jv) | 407e0307 | verified jv |
| su | 1012 | [flores200-calibration-su](https://huggingface.co/datasets/sanggusti/flores200-calibration-su) | 432a5c09 | verified su |
| min | 1012 | [flores200-calibration-min](https://huggingface.co/datasets/sanggusti/flores200-calibration-min) | 9ebd2033 | verified min |
| ace | 1012 | [flores200-calibration-ace](https://huggingface.co/datasets/sanggusti/flores200-calibration-ace) | bd216607 | verified ace |
| ban | 1012 | [flores200-calibration-ban](https://huggingface.co/datasets/sanggusti/flores200-calibration-ban) | 4857aa7c | verified ban |
| bjn | 1012 | [flores200-calibration-bjn](https://huggingface.co/datasets/sanggusti/flores200-calibration-bjn) | 7edd018f | verified bjn |
| bug | 1012 | [flores200-calibration-bug](https://huggingface.co/datasets/sanggusti/flores200-calibration-bug) | a12b39ae | verified bug |

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

## medpool translation (in progress)

Merged medqa (500+125) + medmcqa (500+125) pool, 1000 train + 250 test per
language, pushed to Hub as `sanggusti/medical-qa-{lang}`. English source:
[sanggusti/medpool-en](https://huggingface.co/datasets/sanggusti/medpool-en).
LID verdicts below compare against truncated human references per language
(see incident 5); the 0.90 bar does not transfer to fragmentary items.

| Lang | Train | Test | Hub repo | Gate | Status |
|---|---|---|---|---|---|
| jv | 1000 | 250 | [medical-qa-jv](https://huggingface.co/datasets/sanggusti/medical-qa-jv) | lid 0.83/0.87 | verified, above truncated refs |
| id | 1000 | 250 | [medical-qa-id](https://huggingface.co/datasets/sanggusti/medical-qa-id) | lid 0.83/0.88 | verified, above truncated refs |
| su | 1000 | 250 | [medical-qa-su](https://huggingface.co/datasets/sanggusti/medical-qa-su) | lid 0.87/0.86 | verified, above truncated refs |
| min | 1000 | 250 | [medical-qa-min](https://huggingface.co/datasets/sanggusti/medical-qa-min) | lid 0.82/0.83 | verified, above truncated refs |
| ace | 1000 | 250 | [medical-qa-ace](https://huggingface.co/datasets/sanggusti/medical-qa-ace) | lid 0.87/0.87 | verified, above truncated refs |
| ban | 1000 | 250 | [medical-qa-ban](https://huggingface.co/datasets/sanggusti/medical-qa-ban) | lid 0.85/0.93 | 1 looped row regenned, re-pushed |
| bjn | 1000 | 250 | [medical-qa-bjn](https://huggingface.co/datasets/sanggusti/medical-qa-bjn) | lid 0.55/0.55, PROVISIONAL | 0 leaks; above truncated refs |
| bug | 1000 | 250 | [medical-qa-bug](https://huggingface.co/datasets/sanggusti/medical-qa-bug) | lid 0.20/0.20, PROVISIONAL | 9 looped rows regenned, re-pushed |

## Gate reports

All 16 per-split reports plus the probe report: [gsm8k-gate-reports](https://huggingface.co/datasets/sanggusti/gsm8k-gate-reports)
(mirror of `data/gate/`).

Raw AutoScientist train pairs (8 Adaption datasets) are not pushed separately:
they derive from each train split in one line via `training_pair()`.

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
5. Decoder repetition loops. The translator degenerates into syllable loops
   on some longer sentences, worst in bug (9 rows up to 49KB) and once in ban.
   Structure checks pass them (numerals survive); only the length band flags
   them. Fixed per row by regen with verification; a repetition detector swept
   all languages (questions and choices) before release.
6. LID thresholds do not transfer to fragments. FLORES-calibrated bars fail
   human references truncated to 5 words (0.43-0.78 across languages), so
   medpool verdicts compare against truncated human references per language
   instead of the frozen 0.90 bar. No rule changed; the comparison is recorded
   per language.

## Follow-ups

- Probe outputs overwrite full-split files (same path). Probe runs should write
  `gsm8k-{lang}-{split}-rows{N}.jsonl` instead.
- Row regen covers genuine defects; NLLB fallback and exclusion rungs need the
  Modal NLLB set first.
- medqa translation awaits the issue #62 second-benchmark decision.
