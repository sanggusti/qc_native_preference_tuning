---
language:
  - {language}
tags:
  - translated-qa
  - {benchmark}
license: {source_license}
---

# Dataset card: {benchmark}-{language}

Translated {benchmark} ({language}, {register}). Train rows: {train_rows}. Test rows: {test_rows}.

## Translation

Blueprint sha {blueprint_hash}. Translator runs {translator_runs} on {translator_date}. Register: {register}.

## Gate results

{gate_summary}

## Calibration

{calibration}

## Excluded items

{excluded_ids}

## Source license

{source_license}. NLLB-derived rows are CC-BY-NC.

## Canary

Test splits embed this canary string for future contamination detection: {canary}

## Do not train

Do not train on the test split.

## Annotators

{annotators}
