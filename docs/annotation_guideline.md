# Error-span annotation guideline (S01 human review)

Two native annotators review 150 items per (benchmark, language) test split: the 50 highest-risk rows plus 100 uniformly random rows whose ids are shared across languages. Training splits get 100 random rows plus the bottom 5 percent by risk. Annotators work independently in the exported JSONL or CSV, then scores merge in `review_import.py`.

## 1 What to mark

For each item, read the English source, then the translation. Mark every error as a span: the minimal character range covering the error, with a severity and a category.

Severities:

- minor: small fluency or style slip that a reader easily looks past (wrong particle, awkward word order, punctuation).
- major: meaning changed, added, or lost, but the item is still answerable (mistranslated clause, wrong entity, dropped condition).
- critical: the item is unanswerable or the answer key no longer holds (negation flipped, numbers changed, correct option corrupted).

Categories: accuracy, fluency, terminology, style, locale (numbers, units, date formats), other.

## 2 Scores and verdicts per item

- esa_score, 0 to 100: your direct judgment of overall translation quality. 90 to 100 is clean publishable text, 70 to 89 usable with small issues, 40 to 69 serious issues, below 40 wrong or misleading. The score is your judgment, not a formula over spans; the spans are the evidence.
- answer_preserved, true or false: the translated item is still correct and has a unique answer matching the source key. Any doubt means false.
- spans: the list of {start, end, severity, category, note}. An item with no errors has an empty list.

## 3 Pinned registers

Translate and judge in exactly these registers. A correct translation in the wrong register is a major style error.

| Code | Language | Pinned register |
|---|---|---|
| en | English | neutral standard English |
| id | Indonesian | standard formal Indonesian |
| jv | Javanese | ngoko (everyday; not madya or krama) |
| su | Sundanese | loma (everyday; not lemes) |
| min | Minangkabau | standard written Minangkabau, Latin script |
| ace | Acehnese | standard written Acehnese, Latin script |
| ban | Balinese | standard written Balinese, Latin script |
| bjn | Banjarese | standard written Banjarese, Latin script |
| bug | Buginese | standard written Buginese, Latin script |

## 4 Borrowing policy for medical terminology

A medical term with no established equivalent in the target language is kept in Indonesian or Latin and is not an error. Examples: drug names (paracetamol), anatomy in Latin (myocardium), lab values with units. Mark terminology errors only when an established native term exists and was missed, or when a kept term is wrong for the context. The kept terms feed the borrowed_term_share covariate; do not inflate severity to punish borrowing.

## 5 Worked examples

Each example shows source, translation with a marked span, severity, and verdict. Scores are illustrative.

en (reference, no annotation): source and translation are identical by construction.

id: source "The patient has a fever of 39C." translation "Pasien mengalami demam 39C." No span. esa 95, preserved true.

jv: source "Budi has 3 apples." translation "Budi duwe 3 apel [lan tuku → MAJOR accuracy: added clause not in source]." esa 55, preserved true (answer intact).

su: source "Which organ produces energy?" translation "Organel naon nu ngahasilkeun [tnaga → minor fluency: typo for tanaga]?" esa 88, preserved true.

min: source "She reads every night." translation "Inyo mambaco [satiok malam → minor style: Indonesian satiok; Minangkabau saban]." esa 85, preserved true.

ace: source "Total: 1500 rupiah." translation "Jumeulah: [1.500 → CRITICAL locale: numeral rewritten]." esa 30, preserved false.

ban: source "Choose option B." translation "Pilih [pilihan B → minor style: redundant]." esa 90, preserved true.

bjn: source "The river floods yearly." translation "Sungai [banjir saban tahun → no error]." No span. esa 95, preserved true.

bug: source "Do not take on an empty stomach." translation "[Minung nalikan weteng kosong → CRITICAL accuracy: negation dropped]." esa 20, preserved false.

## 6 Agreement and provisional languages

Agreement between the two annotators is reported as Cohen's kappa on answer_preserved, Pearson r on esa_score, and mean absolute esa difference. Systematic disagreement goes back for adjudication; the merged verdict is conservative (both must agree the answer is preserved).

Where two native reviewers cannot be recruited (expected for min, ace, ban, bjn, bug), one reviewer suffices and the language is flagged provisional in the card, scheduled last in its tier, and reported in a separate column.

## 7 Upper bounds reported with the counts

0 answer-changed in 100 gives a 3.0 percent one-sided 95 percent upper bound; 3 major-or-critical in 100 gives 7.6 percent. Bounds come from `review_import.error_upper_bound` (exact Clopper-Pearson), never by hand.
