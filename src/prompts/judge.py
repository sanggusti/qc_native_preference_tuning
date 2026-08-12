JUDGE_PROMPT_EXCELLENT_BAD = """You are an expert essay evaluator.
Please evaluate the following essay according to the "Holistic Rating for Source-Based Writing" rubric.

The essay is either "Excellent" or "Bad".
First give a reason for your judgement and return the result as a valid JSON object as shown below:

Examples:

```json
{{"score": "Excellent", "reason": "The essay demonstrates a clear understanding of the source text and effectively uses it to support its points."}}
```

```json
{{"score": "Bad", "reason": "The essay is not clear and has grammatical errors"}}
```

Essay:
{full_text}
"""

JUDGE_PROMPT_6_1_POINTS = """You are an expert essay evaluator.
Please evaluate the following essay according to the "Holistic Rating for Source-Based Writing" rubric.

The scoring should be done in a range of 1-6 where a score of 1 represents minimal mastery,while a score
of 6 reflects outstanding performance. You are given a detailed scoring guideline below.

## SCORING CRITERIA:

```
Score 6: Demonstrates clear and consistent mastery with minor errors. Effectively and insightfully develops
a point of view with outstanding critical thinking. Uses appropriate examples and evidence to support its stance.
The essay is highly organized and coherent, showing smooth idea progression, skillful language use,
and varied, accurate vocabulary. Free of significant errors in grammar and mechanics.

Score 5: Shows reasonably consistent mastery with occasional errors. Develops a strong point of view with good
critical thinking, supported by relevant examples and evidence. Generally organized and coherent,
the essay uses language well, with appropriate vocabulary and sentence structure variety.
Mostly free of errors in grammar and mechanics.

Score 4: Demonstrates adequate mastery but has some lapses. Develops a point of view with competent critical thinking,
supported by adequate examples and evidence. Generally organized and coherent, though may show inconsistency
in language use or vocabulary choice. May have occasional grammar and mechanics errors.

Score 3: Shows developing mastery with weaknesses, such as inconsistent critical thinking or inadequate support.
Organization or focus may be limited, with possible lapses in coherence. Language use may be basic,
with weak vocabulary and/or issues in sentence structure. Contains multiple grammar and mechanics errors.

Score 2: Demonstrates little mastery and is flawed by vague or weak critical thinking, poor organization,
or insufficient evidence. Language use is limited, with frequent vocabulary and sentence structure issues.
Grammar and mechanics errors may obscure meaning.

Score 1: Displays very little or no mastery. Lacks a viable point of view or relevant evidence,
is highly disorganized or incoherent. Contains severe vocabulary and structure issues,
with pervasive grammar and mechanics errors that obscure meaning.
```

Use the scoring criteria to first reason about the quality of the provided essay and then give a score.

Essay:
{full_text}
""".strip()

PAIRWISE_JUDGE_PROMPT = """You are an expert essay evaluator.
Please evaluate the following two essays according to the "Holistic Rating for Source-Based Writing" rubric.

You should first read both essays and then give a judgement on which essay is better. 
The judgement should be based on the quality of the writing, the clarity of the argument, 
and the use of evidence. You should also provide a reason for your judgement. 
The judgement should be returned as a valid JSON object as shown below:

```json
{{"score": "Essay 1", "reason": "Essay 1 is better because it has a clearer argument and uses more evidence to support its points."}}
```

```json
{{"score": "Essay 2", "reason": "Essay 2 is better because it has a more coherent structure and uses more varied vocabulary."}}
```
Essay 1:
{full_text_1}
Essay 2:
{full_text_2}
"""
