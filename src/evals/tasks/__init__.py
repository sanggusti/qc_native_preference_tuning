"""Evaluation task implementations."""

import logging
import re
from typing import List, Optional

from src.evals import BaseEvaluator, EvalResult, EvalSample

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating the quality of two AI assistant responses.
Given a user prompt and two responses (A and B), decide which response is better.

Respond ONLY with one of: "A", "B", or "TIE".
Then on a new line provide a brief reasoning (1-2 sentences).
"""

_JUDGE_USER_TEMPLATE = """\
### User Prompt
{prompt}

### Response A
{response_a}

### Response B
{response_b}

### Which response is better? Answer with "A", "B", or "TIE" on the first line."""


class LLMJudgeEvaluator(BaseEvaluator):
    """Use an LLM as a judge to evaluate preference pairs."""

    def __init__(
        self,
        config: dict,
        client=None,
    ):
        super().__init__(config)
        self.judge_model: str = config.get("judge_model", "gpt-4o-mini")
        self.temperature: float = config.get("judge_temperature", 0.0)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def evaluate_pair(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single preference pair using an LLM judge.

        Args:
            sample: EvalSample containing prompt and two responses.

        Returns:
            EvalResult with winner and reasoning.
        """
        user_content = _JUDGE_USER_TEMPLATE.format(
            prompt=sample.prompt,
            response_a=sample.response_a,
            response_b=sample.response_b,
        )
        response = self.client.chat.completions.create(
            model=self.judge_model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content.strip()
        winner, reasoning = _parse_judge_output(raw)
        return EvalResult(
            prompt=sample.prompt,
            response_a=sample.response_a,
            response_b=sample.response_b,
            winner=winner,
            reasoning=reasoning,
            language=sample.language,
            metadata=sample.metadata,
        )


def _parse_judge_output(raw: str) -> tuple:
    """Parse raw judge output into (winner, reasoning).

    Args:
        raw: Raw string output from the judge model.

    Returns:
        Tuple of (winner_str, reasoning_str) where winner is 'a', 'b', or 'tie'.
    """
    lines = raw.strip().splitlines()
    first_line = lines[0].strip().upper() if lines else ""
    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    if re.search(r"\bA\b", first_line):
        winner = "a"
    elif re.search(r"\bB\b", first_line):
        winner = "b"
    else:
        winner = "tie"
    return winner, reasoning


__all__ = ["LLMJudgeEvaluator"]
