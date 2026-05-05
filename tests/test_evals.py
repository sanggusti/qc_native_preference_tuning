"""Tests for src/evals."""

from src.evals import EvalSample, EvalResult, compute_win_rate


def test_compute_win_rate_all_wins():
    results = [
        EvalResult(prompt="p", response_a="a", response_b="b", winner="a")
        for _ in range(4)
    ]
    assert compute_win_rate(results, model_is_a=True) == 1.0


def test_compute_win_rate_all_losses():
    results = [
        EvalResult(prompt="p", response_a="a", response_b="b", winner="b")
        for _ in range(4)
    ]
    assert compute_win_rate(results, model_is_a=True) == 0.0


def test_compute_win_rate_with_ties():
    results = [
        EvalResult(prompt="p", response_a="a", response_b="b", winner="a"),
        EvalResult(prompt="p", response_a="a", response_b="b", winner="tie"),
        EvalResult(prompt="p", response_a="a", response_b="b", winner="b"),
        EvalResult(prompt="p", response_a="a", response_b="b", winner="tie"),
    ]
    # wins=1, ties=2, total=4  ->  (1 + 0.5*2) / 4 = 0.5
    assert compute_win_rate(results, model_is_a=True) == 0.5


def test_compute_win_rate_empty():
    assert compute_win_rate([]) == 0.0


def test_llm_judge_evaluator(mock_openai_client):
    from src.evals.tasks import LLMJudgeEvaluator

    evaluator = LLMJudgeEvaluator(
        config={"judge_model": "gpt-4o-mini", "judge_temperature": 0.0},
        client=mock_openai_client,
    )
    sample = EvalSample(
        prompt="What is 2+2?",
        response_a="4",
        response_b="Five",
    )
    result = evaluator.evaluate_pair(sample)
    assert result.winner in ("a", "b", "tie")
    assert result.prompt == sample.prompt

    # evaluate_batch returns (results, failures)
    results, failures = evaluator.evaluate_batch([sample, sample])
    assert len(results) == 2
    assert failures == 0
