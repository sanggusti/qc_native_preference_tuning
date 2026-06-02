import pytest
from pipeline.evals.litai_tools import inference_with_litai


@pytest.mark.asyncio
async def test_inference_with_litai(monkeypatch):
    monkeypatch.setenv("LITAI_MODEL", "lightning-ai/gpt-oss-120b")

    question = "What is the capital of France?"
    response = await inference_with_litai(question)

    assert isinstance(response, str), "Response should be a string"
    assert "Paris" in response, "Response should contain 'Paris' as the capital of France"
