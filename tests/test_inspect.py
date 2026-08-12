"""Tests for Inspect AI integration."""

from inspect_ai import Task
from src.evals.tasks.multilingual_qa import multilingual_qa


def test_multilingual_qa_returns_task():
    """multilingual_qa() returns an Inspect Task with the sample dataset."""
    result = multilingual_qa()
    assert isinstance(result, Task)


def test_multilingual_qa_has_metadata():
    """Task metadata contains domain and language."""
    result = multilingual_qa(domain="science", language="jv")
    assert result.metadata["domain"] == "science"
    assert result.metadata["language"] == "jv"
