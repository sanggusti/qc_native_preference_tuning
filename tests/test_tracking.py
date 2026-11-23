"""Hermetic tests for tracking: everything is a no-op when disabled."""

from src.utils.tracking import (
    finish,
    init_run,
    log_artifact,
    log_call,
    log_figure,
    log_metrics,
    log_table,
    summarize,
)


def test_disabled_handle_never_raises(tmp_path):
    handle = {"enabled": False, "run": None, "weave": None, "name": "x"}
    log_metrics(handle, {"a": 1})
    log_table(handle, "t", ["a"], [[1]])
    log_figure(handle, "f", tmp_path / "missing.png")
    log_artifact(handle, "a", tmp_path)
    log_call(handle, "op", {"in": 1}, {"out": 2})
    finish(handle)


def test_init_run_degrades_gracefully_without_server(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DISABLE_CODE", "true")
    handle = init_run("proj", "run", tags=["t"], config={"a": 1}, tracing_enabled=False)
    assert handle["name"] == "run"
    log_metrics(handle, {"a": 1})
    finish(handle)


def test_summarize_truncates_nested_payloads():
    long = "x" * 1000
    assert summarize({"a": long, "b": list(range(50))})["a"].endswith("...")
    assert summarize({"b": list(range(50))})["b"]["len"] == 50
    assert summarize([1, 2])["type"] == "list"
    assert summarize(3) == "3"
    assert summarize(None) == "None"
