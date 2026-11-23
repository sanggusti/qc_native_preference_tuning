"""Unified experiment tracking: wandb runs plus Weave lineage calls.

Every stage logs through here. wandb carries runs (config, metrics), tables,
figures and artifacts; Weave carries one call node per stage run so the full
sweep is traceable from datasets through finetunes to eval scores. All entry
points degrade gracefully: without credentials, network, or the packages,
calls warn once and continue, so tracking can never break a pipeline run.
Set WANDB_MODE=offline to queue locally and `wandb sync` later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MAX_TABLE_ROWS = 10000
MAX_SUMMARY_ITEMS = 20
MAX_SUMMARY_CHARS = 500


_SEEN_WARNINGS: set[str] = set()


def _warned(message: str) -> None:
    if message not in _SEEN_WARNINGS:
        _SEEN_WARNINGS.add(message)
        print(f"tracking: {message}")


def summarize(value: Any, depth: int = 0) -> Any:
    """Summarize nested values for Weave call IO (shapes, not full payloads)."""
    if isinstance(value, dict):
        items = list(value.items())[:MAX_SUMMARY_ITEMS]
        return {str(key): summarize(item, depth + 1) for key, item in items}
    if isinstance(value, (list, tuple)):
        items = list(value)[:MAX_SUMMARY_ITEMS]
        return {"type": type(value).__name__, "len": len(value),
                "sample": [summarize(item, depth + 1) for item in items]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value)
        return text if len(text) <= MAX_SUMMARY_CHARS else text[:MAX_SUMMARY_CHARS] + "..."
    return f"<{type(value).__name__}>"


def init_run(
    project: str,
    name: str,
    tags: list[str] | None = None,
    config: Any = None,
    tracing_enabled: bool = True,
) -> dict[str, Any]:
    """Start a wandb run plus Weave tracing; never raises, may be disabled."""
    handle: dict[str, Any] = {"enabled": False, "run": None, "weave": None, "name": name}
    try:
        import wandb

        cfg = dict(config) if config is not None else None
        handle["run"] = wandb.init(
            project=project, name=name, tags=list(tags or []), config=cfg, resume="allow"
        )
        handle["enabled"] = True
    except Exception as exc:  # noqa: BLE001 - tracking must never break runs
        _warned(f"wandb init failed ({exc}); continuing untracked")
        return handle
    if tracing_enabled:
        try:
            import weave

            handle["weave"] = weave.init(project)
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort
            _warned(f"weave init failed ({exc}); continuing without lineage calls")
    return handle


def log_metrics(handle: dict[str, Any], metrics: dict[str, Any], step: int | None = None) -> None:
    """Log scalar metrics to the run (no-op when disabled)."""
    if not handle.get("enabled"):
        return
    try:
        if step is None:
            handle["run"].log(metrics)
        else:
            handle["run"].log(metrics, step=step)
    except Exception as exc:  # noqa: BLE001
        _warned(f"wandb log failed ({exc})")


def log_table(
    handle: dict[str, Any], name: str, columns: list[str], rows: list[list[Any]]
) -> None:
    """Log a wandb Table, capped at MAX_TABLE_ROWS (no-op when disabled)."""
    if not handle.get("enabled"):
        return
    try:
        import wandb

        capped = rows[:MAX_TABLE_ROWS]
        handle["run"].log({name: wandb.Table(columns=columns, data=capped)})
        if len(rows) > MAX_TABLE_ROWS:
            _warned(f"table {name} capped at {MAX_TABLE_ROWS} of {len(rows)} rows")
    except Exception as exc:  # noqa: BLE001
        _warned(f"wandb table {name} failed ({exc})")


def log_figure(handle: dict[str, Any], name: str, path: str | Path) -> None:
    """Log a figure file as a wandb Image (no-op when disabled)."""
    if not handle.get("enabled"):
        return
    try:
        import wandb

        handle["run"].log({name: wandb.Image(str(path))})
    except Exception as exc:  # noqa: BLE001
        _warned(f"wandb figure {name} failed ({exc})")


def log_artifact(
    handle: dict[str, Any], name: str, path: str | Path, artifact_type: str = "dataset"
) -> None:
    """Log a file or directory as a versioned wandb Artifact (no-op when disabled)."""
    if not handle.get("enabled"):
        return
    try:
        import wandb

        artifact = wandb.Artifact(name, type=artifact_type)
        target = Path(path)
        if target.is_dir():
            artifact.add_dir(str(target))
        else:
            artifact.add_file(str(target))
        handle["run"].log_artifact(artifact)
    except Exception as exc:  # noqa: BLE001
        _warned(f"wandb artifact {name} failed ({exc})")


def log_call(handle: dict[str, Any], op_name: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Record one Weave lineage call with summarized IO (no-op when disabled)."""
    client = handle.get("weave")
    if not handle.get("enabled") or client is None:
        return
    try:
        import weave

        summarized = {key: summarize(value) for key, value in outputs.items()}

        @weave.op(name=op_name)
        def _call(**kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return summarized

        _call(**{key: summarize(value) for key, value in inputs.items()})
    except Exception as exc:  # noqa: BLE001
        _warned(f"weave call {op_name} failed ({exc})")


def finish(handle: dict[str, Any]) -> None:
    """Finish the run (no-op when disabled)."""
    if not handle.get("enabled"):
        return
    try:
        handle["run"].finish()
    except Exception as exc:  # noqa: BLE001
        _warned(f"wandb finish failed ({exc})")
    handle["enabled"] = False
