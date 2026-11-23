"""Shared output helpers for the covariate scripts."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_json(payload: dict[str, Any], output_dir: str | Path, name: str) -> Path:
    """Write payload plus provenance to output_dir/name and return the path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"date": date.today().isoformat(), **payload}
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def maybe_log_wandb(project: str, run_name: str, config: Any, payload: dict[str, Any]) -> None:
    """Log payload via the shared tracking module (graceful when offline)."""
    from src.utils.tracking import finish, init_run, log_metrics

    handle = init_run(project, run_name, config=config)
    log_metrics(handle, payload)
    log_call_safe(handle, run_name, {"config": config}, payload)
    finish(handle)


def log_call_safe(handle: dict[str, Any], name: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Best-effort Weave lineage call (import-local to keep collection light)."""
    from src.utils.tracking import log_call

    log_call(handle, name, inputs, outputs)
