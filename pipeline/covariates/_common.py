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
    """Log payload to wandb; warn and continue when wandb is not installed."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping run logging (JSON output still written)")
        return
    run = wandb.init(project=project, name=run_name, config=dict(config))
    run.log(payload)
    run.finish()
