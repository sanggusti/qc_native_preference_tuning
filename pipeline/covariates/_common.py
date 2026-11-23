"""Shared output helpers for the covariate scripts."""

from __future__ import annotations

import csv
import json
import sys
import tarfile
import tempfile
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FLORES_TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def read_parallel_csv(source: str | Path) -> list[dict[str, str]]:
    """Read a parallel-sentences CSV from a local path or http(s) URL."""
    text = str(source)
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text) as response:  # noqa: S310 - allowlisted hosts only at call sites
            content = response.read().decode("utf-8")
        lines = content.splitlines()
    else:
        with open(text, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    return [dict(row) for row in csv.DictReader(lines)]


def _tarball_member_name(members: list[str], split: str, flores_code: str) -> str:
    suffix = f"{split}/{flores_code}.{split}"
    matches = [name for name in members if name.endswith(suffix)]
    if not matches:
        raise ValueError(f"tarball has no {suffix}")
    return matches[0]


def flores_from_tarball(tarball: str | Path, split: str, flores_code: str) -> list[str]:
    """Read aligned sentences for one code from a FLORES-200 tarball file."""
    with tarfile.open(tarball, "r:gz") as archive:
        member = _tarball_member_name(archive.getnames(), split, flores_code)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"cannot extract {member}")
        return [line.decode("utf-8").rstrip("\n") for line in handle if line.strip()]


def flores_sentences(
    dataset: str,
    flores_code: str,
    split: str,
    tarball_url: str = FLORES_TARBALL_URL,
    cache_dir: str | Path | None = None,
) -> list[str]:
    """FLORES sentences via the Hub, falling back to the public tarball.

    The facebook/flores Hub repo is gated; when Hub access fails for any
    reason, download (once, cached) and read the original release tarball.
    """
    try:
        from datasets import load_dataset

        rows = load_dataset(dataset, flores_code, split=split)
        return [str(row["sentence"]) for row in rows]
    except Exception as exc:  # noqa: BLE001 - any Hub failure triggers the fallback
        print(f"Hub FLORES load failed ({exc}); falling back to the release tarball")
    directory = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / "flores200_dataset.tar.gz"
    if not cached.exists():
        urllib.request.urlretrieve(tarball_url, cached)  # noqa: S310 - fixed release URL
    return flores_from_tarball(cached, split, flores_code)


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
