"""Modal spend ledger with cap enforcement (operator-set budget).

Every programmatic Modal submission checks the cap first and records measured
wall time after completion. Manual deploys (`modal deploy`) go through the CLI:
`python -m pipeline.modal_runner.budget status|check|record`. The ledger is a
local JSON file under data/ (gitignored); paste totals into experiments.md
section 6 after each phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RATES_PATH = PROJECT_ROOT / "configs" / "modal_rates.yaml"


def load_rates(path: str | Path = DEFAULT_RATES_PATH) -> dict[str, Any]:
    """Load cap, ledger path and GPU rates from the rates config."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ledger_path(rates: dict[str, Any]) -> Path:
    path = Path(rates["spend_ledger"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_spend(rates: dict[str, Any]) -> dict[str, Any]:
    """Read the spend ledger (empty ledger when the file is missing)."""
    path = _ledger_path(rates)
    if not path.exists():
        return {"entries": []}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def total_usd(ledger: dict[str, Any]) -> float:
    """Sum recorded spend in USD."""
    return float(sum(entry.get("usd", 0.0) for entry in ledger.get("entries", [])))


def estimate_usd(rates: dict[str, Any], gpu: str, seconds: float) -> float:
    """Estimate one job's cost from GPU type and wall seconds."""
    hourly = rates["rates_usd_per_hour"].get(gpu)
    if hourly is None:
        raise ValueError(f"unknown GPU {gpu!r}; add a rate to configs/modal_rates.yaml")
    return float(hourly) * seconds / 3600


def check_budget(rates: dict[str, Any], extra_usd: float = 0.0) -> dict[str, Any]:
    """Refuse when recorded spend plus extra_usd already covers the cap."""
    spent = total_usd(read_spend(rates))
    cap = float(rates["monthly_cap_usd"])
    status = {"spent_usd": spent, "cap_usd": cap, "remaining_usd": cap - spent - extra_usd}
    if spent + extra_usd >= cap:
        raise ValueError(f"Modal budget exhausted: {spent:.2f} + {extra_usd:.2f} >= cap {cap:.2f}")
    return status


def record_spend(
    rates: dict[str, Any], gpu: str, seconds: float, job: str, model: str = ""
) -> dict[str, Any]:
    """Append one measured job to the ledger; returns the entry plus totals."""
    usd = estimate_usd(rates, gpu, seconds)
    ledger = read_spend(rates)
    entry = {
        "date": date.today().isoformat(),
        "job": job,
        "model": model,
        "gpu": gpu,
        "seconds": seconds,
        "usd": round(usd, 4),
    }
    ledger["entries"].append(entry)
    path = _ledger_path(rates)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
    return {"entry": entry, "total_usd": total_usd(ledger), "cap_usd": float(rates["monthly_cap_usd"])}


def main(argv: list[str] | None = None) -> None:
    """CLI: status | check --gpu GPU --hours H | record --gpu GPU --seconds S --job J."""
    parser = argparse.ArgumentParser(prog="budget", description="Modal spend ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print recorded spend against the cap")
    check = sub.add_parser("check", help="refuse when a planned job would breach the cap")
    check.add_argument("--gpu", required=True)
    check.add_argument("--hours", type=float, required=True)
    record = sub.add_parser("record", help="record a finished manual job")
    record.add_argument("--gpu", required=True)
    record.add_argument("--seconds", type=float, required=True)
    record.add_argument("--job", required=True)
    record.add_argument("--model", default="")
    args = parser.parse_args(argv)
    rates = load_rates()
    if args.command == "status":
        ledger = read_spend(rates)
        print(f"spent ${total_usd(ledger):.2f} of ${float(rates['monthly_cap_usd']):.2f} "
              f"across {len(ledger['entries'])} jobs")
    elif args.command == "check":
        status = check_budget(rates, estimate_usd(rates, args.gpu, args.hours * 3600))
        print(f"ok: ${status['remaining_usd']:.2f} remaining after the planned job")
    elif args.command == "record":
        result = record_spend(rates, args.gpu, args.seconds, args.job, args.model)
        print(f"recorded ${result['entry']['usd']:.4f}; total ${result['total_usd']:.2f}")


if __name__ == "__main__":
    main()
