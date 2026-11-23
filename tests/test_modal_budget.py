"""Tests for the Modal budget ledger and the judge endpoint option."""

import json

import pytest

from pipeline.datagenerator.gate import openai_judge_fn
from pipeline.modal_runner.budget import (
    check_budget,
    estimate_usd,
    load_rates,
    read_spend,
    record_spend,
    total_usd,
)


def _rates(tmp_path, cap=200.0):
    return {
        "monthly_cap_usd": cap,
        "spend_ledger": str(tmp_path / "spend.json"),
        "rates_usd_per_hour": {"A10G": 0.60, "A100": 2.00},
    }


def test_estimate_usd_and_unknown_gpu():
    rates = {"rates_usd_per_hour": {"A10G": 0.60}}
    assert estimate_usd(rates, "A10G", 3600) == 0.60
    assert estimate_usd(rates, "A10G", 1800) == 0.30
    with pytest.raises(ValueError, match="unknown GPU"):
        estimate_usd(rates, "H200", 60)


def test_record_and_total_round_trip(tmp_path):
    rates = _rates(tmp_path)
    assert total_usd(read_spend(rates)) == 0.0
    first = record_spend(rates, "A10G", 3600, "nllb-test", "m")
    assert first["entry"]["usd"] == 0.60
    record_spend(rates, "A100", 1800, "sft-test", "m")
    assert total_usd(read_spend(rates)) == 1.60
    assert json.loads((tmp_path / "spend.json").read_text())["entries"][0]["job"] == "nllb-test"


def test_check_budget_refuses_past_cap(tmp_path):
    rates = _rates(tmp_path, cap=1.0)
    record_spend(rates, "A10G", 3600, "job-1")
    status = check_budget(rates)
    assert status["remaining_usd"] == 0.40
    with pytest.raises(ValueError, match="budget exhausted"):
        check_budget(rates, extra_usd=0.50)


def test_load_rates_reads_configured_cap():
    rates = load_rates()
    assert rates["monthly_cap_usd"] == 200
    assert rates["rates_usd_per_hour"]["A10G"] > 0


def test_judge_factory_accepts_modal_base_url():
    direct = openai_judge_fn("gpt-4o-mini")
    modal = openai_judge_fn("google/gemma-4-12B-it", base_url="https://modal.example/v1")
    assert callable(direct) and callable(modal)
    with pytest.raises(ValueError, match="judge_base_url"):
        openai_judge_fn("google/gemma-4-12B-it")
