"""Tests for FLORES calibration helpers (no network, stub client)."""

from pipeline.datagenerator.calibrate_flores import (
    build_upload_rows,
    calibration_key,
    render_blueprint,
)
from pipeline.datagenerator.translate_benchmark import probe_language_expansion


class _Datasets:
    def __init__(self, reject=()):
        self.reject = set(reject)
        self.calls = []

    def run(self, dataset_id, **kwargs):
        self.calls.append(kwargs)
        expansion = kwargs.get("language_expansion") or {}
        for code in expansion.get("languages", []):
            if code in self.reject:
                raise RuntimeError(f"400 unknown language {code}")
        return {"quote": "1 credit"}


class _Client:
    def __init__(self, reject=()):
        self.datasets = _Datasets(reject)


def test_render_blueprint_substitutes_language_and_register():
    text = render_blueprint("Translate into {language}. {register_instruction}", "Javanese", "Use ngoko.")
    assert "Javanese" in text and "ngoko" in text
    assert "{" not in text


def test_calibration_key_scopes_smoke_and_full():
    assert calibration_key("jv", 50) == "flores-jv-rows50"
    assert calibration_key("jv", None) == "flores-jv-rowsfull"


def test_build_upload_rows_echo_completion():
    rows = build_upload_rows(["hello", "world"])
    assert rows == [
        {"id": "eng_Latn:devtest:0000", "question": "hello", "answer": "hello"},
        {"id": "eng_Latn:devtest:0001", "question": "world", "answer": "world"},
    ]


def test_probe_records_accepted_and_rejected_per_code():
    client = _Client()
    results = probe_language_expansion(client, "ds", {"prompt": "q"}, ["en", "id", "min"])
    assert results["en"]["accepted"] and results["id"]["accepted"]
    assert results["min"] == {"accepted": False, "detail": "no ISO 639-1 code"}
    assert all(call.get("estimate") is True for call in client.datasets.calls)


def test_probe_records_api_rejection():
    client = _Client(reject=("xx",))
    client.datasets.run = lambda dataset_id, **kwargs: (_ for _ in ()).throw(RuntimeError("400 nope"))
    results = probe_language_expansion(client, "ds", {}, ["en"])
    assert results["en"]["accepted"] is False
    assert "400" in results["en"]["detail"]
