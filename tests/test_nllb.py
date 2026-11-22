"""No-network tests for the NLLB job: batching, codes, schema, orchestration."""

import pytest

from pipeline.modal_runner.nllb_translate import (
    build_output_row,
    chunk_rows,
    orchestrate,
    resolve_codes,
)


def _rows(n):
    return [{"source_id": f"id-{i:03d}", "text": f"text {i}"} for i in range(n)]


def _stub_translate(batches, source_code, target_code):
    assert source_code and target_code
    return [f"[{target_code}]{text}" for batch in batches for text in batch]


def test_chunk_rows_batches_and_validates():
    assert chunk_rows(_rows(5), 2) == [_rows(5)[0:2], _rows(5)[2:4], _rows(5)[4:5]]
    assert chunk_rows([], 8) == []
    with pytest.raises(ValueError, match="batch_size"):
        chunk_rows(_rows(3), 0)


def test_resolve_codes_maps_series_codes_to_flores():
    assert resolve_codes("en", "jv") == ("eng_Latn", "jav_Latn")
    assert resolve_codes("min", "en") == ("min_Latn", "eng_Latn")
    assert resolve_codes("bug", "id") == ("bug_Latn", "ind_Latn")


def test_resolve_codes_rejects_unknown_language():
    with pytest.raises(FileNotFoundError):
        resolve_codes("en", "xx")


def test_build_output_row_schema():
    row = build_output_row("s-1", "hello", "halo", "eng_Latn", "ind_Latn", "second_opinion")
    assert row["source_id"] == "s-1"
    assert row["mode"] == "second_opinion"
    assert row["direction"] == "eng_Latn->ind_Latn"
    assert row["source_text"] == "hello"
    assert row["translated_text"] == "halo"
    assert row["model_id"] == "facebook/nllb-200-3.3B"
    assert row["license"] == "CC-BY-NC"
    assert row["translated_at"]


def test_orchestrate_translates_batches_with_stub():
    outputs = orchestrate(_rows(5), "eng_Latn", "jav_Latn", "second_opinion", 2, _stub_translate)
    assert len(outputs) == 5
    assert [o["source_id"] for o in outputs] == [f"id-{i:03d}" for i in range(5)]
    assert all(o["translated_text"].startswith("[jav_Latn]") for o in outputs)
    assert all(o["direction"] == "eng_Latn->jav_Latn" for o in outputs)


def test_orchestrate_back_translation_direction():
    outputs = orchestrate(_rows(2), "jav_Latn", "eng_Latn", "back_translation", 8, _stub_translate)
    assert all(o["direction"] == "jav_Latn->eng_Latn" for o in outputs)
    assert all(o["mode"] == "back_translation" for o in outputs)


def test_orchestrate_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        orchestrate(_rows(1), "eng_Latn", "jav_Latn", "nope", 8, _stub_translate)
