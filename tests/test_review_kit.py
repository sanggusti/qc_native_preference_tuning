"""No-network tests for the review kit: round-trip, agreement, bounds, cards."""

import csv

from pipeline.datagenerator.review_export import build_review_rows, write_csv, write_jsonl
from pipeline.datagenerator.review_import import (
    cohen_kappa,
    error_upper_bound,
    merge_annotations,
    parse_annotation,
    pearson_r,
)


def _rows_by_id():
    return {
        f"s-{i:03d}": {
            "question_en": f"q_en {i}",
            "answer_en": f"a_en {i}",
            "question": f"q {i}",
            "answer": f"a {i}",
        }
        for i in range(5)
    }


def _annotated(source_id, annotator, score, preserved, severities=()):
    return {
        "source_id": source_id,
        "annotator": annotator,
        "esa_score": score,
        "answer_preserved": preserved,
        "spans": [{"start": 0, "end": 3, "severity": s} for s in severities],
    }


def test_export_import_round_trip_jsonl(tmp_path):
    rows = build_review_rows(["s-000", "s-001"], _rows_by_id(), "a1", "gsm8k", "test", "jv", "ngoko")
    assert rows[0]["review_id"] == "a1-0000"
    assert rows[0]["question_en"] == "q_en 0" and rows[0]["answer_preserved"] is None
    path = write_jsonl(rows, tmp_path / "review.jsonl")
    assert path.exists()
    annotated = [_annotated("s-000", "a1", 90, True), _annotated("s-001", "a1", 60, True, ["minor"])]
    parsed = [parse_annotation(dict(r, source_id=r["source_id"])) for r in annotated]
    assert parsed[0]["esa_score"] == 90.0
    assert parsed[1]["major_or_critical"] == 0
    assert parse_annotation(_annotated("s-1", "a1", 40, False, ["major", "critical"]))["major_or_critical"] == 2


def test_export_csv_columns(tmp_path):
    rows = build_review_rows(["s-000"], _rows_by_id(), "a1", "gsm8k", "test", "jv", "ngoko")
    path = write_csv(rows, tmp_path / "review.csv")
    with open(path, encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    assert len(reader) == 1
    assert reader[0]["source_id"] == "s-000" and reader[0]["esa_score"] == ""


def test_cohen_kappa_perfect_and_known_case():
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    assert abs(cohen_kappa([True, True, False, False], [True, False, True, False]) - 0.0) < 1e-9


def test_pearson_r_perfect_and_known_case():
    assert pearson_r([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert pearson_r([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_error_upper_bound_matches_methodology_anchors():
    assert abs(error_upper_bound(0, 100) - 0.030) < 0.001
    assert abs(error_upper_bound(3, 100) - 0.076) < 0.001
    assert error_upper_bound(100, 100) == 1.0
    assert error_upper_bound(1, 100) < error_upper_bound(5, 100)


def test_merge_two_annotators_agreement_and_adjudication():
    first = [_annotated(f"s-{i:03d}", "a1", 90, True) for i in range(4)]
    second = [_annotated(f"s-{i:03d}", "a2", 80, True) for i in range(4)]
    second[0]["answer_preserved"] = False
    merged = merge_annotations([first, second])
    assert merged["provisional"] is False
    assert merged["agreement"]["n_shared"] == 4
    assert merged["agreement"]["mad_esa"] == 10.0
    assert merged["columns"]["s-000"]["needs_adjudication"] is True
    assert merged["columns"]["s-000"]["answer_preserved"] is False
    assert merged["columns"]["s-001"]["answer_preserved"] is True
    assert merged["columns"]["s-001"]["esa_score"] == 85.0


def test_merge_single_reviewer_flags_provisional():
    merged = merge_annotations([[_annotated("s-000", "a1", 90, True)]])
    assert merged["provisional"] is True
    assert merged["agreement"] is None
    assert merged["columns"]["s-000"]["human_verified"] is True


def _render(template_path, values):
    from pathlib import Path

    text = Path(template_path).read_text(encoding="utf-8")
    rendered = text.format(**values)
    assert "{" not in rendered.replace("{{", "").replace("}}", "")
    return rendered


def test_dataset_card_template_renders_all_required_fields():
    values = {
        "benchmark": "gsm8k", "language": "jv", "register": "ngoko",
        "train_rows": 1000, "test_rows": 250, "blueprint_hash": "abc",
        "translator_runs": "run-1", "translator_date": "2026-11-22",
        "source_license": "MIT", "gate_summary": "pass", "calibration": "chrF 55",
        "excluded_ids": "none", "canary": "CANARY-123", "annotators": "A1, A2",
    }
    rendered = _render("docs/cards/dataset_card_template.md", values)
    for section in ("Gate results", "Calibration", "Excluded items", "Canary", "Do not train", "Annotators", "CC-BY-NC"):
        assert section in rendered


def test_model_card_template_renders_all_required_fields():
    values = {
        "repo": "sanggusti/gsm8k-jv-x", "base_model": "base", "series": "s01",
        "train_data": "data", "benchmark": "gsm8k", "language": "jv",
        "condition": "native", "replicate": 1, "disclaimer": "research only",
    }
    rendered = _render("docs/cards/model_card_template.md", values)
    for section in ("Base model", "Training data", "Series constants", "Evaluation", "Disclaimer"):
        assert section in rendered
