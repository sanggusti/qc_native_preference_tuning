"""No-network tests for derived splits and the pooled set with fixtures."""

import json
import random
import re

import pytest

from pipeline.datagenerator.derive_split import (
    build_nllb_row,
    build_rt_row,
    find_numbers,
    format_answer,
    last_expression,
    load_jsonl,
    perturb_mapping,
    propagate_pro1,
    reinstantiate,
    repo_id,
    substitute_numbers,
)
from pipeline.datagenerator.pool_benchmark import composition_record, select_pooled

EN_Q = "Budi has 3 apples and buys 4 more. How many in total?"
EN_A = "He starts with 3, then adds 4: 3 + 4 = 7. #### 7"
JV_Q = "Budi duwe 3 apel lan tuku 4 maneh. Piro kabehe?"
JV_A = "Dheweke duwe 3, banjur tambah 4: 3 + 4 = 7. #### 7"


def _strip_numbers(text):
    return re.sub(r"\d+(?:[.,]\d+)*", "#", text)


def test_repo_id_uses_template_plus_suffix():
    assert repo_id("sanggusti/{benchmark}-{language}", "gsm8k", "jv", "-rt") == "sanggusti/gsm8k-jv-rt"
    assert repo_id("sanggusti/{benchmark}-{language}", "gsm8k", "all") == "sanggusti/gsm8k-all"


def test_build_rt_row_keeps_id_and_original_target():
    row = build_rt_row("s-1", "Budi has apples.", "7", "INSTR_EN", "q_en", "a_en")
    assert row["id"] == "s-1" and row["target"] == "7"
    assert row["language"] == "en" and row["instruction"] == "INSTR_EN"
    assert row["variant"] == "rt"


def test_build_nllb_row_keeps_id_target_and_language():
    row = build_nllb_row("s-1", "Budi duwe apel.", "7", "INSTR", "INSTR_EN", "q", "a", "jv", "ngoko")
    assert row["id"] == "s-1" and row["target"] == "7"
    assert row["language"] == "jv" and row["instruction"] == "INSTR"
    assert row["variant"] == "nllb"


def test_find_numbers_and_substitution_edges():
    assert find_numbers("Room 15, bed 1, price 2.5") == ["15", "1", "2.5"]
    assert substitute_numbers("Room 15, bed 1", {"15": "27", "1": "9"}) == "Room 27, bed 9"


def test_perturb_mapping_preserves_shape_and_seed():
    first = perturb_mapping(["3", "15", "2.5"], random.Random(47))
    assert first == perturb_mapping(["3", "15", "2.5"], random.Random(47))
    assert len(first["15"]) == 2 and first["15"][0] != "0"
    assert len(first["2.5"].split(".")[1]) == 1


def test_last_expression_and_format_answer():
    assert last_expression(EN_A).strip() == "3 + 4"
    assert last_expression("The answer is seven. #### 7") is None
    assert format_answer(7.0) == "7"
    assert format_answer(2.5) == "2.5"


def test_reinstantiate_recomputes_target_consistently():
    new_q, new_a, manifest = reinstantiate(EN_Q, EN_A, random.Random(47))
    assert manifest["solved"]
    a, b = int(manifest["mapping"]["3"]), int(manifest["mapping"]["4"])
    assert new_a.endswith(f"#### {a + b}")
    assert _strip_numbers(new_q) == _strip_numbers(EN_Q)
    assert _strip_numbers(new_a) == _strip_numbers(EN_A.replace("#### 7", f"#### {a + b}"))


def test_reinstantiate_falls_back_unperturbed_when_unsolvable():
    new_q, new_a, manifest = reinstantiate("How are you?", "Fine. #### 1", random.Random(47))
    assert (new_q, new_a) == ("How are you?", "Fine. #### 1")
    assert manifest["solved"] is False


def test_propagate_pro1_applies_same_mapping_across_languages():
    rows = {"en": {"question": EN_Q, "answer": EN_A}, "jv": {"question": JV_Q, "answer": JV_A}}
    out, manifest = propagate_pro1(rows, seed=47)
    assert manifest["solved"]
    assert out["en"][0]["answer"].split("####")[1].strip() == out["jv"][0]["answer"].split("####")[1].strip()
    assert _strip_numbers(out["jv"][0]["question"]) == _strip_numbers(JV_Q)
    for literal, new in manifest["mapping"].items():
        assert literal not in out["en"][0]["question"] or literal == new


def test_load_jsonl_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}]


def test_select_pooled_exact_total_and_rotation():
    pool = {lang: [{"id": f"{lang}-{i}"} for i in range(10)] for lang in ("id", "jv", "su")}
    selected, composition = select_pooled(pool, 10, seed=47)
    assert len(selected) == 10
    assert composition == {"id": 4, "jv": 3, "su": 3}
    assert all("pool_language" in row for row in selected)
    again, _ = select_pooled(pool, 10, seed=47)
    assert [r["id"] for r in selected] == [r["id"] for r in again]


def test_select_pooled_raises_on_short_language():
    pool = {"id": [{"id": "x"}], "jv": [{"id": f"j-{i}"} for i in range(10)]}
    with pytest.raises(ValueError, match="needs"):
        select_pooled(pool, 10, seed=47)


def test_composition_record():
    record = composition_record({"id": 4, "jv": 3}, seed=47)
    assert record == {"seed": 47, "total": 7, "per_language": {"id": 4, "jv": 3}}
