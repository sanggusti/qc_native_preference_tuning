"""No-network tests for the medpool builder cores."""

import pytest

from pipeline.datagenerator.build_medpool import (
    normalize_medmcqa,
    normalize_medqa,
    sample_indices,
)


def test_sample_indices_deterministic_and_disjoint():
    first = sample_indices(1000, 100, 47)
    assert first == sample_indices(1000, 100, 47) == sorted(first)
    with pytest.raises(ValueError, match="exceeds source size"):
        sample_indices(10, 11, 47)


def test_normalize_medqa_validates_schema():
    row = {"id": "m1", "question": "Q?", "choices": ["a", "b", "c", "d"],
           "answer": ["c"]}
    out = normalize_medqa(row, 0, "train")
    assert out["id"] == "medpool-train-medqa-00000" and out["source"] == "medqa"
    assert out["answer"] == ["c"]
    with pytest.raises(ValueError, match="4"):
        normalize_medqa({**row, "choices": ["a", "b"]}, 0, "train")
    with pytest.raises(ValueError, match="not among choices"):
        normalize_medqa({**row, "answer": ["z"]}, 0, "train")


def test_normalize_medmcqa_maps_cop():
    row = {"id": "c1", "question": "Q?", "opa": "a", "opb": "b", "opc": "c",
           "opd": "d", "cop": 2}
    out = normalize_medmcqa(row, 3, "test")
    assert out["id"] == "medpool-test-medmcqa-00003" and out["answer"] == ["c"]
    out = normalize_medmcqa({**row, "cop": 0}, 3, "test")
    assert out["answer"] == ["a"]
    with pytest.raises(ValueError, match="outside 0-3"):
        normalize_medmcqa({**row, "cop": 4}, 3, "test")
