"""Tests for src/utils."""

from src.utils import batch_iterable, flatten_dict, ensure_dir
import tempfile
from pathlib import Path


def test_batch_iterable_even():
    data = list(range(6))
    batches = list(batch_iterable(data, 2))
    assert batches == [[0, 1], [2, 3], [4, 5]]


def test_batch_iterable_uneven():
    data = list(range(5))
    batches = list(batch_iterable(data, 2))
    assert len(batches) == 3
    assert batches[-1] == [4]


def test_batch_iterable_empty():
    assert list(batch_iterable([], 4)) == []


def test_flatten_dict():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    flat = flatten_dict(nested)
    assert flat == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_flatten_dict_custom_sep():
    nested = {"x": {"y": 10}}
    flat = flatten_dict(nested, sep="/")
    assert flat == {"x/y": 10}


def test_ensure_dir_creates_path():
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = Path(tmp) / "a" / "b" / "c"
        result = ensure_dir(new_dir)
        assert result.is_dir()
