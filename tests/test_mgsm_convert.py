"""No-network tests for the MGSM conversion pipeline against the SDK 0.10.0 download shape."""

import json

import pytest
from datasets import Dataset

from pipeline.datagenerator.evals_translate import mgsm_convert as mc

SOURCE = Dataset.from_list(
    [
        {
            "question": "Q1?",
            "answer": "A1 #### 3",
            "answer_number": 3,
            "equation_solution": "1+2=3",
        }
    ]
)


class _Body:
    def __init__(self, rows):
        self.rows = rows

    def write_to_file(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row) + "\n")


class _Datasets:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def download(self, dataset_id, file_format="jsonl"):
        self.calls.append((dataset_id, file_format))
        return _Body(self.rows)


class _Client:
    def __init__(self, rows):
        self.datasets = _Datasets(rows)


def test_download_dataset_reads_binary_body():
    client = _Client([{"enhanced_prompt": "P", "enhanced_completion": "C"}])
    dataset = mc.download_dataset(client, "ds_1")
    assert client.datasets.calls == [("ds_1", "jsonl")]
    assert dataset[0]["enhanced_prompt"] == "P"


def test_preference_dataset_maps_enhanced_columns():
    translated = Dataset.from_list(
        [{"enhanced_prompt": "Pertanyaan?", "enhanced_completion": "Jawab #### 3"}]
    )
    out = mc.to_preference_dataset(SOURCE, translated, "en", "Indonesian")
    row = out[0]
    assert row["prompt"] == "Pertanyaan?"
    assert row["chosen"] == "Jawab #### 3"
    assert row["rejected"] == "A1 #### 3"
    assert row["question_en"] == "Q1?"


def test_preference_dataset_accepts_source_column_names():
    translated = Dataset.from_list([{"question": "Pertanyaan?", "answer": "Jawab #### 3"}])
    out = mc.to_preference_dataset(SOURCE, translated, "en", "Indonesian")
    assert out[0]["prompt"] == "Pertanyaan?"


def test_preference_dataset_refuses_silent_english_fallback():
    translated = Dataset.from_list([{"something_else": "x"}])
    with pytest.raises(KeyError):
        mc.to_preference_dataset(SOURCE, translated, "en", "Indonesian")
