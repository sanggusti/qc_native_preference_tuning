"""Build the merged English medical pool (medqa + medmcqa) for medpool.

Samples seeded subsets per source, normalizes to one choice-QA schema, and
pushes the English pool that translate_benchmark reads. Pure cores are
stub-tested with no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sample_indices(n_items: int, n_pick: int, seed: int) -> list[int]:
    """Seeded subset indices, deterministic for one (n, k, seed)."""
    import random

    if n_pick > n_items:
        raise ValueError(f"pick {n_pick} exceeds source size {n_items}")
    return sorted(random.Random(seed).sample(range(n_items), n_pick))


def normalize_medqa(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    """bigbio med_qa row (4 options) to the pool schema."""
    choices = [str(c) for c in row["choices"]]
    if len(choices) != 4:
        raise ValueError(f"medqa row {index} has {len(choices)} choices, want 4")
    answer = [str(a) for a in row["answer"]]
    if len(answer) != 1 or answer[0] not in choices:
        raise ValueError(f"medqa row {index} answer not among choices")
    return {
        "id": f"medpool-{split}-medqa-{index:05d}",
        "question": str(row["question"]),
        "choices": choices,
        "answer": answer,
        "source": "medqa",
        "source_id": str(row.get("id", index)),
    }


def normalize_medmcqa(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    """MedMCQA row (opa-d, 0-based cop) to the pool schema."""
    choices = [str(row[f"op{k}"]) for k in ("a", "b", "c", "d")]
    cop = int(row["cop"])
    if cop not in (0, 1, 2, 3):
        raise ValueError(f"medmcqa row {index} cop {cop} outside 0-3")
    return {
        "id": f"medpool-{split}-medmcqa-{index:05d}",
        "question": str(row["question"]),
        "choices": choices,
        "answer": [choices[cop]],
        "source": "medmcqa",
        "source_id": str(row.get("id", index)),
    }


POOL_CARD = """---
license: other
language:
- en
task_categories:
- multiple-choice
tags:
- medical
- medqa
- medmcqa
- qc-native-preference-tuning
---

# Medical pool (English source)

Merged English source for the medpool benchmark: seeded subsets of
bigbio/med_qa (USMLE-style 4-option) and openlifescienceai/medmcqa.
Translated per language by pipeline/datagenerator/translate_benchmark.py.

- sampling: {sizes}; seed {seed}
- medmcqa test-pool rows come from the medmcqa train split (its test labels
  are not public); the train/test-pool subsets are disjoint.
- columns: id, question, choices, answer, source, source_id

License: research use. medmcqa is Apache-2.0; bigbio/med_qa carries no
machine-readable license (USMLE content, see the source repository).
Redistribution of the medqa portion stays research-only.
"""


def main() -> None:
    """Hydra entry point (deferred imports keep hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(version_base=None, config_path="../../configs/datagenerator",
               config_name="medpool_build")
    def hydra_main_fn(cfg: DictConfig) -> None:
        import tempfile

        from datasets import Dataset, DatasetDict, load_dataset
        from dotenv import load_dotenv
        from huggingface_hub import HfApi

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        seed = int(cfg.seed)
        medqa_train = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa",
                                   split="train")
        medqa_test = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa",
                                  split="test")
        medmcqa_train = load_dataset("openlifescienceai/medmcqa", split="train")
        pool: dict[str, list[dict[str, Any]]] = {}
        medmcqa_all = sample_indices(
            len(medmcqa_train),
            int(cfg.train.medmcqa_rows) + int(cfg.test.medmcqa_rows), seed)
        for split in ("train", "test"):
            n_medqa = int(cfg[split].medqa_rows)
            n_medmcqa = int(cfg[split].medmcqa_rows)
            medqa_rows = medqa_train if split == "train" else medqa_test
            rows = [normalize_medqa(dict(medqa_rows[i]), i, split)
                    for i in sample_indices(len(medqa_rows), n_medqa, seed)]
            offset = 0 if split == "train" else int(cfg.train.medmcqa_rows)
            take = medmcqa_all[offset:offset + n_medmcqa]
            rows += [normalize_medmcqa(dict(medmcqa_train[i]), i, split) for i in take]
            pool[split] = rows
            print(f"pool {split}: {len(rows)} rows "
                  f"({n_medqa} medqa + {n_medmcqa} medmcqa)")
        DatasetDict({s: Dataset.from_list(pool[s]) for s in pool}).push_to_hub(
            cfg.push_to_hub, private=bool(cfg.private))
        print(f"Pushed to https://huggingface.co/datasets/{cfg.push_to_hub}")
        sizes = (f"train {cfg.train.medqa_rows}+{cfg.train.medmcqa_rows}, "
                 f"test {cfg.test.medqa_rows}+{cfg.test.medmcqa_rows}")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
            handle.write(POOL_CARD.format(sizes=sizes, seed=seed))
            card_path = handle.name
        import os

        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=card_path, path_in_repo="README.md",
            repo_id=cfg.push_to_hub, repo_type="dataset")
        Path(card_path).unlink(missing_ok=True)
        print("Card uploaded")

    hydra_main_fn()


if __name__ == "__main__":
    main()
