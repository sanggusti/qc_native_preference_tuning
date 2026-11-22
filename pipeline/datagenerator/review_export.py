"""Export the human review sample, one file per annotator (issue #58).

Sample selection comes from gate.select_review_sample; this module renders the
selected ids as JSONL (primary) and CSV (spreadsheet) with the English source
beside the translation plus empty annotation columns.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CSV_COLUMNS = [
    "review_id", "source_id", "annotator", "benchmark", "split", "language",
    "register", "question_en", "answer_en", "question", "answer",
    "esa_score", "answer_preserved", "spans_json",
]


def build_review_rows(
    sample_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
    annotator: str,
    benchmark: str,
    split: str,
    language: str,
    register: str,
) -> list[dict[str, Any]]:
    """Build review rows pairing English source with translation per id."""
    review = []
    for position, source_id in enumerate(sample_ids):
        row = rows_by_id[source_id]
        review.append({
            "review_id": f"{annotator}-{position:04d}",
            "source_id": source_id,
            "annotator": annotator,
            "benchmark": benchmark,
            "split": split,
            "language": language,
            "register": register,
            "question_en": row["question_en"],
            "answer_en": row["answer_en"],
            "question": row["question"],
            "answer": row.get("answer", ""),
            "esa_score": None,
            "answer_preserved": None,
            "spans": [],
        })
    return review


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write review rows as JSONL and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write review rows as a spreadsheet CSV and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in CSV_COLUMNS if key != "spans_json"}
            record["spans_json"] = json.dumps(row.get("spans", []), ensure_ascii=False)
            record["esa_score"] = row.get("esa_score") if row.get("esa_score") is not None else ""
            preserved = row.get("answer_preserved")
            record["answer_preserved"] = "" if preserved is None else str(bool(preserved)).lower()
            writer.writerow(record)
    return out


def main() -> None:
    """Hydra entry point (deferred imports keep hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(version_base=None, config_path="../../configs", config_name="review_kit")
    def hydra_main_fn(cfg: DictConfig) -> None:
        from pipeline.datagenerator.gate import select_review_sample

        with open(cfg.sample_jsonl, encoding="utf-8") as handle:
            dataset = [json.loads(line) for line in handle if line.strip()]
        by_id = {row["source_id"]: row for row in dataset}
        sample = select_review_sample(
            [{"source_id": r["source_id"], "risk": r.get("risk", 0.0)} for r in dataset],
            "risk", "source_id", cfg.review_top_risk, cfg.review_random, cfg.review_seed,
        )
        for annotator in cfg.annotators:
            for name, ids in (("top_risk", sample["top_risk"]), ("random", sample["random"])):
                rows = build_review_rows(
                    ids, by_id, annotator, cfg.benchmark.name, cfg.split,
                    cfg.language.code, cfg.language.translation_register,
                )
                out = Path(cfg.output_dir) / f"review-{cfg.language.code}-{name}-{annotator}"
                write_jsonl(rows, out.with_suffix(".jsonl"))
                write_csv(rows, out.with_suffix(".csv"))
                print(f"Wrote {len(rows)} rows for {annotator} ({name})")

    hydra_main_fn()


if __name__ == "__main__":
    main()
