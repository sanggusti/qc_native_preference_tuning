"""Import error-span annotations and compute agreement plus error bounds (#58).

Reads per-annotator JSONL back into human_verified/esa_score/answer_preserved
columns, computes inter-annotator agreement, and reports exact one-sided 95%
upper bounds on error rates. Works with one reviewer (flags provisional).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAJOR_OR_CRITICAL = {"major", "critical"}


def parse_annotation_file(path: str | Path) -> list[dict[str, Any]]:
    """Parse one annotated JSONL file into normalized records."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(parse_annotation(json.loads(line)))
    return records


def parse_annotation(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one annotated row (esa score, verdict, severity counts)."""
    spans = row.get("spans", []) or []
    major_or_critical = sum(1 for span in spans if span.get("severity") in MAJOR_OR_CRITICAL)
    return {
        "source_id": row["source_id"],
        "annotator": row.get("annotator"),
        "esa_score": float(row["esa_score"]) if row.get("esa_score") is not None else None,
        "answer_preserved": row.get("answer_preserved"),
        "major_or_critical": major_or_critical,
        "spans": spans,
    }


def cohen_kappa(first: list[bool], second: list[bool]) -> float:
    """Cohen's kappa for two binary verdict lists (no strict zip: lengths checked)."""
    if len(first) != len(second) or not first:
        raise ValueError("kappa needs two non-empty verdict lists of equal length")
    agreements = sum(a == b for a, b in zip(first, second, strict=True))
    observed = agreements / len(first)
    positive = (sum(first) + sum(second)) / (2 * len(first))
    expected = positive * positive + (1 - positive) * (1 - positive)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def pearson_r(first: list[float], second: list[float]) -> float:
    """Pearson correlation for two score lists."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("pearson needs two score lists of equal length >= 2")
    mean_first, mean_second = statistics.fmean(first), statistics.fmean(second)
    numerator = sum((a - mean_first) * (b - mean_second) for a, b in zip(first, second, strict=True))
    denominator = math.sqrt(
        sum((a - mean_first) ** 2 for a in first) * sum((b - mean_second) ** 2 for b in second)
    )
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def _betacf(a: float, b: float, x: float) -> float:
    maxit, eps, tiny = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        result *= delta
        if abs(delta - 1.0) < eps:
            break
    return result


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def error_upper_bound(errors: int, n_items: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper bound on the error rate (Clopper-Pearson).

    Regression anchors from methodology section 6: 0/100 -> 3.0%, 3/100 -> 7.6%.
    """
    if errors >= n_items:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        if _betai(n_items - errors, errors + 1, 1.0 - mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def merge_annotations(annotators: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Merge annotator records: agreement, adjudicated columns, provisional flag."""
    if not annotators or not annotators[0]:
        raise ValueError("need at least one annotator with records")
    first = {record["source_id"]: record for record in annotators[0]}
    if len(annotators) == 1:
        columns = {
            source_id: {
                "human_verified": True,
                "esa_score": record["esa_score"],
                "answer_preserved": record["answer_preserved"],
            }
            for source_id, record in first.items()
        }
        return {"columns": columns, "agreement": None, "provisional": True}
    second = {record["source_id"]: record for record in annotators[1]}
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("annotators share no source ids")
    verdicts_first = [bool(first[source_id]["answer_preserved"]) for source_id in shared]
    verdicts_second = [bool(second[source_id]["answer_preserved"]) for source_id in shared]
    scores_first = [float(first[source_id]["esa_score"]) for source_id in shared]
    scores_second = [float(second[source_id]["esa_score"]) for source_id in shared]
    columns = {}
    for source_id in shared:
        preserved = verdicts_first[shared.index(source_id)] and verdicts_second[shared.index(source_id)]
        columns[source_id] = {
            "human_verified": True,
            "esa_score": statistics.fmean(
                [scores_first[shared.index(source_id)], scores_second[shared.index(source_id)]]
            ),
            "answer_preserved": preserved,
            "needs_adjudication": verdicts_first[shared.index(source_id)]
            != verdicts_second[shared.index(source_id)],
        }
    agreement = {
        "kappa_answer_preserved": cohen_kappa(verdicts_first, verdicts_second),
        "pearson_esa": pearson_r(scores_first, scores_second),
        "mad_esa": statistics.fmean(
            abs(a - b) for a, b in zip(scores_first, scores_second, strict=True)
        ),
        "n_shared": len(shared),
    }
    return {"columns": columns, "agreement": agreement, "provisional": False}


def main() -> None:
    """Hydra entry point (deferred imports keep hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(version_base=None, config_path="../../configs", config_name="review_kit")
    def hydra_main_fn(cfg: DictConfig) -> None:
        annotators = [parse_annotation_file(path) for path in cfg.annotation_files]
        merged = merge_annotations(annotators)
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"annotations-{cfg.language.code}.json"
        path.write_text(json.dumps(merged["columns"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(merged['columns'])} merged records to {path}")
        print(f"Agreement: {merged['agreement']} provisional={merged['provisional']}")

    hydra_main_fn()


if __name__ == "__main__":
    main()
