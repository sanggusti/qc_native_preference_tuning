"""Derived evaluation splits: -rt, -nllb, -pro1 (issue #53).

Pure, tested cores plus a Hydra entry. Repository names render from the
series naming template, never composed by hand.
"""

from __future__ import annotations

import ast
import json
import operator
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_EXPR = re.compile(r"[0-9+\-*/(). ]+")
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def repo_id(naming_template: str, benchmark: str, language: str, suffix: str = "") -> str:
    """Render a dataset repo id from the series naming template plus suffix."""
    return naming_template.format(benchmark=benchmark, language=language, suffix="") + suffix


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows from a local file."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_hf_rows(repo_id: str, split: str) -> list[dict[str, Any]]:
    """Load Hub rows as plain dicts."""
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(repo_id, split=split)]


def build_rt_row(
    source_id: str,
    back_translated: str,
    target: Any,
    instruction_en: str,
    question_en: str,
    answer_en: str,
) -> dict[str, Any]:
    """Canonical English row for the -rt split (same id, original target)."""
    return {
        "id": source_id,
        "question": back_translated,
        "target": target,
        "instruction": instruction_en,
        "instruction_en": instruction_en,
        "question_en": question_en,
        "answer_en": answer_en,
        "language": "en",
        "register": "neutral",
        "variant": "rt",
    }


def build_nllb_row(
    source_id: str,
    translated: str,
    target: Any,
    instruction: str,
    instruction_en: str,
    question_en: str,
    answer_en: str,
    language: str,
    register: str,
) -> dict[str, Any]:
    """Canonical L row for the -nllb split (same id, original target)."""
    return {
        "id": source_id,
        "question": translated,
        "target": target,
        "instruction": instruction,
        "instruction_en": instruction_en,
        "question_en": question_en,
        "answer_en": answer_en,
        "language": language,
        "register": register,
        "variant": "nllb",
    }


def find_numbers(text: str) -> list[str]:
    """Numeric literals in English formatting, in order of appearance."""
    return _NUMBER.findall(text)


def _perturb_int(digits: str, rng: random.Random) -> str:
    if len(digits) <= 1:
        return str(rng.randrange(10))
    return str(rng.randrange(1, 10)) + "".join(str(rng.randrange(10)) for _ in digits[1:])


def perturb_mapping(numbers: list[str], rng: random.Random) -> dict[str, str]:
    """Seeded digit-preserving perturbation per distinct literal.

    Thousands separators are dropped (plain English formatting); decimal
    places are preserved.
    """
    mapping = {}
    for literal in dict.fromkeys(numbers):
        if "." in literal or "," in literal:
            clean = literal.replace(",", "")
            head, dot, tail = clean.partition(".")
            new_head = _perturb_int(head, rng)
            new_tail = "".join(str(rng.randrange(10)) for _ in tail)
            mapping[literal] = f"{new_head}.{new_tail}" if dot else new_head
        else:
            mapping[literal] = _perturb_int(literal, rng)
    return mapping


def substitute_numbers(text: str, mapping: dict[str, str]) -> str:
    """Replace whole numeric literals left to right (longest match wins)."""
    if not mapping:
        return text

    def replace(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    return _NUMBER.sub(replace, text)


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def last_expression(solution: str) -> str | None:
    """Last arithmetic-looking run in the solution text, else None."""
    candidates = [m.group(0).strip() for m in _EXPR.finditer(solution)]
    candidates = [c for c in candidates if re.search(r"\d", c) and re.search(r"[+\-*/]", c)]
    return candidates[-1] if candidates else None


def format_answer(value: float) -> str:
    """Format a recomputed answer: int when integral, else trimmed decimals."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def reinstantiate(
    question: str, answer: str, rng: random.Random
) -> tuple[str, str, dict[str, Any]]:
    """Perturb numeric literals and recompute the #### answer deterministically.

    Returns (new_question, new_answer, manifest). When no arithmetic expression
    can be extracted and evaluated, the item stays unperturbed and the manifest
    records solved=false so analysis can filter it.
    """
    numbers = find_numbers(question + " " + answer)
    mapping = perturb_mapping(numbers, rng)
    manifest: dict[str, Any] = {"numbers": numbers, "mapping": mapping}
    expression = last_expression(answer)
    if expression is None:
        manifest.update({"solved": False, "expression": None})
        return question, answer, manifest
    try:
        new_expression = substitute_numbers(expression, mapping)
        expected = format_answer(_safe_eval(ast.parse(new_expression, mode="eval")))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        manifest.update({"solved": False, "expression": expression})
        return question, answer, manifest
    new_question = substitute_numbers(question, mapping)
    new_solution = substitute_numbers(answer, mapping)
    if "####" in new_solution:
        head = new_solution.split("####", 1)[0]
        new_answer = f"{head}#### {expected}"
    else:
        new_answer = new_solution
    manifest.update({"solved": True, "expression": expression, "expected": expected})
    return new_question, new_answer, manifest


def propagate_pro1(
    rows_by_language: dict[str, dict[str, str]], seed: int
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    """Apply one perturbation mapping per item across every language.

    Input maps language -> {"question": ..., "answer": ...} for a single item;
    the mapping derives from the English row and substitutes structurally into
    every language (numbers are preserved fields). Returns (rows, manifest).
    """
    rng = random.Random(seed)
    english = rows_by_language["en"]
    _, _, manifest = reinstantiate(english["question"], english["answer"], rng)
    mapping = manifest["mapping"]
    out: dict[str, list[dict[str, str]]] = {}
    for language, row in rows_by_language.items():
        if manifest["solved"]:
            new_question = substitute_numbers(row["question"], mapping)
            new_solution = substitute_numbers(row["answer"], mapping)
            head = new_solution.split("####", 1)[0]
            new_answer = f"{head}#### {manifest['expected']}" if "####" in new_solution else new_solution
        else:
            new_question, new_answer = row["question"], row["answer"]
        out[language] = [{"question": new_question, "answer": new_answer}]
    return out, manifest


def main() -> None:
    """Hydra entry point (deferred imports keep hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(version_base=None, config_path="../../configs", config_name="derive_split")
    def hydra_main_fn(cfg: DictConfig) -> None:
        print(f"derive_split variant={cfg.variant} benchmark={cfg.benchmark.name}")
        print("Full orchestration (Hub reads, NLLB joins, pushes) runs with live data;")
        print("pure cores are covered by tests/test_derive_split.py.")

    hydra_main_fn()


if __name__ == "__main__":
    main()
