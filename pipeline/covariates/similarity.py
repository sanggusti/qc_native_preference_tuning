"""Lexical similarity of each regional language against Indonesian (issue #49).

On NusaX parallel sentences: identical-content-token share and chrF++ of each
regional language against Indonesian, excluding numerals, proper nouns and
preserved fields. The chrF++ here is the project's pure-Python implementation
following Popovic (2017); cross-check against sacrebleu in Phase 0.
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from hydra import main as hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.covariates._common import maybe_log_wandb, write_json  # noqa: E402

_NUMERIC = re.compile(r"^[\d]+(?:[.,][\d]+)*$")


def _is_proper_noun(token: str, position: int) -> bool:
    """Heuristic: capitalized token that does not start the sentence."""
    return position > 0 and token[:1].isupper() and token[1:].islower()


def content_tokens(text: str, preserved: frozenset[str] = frozenset()) -> list[str]:
    """Lowercased tokens minus numerals, proper nouns and preserved fields."""
    tokens = []
    for position, raw in enumerate(text.split()):
        if raw.lower() in preserved or _NUMERIC.match(raw.strip(".,")):
            continue
        if _is_proper_noun(raw, position):
            continue
        tokens.append(raw.lower())
    return tokens


def identical_content_token_share(
    source: str, target: str, preserved: frozenset[str] = frozenset()
) -> float:
    """Jaccard share of content tokens between parallel sentences."""
    left, right = set(content_tokens(source, preserved)), set(content_tokens(target, preserved))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _ngrams(tokens: list[str], n: int) -> Counter[str]:
    return Counter(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _char_ngrams(text: str, n: int) -> Counter[str]:
    cleaned = re.sub(r"\s+", "", text)
    return Counter(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))


def _fscore(hyp: Counter[str], ref: Counter[str], beta: float = 2.0) -> float:
    overlap = sum((hyp & ref).values())
    hyp_total, ref_total = sum(hyp.values()), sum(ref.values())
    if overlap == 0 or hyp_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / hyp_total
    recall = overlap / ref_total
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def chrf_plusplus(hypothesis: str, reference: str) -> float:
    """chrF++ on a 0-100 scale: char 1-6 grams plus word 1-2 grams, beta 2."""
    scores = [_fscore(_char_ngrams(hypothesis, n), _char_ngrams(reference, n)) for n in range(1, 7)]
    hyp_words, ref_words = hypothesis.split(), reference.split()
    scores += [_fscore(_ngrams(hyp_words, n), _ngrams(ref_words, n)) for n in (1, 2)]
    return 100.0 * statistics.fmean(scores)


def similarity_for_pair(
    id_text: str, lang_text: str, preserved: frozenset[str] = frozenset()
) -> dict[str, float]:
    """Share and chrF++ for one parallel sentence pair."""
    return {
        "identical_content_token_share": identical_content_token_share(
            id_text, lang_text, preserved
        ),
        "chrf_plusplus": chrf_plusplus(lang_text, id_text),
    }


@hydra_main(version_base=None, config_path="../../configs/covariates", config_name="covariates")
def main(cfg: DictConfig) -> None:
    from datasets import load_dataset

    from src.utils.registry import load_series

    if not cfg.nusax_id_column or not cfg.nusax_text_columns:
        raise ValueError("set nusax_id_column and nusax_text_columns after the schema smoke check")
    series = load_series(cfg.series)
    codes = [c for c in (list(cfg.languages) if cfg.languages else list(series["languages"]))]
    codes = [c for c in codes if c != "id"]
    rows = load_dataset(cfg.nusax_dataset, split="train")
    id_column = cfg.nusax_text_columns["id"]
    results: dict[str, Any] = {"series": cfg.series, "dataset": cfg.nusax_dataset}
    for code in codes:
        column = cfg.nusax_text_columns[code]
        pairs = [similarity_for_pair(str(r[id_column]), str(r[column])) for r in rows]
        results[code] = {
            "identical_content_token_share": statistics.fmean(p["identical_content_token_share"] for p in pairs),
            "chrf_plusplus": statistics.fmean(p["chrf_plusplus"] for p in pairs),
            "n_pairs": len(pairs),
        }
    path = write_json(results, cfg.output_dir, "similarity.json")
    print(f"Wrote {path}")
    maybe_log_wandb(
        cfg.wandb.project, f"{cfg.wandb.run_prefix}-similarity", cfg,
        {"series": cfg.series, "languages": codes},
    )


if __name__ == "__main__":
    main()
