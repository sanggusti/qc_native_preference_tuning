"""Tokenizer fertility per language per base tokenizer (issue #49).

For each series language and each tokenizer named in the series base_models,
compute tokens per source word, tokens per item, and characters and bytes per
item on FLORES-200 devtest (NusaX when flores_code is null) and on the
project's translated items when they exist.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any, Protocol

from hydra import main as hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.covariates._common import maybe_log_wandb, write_json  # noqa: E402


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def fertility_stats(token_ids: list[int], text: str) -> dict[str, float]:
    """Per-item stats: tokens per word, tokens, characters, bytes."""
    words = text.split()
    encoded = text.encode("utf-8")
    return {
        "tokens_per_word": len(token_ids) / max(len(words), 1),
        "tokens_per_item": float(len(token_ids)),
        "chars_per_item": float(len(text)),
        "bytes_per_item": float(len(encoded)),
    }


def summarize(items: list[dict[str, float]]) -> dict[str, float]:
    """Mean each stat over items (empty input yields zeros)."""
    if not items:
        return {
            "tokens_per_word": 0.0,
            "tokens_per_item": 0.0,
            "chars_per_item": 0.0,
            "bytes_per_item": 0.0,
            "n_items": 0,
        }
    summary = {
        key: statistics.fmean(item[key] for item in items)
        for key in ("tokens_per_word", "tokens_per_item", "chars_per_item", "bytes_per_item")
    }
    summary["n_items"] = len(items)
    return summary


def fertility_for_texts(tokenizer: Tokenizer, texts: list[str]) -> dict[str, float]:
    """Tokenize texts and return the fertility summary."""
    return summarize([fertility_stats(tokenizer.encode(t), t) for t in texts])


def _load_tokenizer(model_id: str) -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("fertility needs the training extra: uv sync --extra training") from exc
    return AutoTokenizer.from_pretrained(model_id)


def _flores_texts(dataset: str, split: str, flores_code: str | None) -> list[str]:
    from datasets import load_dataset

    if flores_code is None:
        raise ValueError("flores_code is null; NusaX fallback is configured per language")
    rows = load_dataset(dataset, flores_code, split=split)
    return [str(row["sentence"]) for row in rows]


@hydra_main(version_base=None, config_path="../../configs/covariates", config_name="covariates")
def main(cfg: DictConfig) -> None:
    from src.utils.registry import load_language, load_series

    series = load_series(cfg.series)
    codes = list(cfg.languages) if cfg.languages else list(series["languages"])
    tokenizers = {role: _load_tokenizer(model) for role, model in series["base_models"].items()}
    results: dict[str, Any] = {"series": cfg.series, "tokenizers": series["base_models"]}
    for code in codes:
        entry = load_language(code)
        texts = _flores_texts(cfg.flores_dataset, cfg.flores_split, entry["flores_code"])
        results[code] = {
            role: fertility_for_texts(tokenizer, texts)
            for role, tokenizer in tokenizers.items()
        }
    path = write_json(results, cfg.output_dir, "fertility.json")
    print(f"Wrote {path}")
    maybe_log_wandb(
        cfg.wandb.project, f"{cfg.wandb.run_prefix}-fertility", cfg,
        {"series": cfg.series, "languages": codes},
    )


if __name__ == "__main__":
    main()
