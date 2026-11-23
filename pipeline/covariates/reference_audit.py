"""Reference audit of FLORES-200 for jv, min, ace, ban, bjn, bug (issue #49).

Register heuristic over jav_Latn references (ngoko vs krama marker lists from
the config) and a diacritic/orthography audit over the Latin-script low-resource
references. The register labels are a documented heuristic, not a classifier.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from hydra import main as hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.covariates._common import maybe_log_wandb, write_json  # noqa: E402


def register_scores(
    text: str, ngoko_markers: list[str], krama_markers: list[str]
) -> dict[str, Any]:
    """Count marker hits and label the text ngoko, krama, mixed or unknown."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    ngoko = sum(w in ngoko_markers for w in words)
    krama = sum(w in krama_markers for w in words)
    if ngoko and krama:
        label = "mixed"
    elif krama:
        label = "krama"
    elif ngoko:
        label = "ngoko"
    else:
        label = "unknown"
    return {"ngoko_hits": ngoko, "krama_hits": krama, "label": label}


def orthography_report(texts: list[str]) -> dict[str, Any]:
    """Character audit: non-ASCII Latin share plus the top codepoints."""
    counts: dict[str, int] = {}
    total = 0
    for text in texts:
        for char in text:
            total += 1
            if ord(char) > 127:
                name = f"U+{ord(char):04X} {unicodedata.name(char, 'UNKNOWN')}"
                counts[name] = counts.get(name, 0) + 1
    flagged = sum(counts.values())
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
    return {
        "n_chars": total,
        "non_ascii_share": flagged / total if total else 0.0,
        "top_codepoints": [{"codepoint": name, "count": n} for name, n in top],
    }


@hydra_main(version_base=None, config_path="../../configs/covariates", config_name="covariates")
def main(cfg: DictConfig) -> None:
    from pipeline.covariates._common import flores_sentences
    from src.utils.registry import load_entry, load_series

    series = load_series(cfg.series)
    codes = list(cfg.languages) if cfg.languages else list(series["languages"])
    results: dict[str, Any] = {"series": cfg.series}
    if "jv" in codes:
        entry = load_entry("language", "jv")
        texts = flores_sentences(
            cfg.flores_dataset, entry["flores_code"], cfg.flores_split,
            cfg.flores_tarball_url, cfg.output_dir,
        )
        labels = [
            register_scores(t, list(cfg.ngoko_markers), list(cfg.krama_markers))["label"]
            for t in texts
        ]
        results["jav_register"] = {
            label: labels.count(label) for label in ("ngoko", "krama", "mixed", "unknown")
        }
        results["jav_register"]["n_items"] = len(texts)
    for code in [c for c in ("min", "ace", "ban", "bjn", "bug") if c in codes]:
        entry = load_entry("language", code)
        texts = flores_sentences(
            cfg.flores_dataset, entry["flores_code"], cfg.flores_split,
            cfg.flores_tarball_url, cfg.output_dir,
        )
        results[f"{code}_orthography"] = orthography_report(texts)
    path = write_json(results, cfg.output_dir, "reference_audit.json")
    print(f"Wrote {path}")
    maybe_log_wandb(
        cfg.wandb.project, f"{cfg.wandb.run_prefix}-reference-audit", cfg,
        {"series": cfg.series, "languages": codes},
    )


if __name__ == "__main__":
    main()
