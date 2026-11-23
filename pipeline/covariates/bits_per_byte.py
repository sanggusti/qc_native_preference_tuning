"""Bits per byte of each candidate base model (issue #49).

GPU job: run on Modal (see the compute skill) with the pre-flight checklist.
Measures base-model bits per byte on the same parallel sentences as fertility,
paired with a held-out slice of project items when provided.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

from hydra import main as hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.covariates._common import maybe_log_wandb, write_json  # noqa: E402


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """Convert summed token NLL (nats) over total bytes to bits per byte."""
    if total_bytes <= 0:
        return 0.0
    return total_nll_nats / (total_bytes * math.log(2))


def aggregate(per_item_nll: list[float], per_item_bytes: list[int]) -> dict[str, float]:
    """Aggregate per-item NLL and byte counts into bits per byte plus spread."""
    total_nll = sum(per_item_nll)
    total_bytes = sum(per_item_bytes)
    per_item_bpb = [
        bits_per_byte(nll, nbytes)
        for nll, nbytes in zip(per_item_nll, per_item_bytes, strict=True)
    ]
    return {
        "bits_per_byte": bits_per_byte(total_nll, total_bytes),
        "stdev_per_item": statistics.pstdev(per_item_bpb) if per_item_bpb else 0.0,
        "n_items": len(per_item_nll),
        "total_bytes": total_bytes,
    }


def run_bpb(model_id: str, texts: list[str]) -> dict[str, float]:
    """Score texts under model_id and return the BPB aggregate (needs GPU env)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("bits_per_byte needs the training extra: uv sync --extra training") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    nlls: list[float] = []
    byte_counts: list[int] = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            outputs = model(ids, labels=ids)
            nlls.append(float(outputs.loss) * ids.numel())
            byte_counts.append(len(text.encode("utf-8")))
    return aggregate(nlls, byte_counts)


@hydra_main(version_base=None, config_path="../../configs/covariates", config_name="covariates")
def main(cfg: DictConfig) -> None:
    import json

    from pipeline.covariates._common import flores_sentences
    from src.utils.registry import load_entry, load_series

    print("Pre-flight: GPU job, run on Modal with timeout set; outputs persist via this JSON.")
    series = load_series(cfg.series)
    codes = list(cfg.languages) if cfg.languages else list(series["languages"])
    heldout: dict[str, list[str]] = {}
    if cfg.heldout_items:
        with open(cfg.heldout_items, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                heldout.setdefault(row["language"], []).append(row["text"])
    results: dict[str, Any] = {"series": cfg.series, "models": series["base_models"]}
    for code in codes:
        entry = load_entry("language", code)
        texts = flores_sentences(
            cfg.flores_dataset, entry["flores_code"], cfg.flores_split,
            cfg.flores_tarball_url, cfg.output_dir,
        )
        results[code] = {}
        for role, model in series["base_models"].items():
            cell: dict[str, Any] = {"flores": run_bpb(model, texts)}
            if code in heldout:
                cell["heldout"] = run_bpb(model, heldout[code])
            results[code][role] = cell
    path = write_json(results, cfg.output_dir, "bits_per_byte.json")
    print(f"Wrote {path}")
    maybe_log_wandb(
        cfg.wandb.project, f"{cfg.wandb.run_prefix}-bits-per-byte", cfg,
        {"series": cfg.series, "languages": codes},
    )


if __name__ == "__main__":
    main()
