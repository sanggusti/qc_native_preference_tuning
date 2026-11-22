"""NLLB-200 translation job on Modal (issue #52).

One remote function translates batches; four Hydra modes drive it:
flores_calibration (devtest en -> L), second_opinion (English source items
-> L), back_translation (L items -> en, for quality columns and the -rt
split), eval_split (English test items -> L for -nllb).

Outputs are JSONL rows with source_id, direction and model version, written
locally, optionally pushed to the Hub (container storage is ephemeral), and
logged to wandb. NLLB outputs are CC-BY-NC; dataset cards must say so.

Pure helpers (chunking, code mapping, row building, orchestration) have no
Modal dependency so tests run without the modal package or a GPU.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

MODEL_ID = "facebook/nllb-200-3.3B"
GPU = "A10G"  # smallest GPU fitting 3.3B params in fp16 (~7GB weights + batch)
TIMEOUT_SECONDS = 3600  # covers a full FLORES devtest sweep per language at batch 8

MODES = ("flores_calibration", "second_opinion", "back_translation", "eval_split")


def chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Split rows into batches of at most batch_size (validates the size)."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def resolve_codes(source_language: str, target_language: str) -> tuple[str, str]:
    """Map series language codes to FLORES codes via the registry."""
    from src.utils.registry import load_entry

    src = load_entry("language", source_language)["flores_code"]
    tgt = load_entry("language", target_language)["flores_code"]
    if src is None or tgt is None:
        raise ValueError(
            f"NLLB needs FLORES codes for {source_language}->{target_language}; "
            "a null flores_code has no NLLB path"
        )
    return str(src), str(tgt)


def build_output_row(
    source_id: str,
    source_text: str,
    translated_text: str,
    source_code: str,
    target_code: str,
    mode: str,
    model_id: str = MODEL_ID,
    model_revision: str | None = None,
) -> dict[str, Any]:
    """Build one output row in the schema gate.py and derive_split.py consume."""
    return {
        "source_id": source_id,
        "mode": mode,
        "direction": f"{source_code}->{target_code}",
        "source_code": source_code,
        "target_code": target_code,
        "source_text": source_text,
        "translated_text": translated_text,
        "model_id": model_id,
        "model_revision": model_revision,
        "translated_at": date.today().isoformat(),
        "license": "CC-BY-NC",
    }


def load_input_rows(
    input_jsonl: str | None,
    input_dataset: str | None,
    input_split: str,
    text_column: str,
    id_column: str,
) -> list[dict[str, str]]:
    """Load [{source_id, text}] from exactly one of JSONL or a Hub dataset."""
    if bool(input_jsonl) == bool(input_dataset):
        raise ValueError("set exactly one of input_jsonl or input_dataset")
    if input_jsonl:
        rows = []
        with open(input_jsonl, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append({"source_id": str(row[id_column]), "text": str(row[text_column])})
        return rows
    from datasets import load_dataset

    dataset = load_dataset(str(input_dataset), split=input_split)
    return [{"source_id": str(row[id_column]), "text": str(row[text_column])} for row in dataset]


def flores_rows(dataset: str, split: str, flores_code: str, max_rows: int | None) -> list[dict[str, str]]:
    """Load English FLORES sentences with stable per-split source ids."""
    from datasets import load_dataset

    rows = load_dataset(dataset, flores_code, split=split)
    items = list(rows)[:max_rows] if max_rows else list(rows)
    return [
        {"source_id": f"{flores_code}:{split}:{index:04d}", "text": str(row["sentence"])}
        for index, row in enumerate(items)
    ]


def orchestrate(
    rows: list[dict[str, str]],
    source_code: str,
    target_code: str,
    mode: str,
    batch_size: int,
    translate_batches_fn: Callable[[list[list[str]], str, str], list[str]],
    model_id: str = MODEL_ID,
) -> list[dict[str, Any]]:
    """Translate rows in batches and build output rows (stub-friendly core)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    batches = chunk_rows([{"source_id": r["source_id"], "text": r["text"]} for r in rows], batch_size)
    outputs: list[dict[str, Any]] = []
    flat_texts = [[r["text"] for r in batch] for batch in batches]
    translated = translate_batches_fn(flat_texts, source_code, target_code)
    cursor = 0
    for batch in batches:
        for row in batch:
            outputs.append(
                build_output_row(
                    row["source_id"], row["text"], translated[cursor],
                    source_code, target_code, mode, model_id,
                )
            )
            cursor += 1
    return outputs


def write_jsonl(rows: list[dict[str, Any]], output_dir: str | Path, name: str) -> Path:
    """Write rows as JSONL under output_dir and return the path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def push_jsonl(path: Path, repo_id: str) -> str:
    """Push a JSONL file to the Hub as a dataset and return the URL."""
    from datasets import Dataset

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    Dataset.from_list(rows).push_to_hub(repo_id)
    return f"https://huggingface.co/datasets/{repo_id}"


def maybe_log_wandb(project: str, run_name: str, config: Any, row_count: int) -> None:
    """Log the run to wandb; warn and continue when wandb is not installed."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping run logging (JSONL output still written)")
        return
    run = wandb.init(project=project, name=run_name, config=dict(config))
    run.log({"row_count": row_count})
    run.finish()


if modal is not None:
    app = modal.App(name="qc-native-preference")
    image = (
        modal.Image.debian_slim()
        .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        .uv_pip_install(
            "torch==2.9.1",
            "transformers==4.56.2",
            "accelerate",
            "huggingface_hub",
            "tokenizers",
            "sentencepiece",
            "protobuf",
            "datasets>=2.16.0",
            "pyarrow>=14.0.0",
        )
    )

    _MODEL_CACHE: dict[str, Any] = {}

    @app.function(
        image=image,
        gpu=GPU,
        timeout=TIMEOUT_SECONDS,
        secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    )
    def translate_batch(
        texts: list[str], source_code: str, target_code: str, max_tokens: int
    ) -> list[str]:
        """Translate one batch with NLLB-200 (model cached per container)."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if MODEL_ID not in _MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16
            ).cuda()
            model.eval()
            _MODEL_CACHE[MODEL_ID] = (tokenizer, model)
        tokenizer, model = _MODEL_CACHE[MODEL_ID]
        tokenizer.src_lang = source_code
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        target_id = tokenizer.convert_tokens_to_ids(target_code)
        with torch.no_grad():
            generated = model.generate(
                **inputs, forced_bos_token_id=target_id, max_new_tokens=max_tokens
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_batches_remote(
        batches: list[list[str]], source_code: str, target_code: str, max_tokens: int = 512
    ) -> list[str]:
        """Translate batches via one starmap over the Modal function."""
        results = translate_batch.starmap(
            [(texts, source_code, target_code, max_tokens) for texts in batches]
        )
        return [text for batch in results for text in batch]

    @app.local_entrypoint()
    def modal_main(
        mode: str = "flores_calibration",
        language: str = "jv",
        source_language: str = "en",
        input_dataset: str = "",
        input_split: str = "test",
        text_column: str = "question",
        id_column: str = "id",
        max_rows: int = 20,
        batch_size: int = 8,
        push_to_hub: str = "",
    ) -> None:
        """Modal CLI: uv run modal run pipeline/modal_runner/nllb_translate.py --mode ..."""
        cfg = {
            "mode": mode,
            "source_language": source_language,
            "target_language": language,
            "input_dataset": input_dataset or None,
            "input_split": input_split,
            "text_column": text_column,
            "id_column": id_column,
            "input_jsonl": None,
            "flores_dataset": "facebook/flores",
            "flores_split": "devtest",
            "batch_size": batch_size,
            "max_rows": max_rows,
            "generation_max_tokens": 512,
            "push_to_hub": push_to_hub or None,
            "output_dir": "data/nllb",
            "wandb": {"project": "qc_native_preference_tuning", "run_prefix": "datagen-nllb"},
        }
        run_job(cfg, remote=True)


def resolve_job_input(cfg: dict[str, Any] | Any) -> tuple[list[dict[str, str]], str, str]:
    """Resolve rows and FLORES codes for the configured mode."""
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    mode = get("mode")
    max_rows = get("max_rows")
    if mode == "flores_calibration":
        src_code, tgt_code = resolve_codes(get("source_language"), get("target_language"))
        rows = flores_rows(get("flores_dataset"), get("flores_split"), src_code, max_rows)
        return rows, src_code, tgt_code
    if mode not in ("second_opinion", "back_translation", "eval_split"):
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    rows = load_input_rows(
        get("input_jsonl"), get("input_dataset"), get("input_split"),
        get("text_column"), get("id_column"),
    )
    if max_rows:
        rows = rows[:max_rows]
    return rows, *resolve_codes(get("source_language"), get("target_language"))


def run_job(cfg: dict[str, Any] | Any, remote: bool) -> Path:
    """Resolve input, translate, persist, and log; returns the JSONL path."""
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    mode = get("mode")
    rows, src_code, tgt_code = resolve_job_input(cfg)
    if remote:
        if modal is None:
            raise RuntimeError("remote=true needs the modal package: uv sync --extra modal")

        def translate_batches_fn(
            batches: list[list[str]], src: str, tgt: str
        ) -> list[str]:
            return translate_batches_remote(batches, src, tgt, get("generation_max_tokens"))
    else:

        def translate_batches_fn(
            batches: list[list[str]], src: str, tgt: str
        ) -> list[str]:
            raise RuntimeError("remote=false is a test path; pass an explicit stub")
    outputs = orchestrate(
        rows, src_code, tgt_code, mode, get("batch_size"), translate_batches_fn
    )
    target = get("target_language")
    path = write_jsonl(outputs, get("output_dir"), f"nllb-{mode}-{target}.jsonl")
    print(f"Wrote {len(outputs)} rows to {path}")
    if get("push_to_hub"):
        print(push_jsonl(path, get("push_to_hub")))
    wandb_cfg = get("wandb")
    project = wandb_cfg["project"] if isinstance(wandb_cfg, dict) else wandb_cfg.project
    prefix = wandb_cfg["run_prefix"] if isinstance(wandb_cfg, dict) else wandb_cfg.run_prefix
    maybe_log_wandb(project, f"{prefix}-{mode}-{target}", cfg, len(outputs))
    return path


def main() -> None:
    """Hydra entry point (deferred import keeps modal/hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(
        version_base=None,
        config_path="../../configs/modal_runner",
        config_name="nllb_translate",
    )
    def hydra_main_fn(cfg: DictConfig) -> None:
        if cfg.mode == "flores_calibration" and cfg.max_rows is None:
            print("Pre-flight: full FLORES devtest per language; confirm GPU budget first.")
        run_job(cfg, remote=bool(cfg.remote))

    hydra_main_fn()


if __name__ == "__main__":
    main()
