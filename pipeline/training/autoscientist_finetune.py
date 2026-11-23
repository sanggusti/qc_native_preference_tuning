"""Pinned AutoScientist finetuning runs (issue #54).

One run per benchmark x base x train language x replicate. Hyperparameters are
pinned once from recommend_hyperparams on the English gsm8k primary dataset and
passed verbatim to every later run; the submission is compared against
best_hyperparams and iterations_completed and any mismatch fails loudly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_COLUMNS = {"prompt": "prompt", "completion": "completion"}


def _field(model: Any, key: str) -> Any:
    return model.get(key) if isinstance(model, dict) else getattr(model, key, None)


def validate_base(models: list[Any], base_id: str) -> dict[str, Any]:
    """Find base_id in list_models output; record methods, types, context."""
    for model in models:
        if _field(model, "id") == base_id or _field(model, "model") == base_id:
            return {
                "id": base_id,
                "methods": list(_field(model, "methods") or []),
                "training_types": list(_field(model, "training_types") or []),
                "context_length": _field(model, "context_length"),
            }
    available = [_field(model, "id") for model in models]
    raise ValueError(f"base model {base_id!r} not in list_models output; available: {available}")


def check_dataset(client: Any, dataset_id: str, expected_rows: int) -> dict[str, Any]:
    """Verify the train dataset exists with the expected row count."""
    status = client.datasets.get_status(dataset_id)
    row_count = getattr(status, "row_count", None)
    if row_count != expected_rows:
        raise ValueError(f"train dataset {dataset_id} has {row_count} rows, expected {expected_rows}")
    record: dict[str, Any] = {"dataset_id": dataset_id, "row_count": row_count}
    try:
        meta = client.datasets.get(dataset_id)
        record["processing_mode"] = getattr(meta, "processing_mode", "unreported")
    except Exception:  # noqa: BLE001 - metadata shape varies; row count is the gate
        record["processing_mode"] = "unreported"
    return record


def build_create_kwargs(
    autoscientist: Any,
    hyperparams: dict[str, Any],
    dataset_id: str,
    model: str,
    column_mapping: dict[str, str],
    idempotency_key: str,
) -> dict[str, Any]:
    """create() kwargs: series constants plus verbatim pinned hyperparams."""
    get = autoscientist.get if isinstance(autoscientist, dict) else autoscientist.__getattr__
    return {
        "dataset_id": dataset_id,
        "model": model,
        "data_format": get("data_format"),
        "column_mapping": column_mapping,
        "hyperparams": dict(hyperparams),
        "augmentation_domain_rows": get("augmentation_domain_rows"),
        "augmentation_general_rows": get("augmentation_general_rows"),
        "idempotency_key": idempotency_key,
        "max_iterations": get("max_iterations"),
        "target_win_rate": get("target_win_rate"),
    }


def idempotency_key_for(series: str, benchmark: str, language: str, base: str, replicate: int) -> str:
    """Fresh idempotency key per replicate."""
    return f"{series}-{benchmark}-{language}-{base}-r{replicate}"


def run_name_for(benchmark: str, language: str, base: str, replicate: int) -> str:
    """wandb run name from the train identity (condition is eval-side)."""
    return f"finetune-{benchmark}-{language}-{base}-r{replicate}"


def needs_pinning(series_hyperparams: Any) -> bool:
    """True when the series file has no pinned hyperparams yet."""
    return series_hyperparams is None


def recommend_for_series(
    client: Any, dataset_id: str, model: str, augmentation_domain_rows: float, augmentation_general_rows: float
) -> dict[str, Any]:
    """Recommend hyperparams; returns plain-dict fields plus the resolved model."""
    response = client.autoscientist.recommend_hyperparams(
        dataset_id=dataset_id,
        model=model,
        augmentation_domain_rows=augmentation_domain_rows,
        augmentation_general_rows=augmentation_general_rows,
    )
    raw = response.hyperparams if hasattr(response, "hyperparams") else response.get("hyperparams")
    fields = dict(raw) if isinstance(raw, dict) else raw.model_dump() if hasattr(raw, "model_dump") else dict(vars(raw))
    return {"model": getattr(response, "model", model), "hyperparams": fields}


def write_series_hyperparams(series_path: str | Path, hyperparams: dict[str, Any], source: str) -> None:
    """Write the pinned fields into the series file, preserving other lines."""
    import yaml

    path = Path(series_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("hyperparams:"))
    end = next(i for i, line in enumerate(lines) if line.strip().startswith("hyperparams_source:"))
    block = yaml.safe_dump({"hyperparams": hyperparams}, sort_keys=False).splitlines(keepends=True)
    block = [("  " + line.lstrip()) if line.strip() else line for line in block]
    replacement = block + [f"  hyperparams_source: {source}  # dataset id and date of the recommend call\n"]
    path.write_text("".join(lines[:start] + replacement + lines[end + 1 :]), encoding="utf-8")


def check_match(
    submitted_hyperparams: dict[str, Any], submitted_iterations: float, run: Any
) -> None:
    """Fail loudly when best_hyperparams or iterations differ from the submission."""
    best = run.best_hyperparams
    best_plain = dict(best) if isinstance(best, dict) else best.model_dump() if hasattr(best, "model_dump") else dict(vars(best))
    mismatches = {
        key: (submitted_hyperparams.get(key), best_plain.get(key))
        for key in submitted_hyperparams
        if submitted_hyperparams.get(key) != best_plain.get(key)
    }
    if mismatches:
        raise ValueError(f"best_hyperparams mismatch on {sorted(mismatches)}: {mismatches}")
    completed = float(run.iterations_completed)
    if completed != float(submitted_iterations):
        raise ValueError(f"iterations_completed {completed} != submitted {submitted_iterations}")


def count_tokens(texts: list[str], encode_fn: Any) -> int:
    """Token count via an injected encode function (lazy transformers at runtime)."""
    return sum(len(encode_fn(text)) for text in texts)


def config_commit_hash() -> str:
    """Current git commit hash, or unknown outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def print_preflight(
    benchmark: str, language: str, base: str, replicate: int, hub_model_id: str, dataset_id: str
) -> None:
    """Print the AGENTS.md pre-flight checklist before create()."""
    print("Pre-flight: finetune")
    print(f"  cell: benchmark={benchmark} language={language} base={base} replicate={replicate}")
    print(f"  train dataset: {dataset_id} (raw, row count verified)")
    print(f"  persistence: {hub_model_id}")
    print("  tracking: wandb finetune run with resolved config + best_win_rate")


def download_artifact(client: Any, run_id: str, output_dir: str | Path) -> Path:
    """Download the best checkpoint via the binary body (SDK 0.10.0)."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.bin"
    client.autoscientist.download(run_id).write_to_file(str(path))
    return path


def push_artifact(path: Path, hub_model_id: str, private: bool, upload_fn: Any | None = None) -> str:
    """Push the artifact to the Hub (upload_fn injectable for tests)."""
    if upload_fn is None:
        from huggingface_hub import upload_file

        def upload_fn(path_arg: str, repo_arg: str) -> None:
            upload_file(path_or_fileobj=path_arg, path_in_repo=Path(path_arg).name, repo_id=repo_arg)

    if private:
        from huggingface_hub import create_repo

        create_repo(hub_model_id, private=True, exist_ok=True)
    upload_fn(str(path), hub_model_id)
    return f"https://huggingface.co/{hub_model_id}"


def maybe_log_wandb(
    project: str, run_name: str, tags: list[str], config: Any, metrics: dict[str, Any]
) -> None:
    """Log the run via the shared tracking module (graceful when offline)."""
    from src.utils.tracking import finish, init_run, log_call, log_metrics

    handle = init_run(project, run_name, tags, config)
    log_metrics(handle, metrics)
    log_call(handle, run_name, {"config": config}, metrics)
    finish(handle)


def ensure_raw_dataset(
    client: Any, train_dataset_id: str | None, hub_repo: str, name: str
) -> str:
    """Return a raw train dataset id, uploading from the Hub repo when needed."""
    if train_dataset_id:
        return train_dataset_id
    from datasets import load_dataset

    rows = load_dataset(hub_repo, split="train")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        local_path = handle.name
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "prompt": f"{row['instruction']}\n\n{row['question']}",
                        "completion": str(row["target"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    try:
        return client.datasets.upload_file(
            local_path, name=name, processing_mode="raw", column_mapping=dict(RAW_COLUMNS)
        ).dataset_id
    finally:
        Path(local_path).unlink(missing_ok=True)


def main() -> None:
    """Hydra entry point (deferred imports keep adaption/hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig, OmegaConf

    @hydra_main(version_base=None, config_path="../../configs", config_name="autoscientist_finetune")
    def hydra_main_fn(cfg: DictConfig) -> None:
        from adaption import Adaption

        series = cfg.series
        bench_name, lang_code = cfg.benchmark.name, cfg.language.code
        base_role = cfg.base
        if base_role not in series["base_models"]:
            raise ValueError(f"unknown base role {base_role!r}")
        model = series["base_models"][base_role]
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        client = Adaption()
        validate_base(list(client.autoscientist.list_models()), model)
        auto = series["autoscientist"]
        if needs_pinning(auto["hyperparams"]):
            if not (bench_name == "gsm8k" and lang_code == "en" and base_role == "primary"):
                raise ValueError(
                    "series hyperparams are unpinned: run the English gsm8k primary cell first"
                )
        hub_repo = series["naming"]["hf_dataset"].format(benchmark=bench_name, language=lang_code)
        dataset_id = ensure_raw_dataset(
            client, cfg.train_dataset_id, hub_repo, f"{bench_name}-{lang_code}-train-raw",
        )
        check_dataset(client, dataset_id, series["train_rows"])
        hyperparams = auto["hyperparams"]
        if needs_pinning(hyperparams):
            recommended = recommend_for_series(
                client, dataset_id, model,
                auto["augmentation_domain_rows"], auto["augmentation_general_rows"],
            )
            if cfg.smoke:
                print(f"smoke: recommended hyperparameters: {recommended}")
                return
            from src.utils.registry import CONFIGS_DIR

            write_series_hyperparams(
                CONFIGS_DIR / "series" / f"{series['series']}.yaml",
                recommended["hyperparams"],
                f"{dataset_id} on {date.today().isoformat()}",
            )
            hyperparams = recommended["hyperparams"]
        elif cfg.smoke:
            print("smoke: hyperparams already pinned; nothing to create")
            return
        key = idempotency_key_for(series["series"], bench_name, lang_code, base_role, cfg.replicate)
        kwargs = build_create_kwargs(auto, hyperparams, dataset_id, model, dict(RAW_COLUMNS), key)
        print_preflight(bench_name, lang_code, base_role, cfg.replicate, cfg.hub_model_id, dataset_id)
        run = client.autoscientist.create(**kwargs)
        final = client.autoscientist.wait_for_completion(run.id)
        check_match(dict(hyperparams), auto["max_iterations"], final)
        artifact = download_artifact(client, run.id, cfg.output_dir)
        url = push_artifact(artifact, cfg.hub_model_id, bool(cfg.private))
        print(f"Pushed to {url}")
        maybe_log_wandb(
            series["naming"]["wandb_project"],
            run_name_for(bench_name, lang_code, base_role, cfg.replicate),
            [bench_name, lang_code, base_role, series["series"]],
            OmegaConf.to_container(cfg, resolve=True),
            {
                "best_win_rate": getattr(final, "best_win_rate", None),
                "dataset_id": dataset_id,
                "hub_model_url": url,
                "config_commit": config_commit_hash(),
            },
        )

    hydra_main_fn()


if __name__ == "__main__":
    main()
