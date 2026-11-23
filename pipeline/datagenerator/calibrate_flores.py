"""FLORES-200 calibration translations, one language per run (Phase 1).

Probes the language_expansion route for every series code with free estimates,
then translates devtest with the pinned blueprint route. Reuses the translate
stage's Adaption helpers; outputs feed gate step 0 calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.covariates._common import flores_sentences, write_json  # noqa: E402
from pipeline.datagenerator.translate_benchmark import (  # noqa: E402
    blueprint_hash,
    build_run_kwargs,
    check_run_succeeded,
    download_rows,
    index_by_id,
    maybe_log_wandb,
    print_preflight,
    probe_language_expansion,
    require_ids,
    verify_row_survival,
    wait_until_ready,
    write_upload_file,
)


def render_blueprint(template: str, language_name: str, register_instruction: str) -> str:
    """Render the calibration blueprint for one language."""
    return template.format(language=language_name, register_instruction=register_instruction)


def calibration_key(language: str, max_rows: int | None) -> str:
    """Idempotency key scope for one calibration run."""
    scope = max_rows if max_rows is not None else "full"
    return f"flores-{language}-rows{scope}"


def build_upload_rows(texts: list[str]) -> list[dict[str, str]]:
    """Upload rows with stable ids; completion echoes the source (translated text
    is read off enhanced_completion, quality off the prompt column)."""
    return [
        {"id": f"eng_Latn:devtest:{position:04d}", "question": text, "answer": text}
        for position, text in enumerate(texts)
    ]


def main() -> None:
    """Hydra entry point (deferred imports keep adaption/hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig, OmegaConf

    @hydra_main(version_base=None, config_path="../../configs", config_name="calibrate_flores")
    def hydra_main_fn(cfg: DictConfig) -> None:
        from adaption import Adaption

        series = cfg.series
        language = cfg.language
        lang_code = language["code"]
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        client = Adaption()
        translation = series["translation"]
        column_mapping = {"prompt": "question", "completion": "answer"}
        texts = flores_sentences(
            "facebook/flores", "eng_Latn", "devtest", cache_dir=cfg.output_dir
        )
        if cfg.max_rows is not None:
            texts = texts[: cfg.max_rows]
        upload = build_upload_rows(texts)
        upload_path = write_upload_file(upload)
        try:
            dataset_id = client.datasets.upload_file(
                upload_path, name=f"flores-calibration-{lang_code}"
            ).dataset_id
            wait_until_ready(client, dataset_id)
            probes = probe_language_expansion(
                client, dataset_id, column_mapping, list(series["languages"])
            )
            print(f"expansion probes: {probes}")
            blueprint = render_blueprint(
                cfg.flores_blueprint, language["name"], language["register_instruction"]
            )
            key = calibration_key(lang_code, cfg.max_rows)
            kwargs = build_run_kwargs(translation, blueprint, column_mapping, cfg.max_rows, key)
            estimate = client.datasets.run(dataset_id, **kwargs, estimate=True)
            print_preflight("flores200", lang_code, series["series"], "devtest", cfg.max_rows, cfg.push_to_hub, estimate)
            check_run_succeeded(client.datasets.run(dataset_id, **kwargs))
            check_run_succeeded(client.datasets.wait_for_completion(dataset_id, timeout=7200))
            downloaded = download_rows(client, dataset_id)
            verify_row_survival(len(downloaded), len(upload))
            by_id = index_by_id(downloaded)
            require_ids(by_id, expected=[row["id"] for row in upload])
            source_by_id = {row["id"]: row["question"] for row in upload}
            outputs = [
                {
                    "source_id": row["id"],
                    "source_text": source_by_id[row["id"]],
                    "translated_text": by_id[row["id"]],
                }
                for row in upload
            ]
            path = write_json(
                {
                    "language": lang_code,
                    "blueprint_hash": blueprint_hash(blueprint),
                    "expansion_probes": probes,
                    "rows": outputs,
                },
                cfg.output_dir,
                f"flores-calibration-{lang_code}.json",
            )
            print(f"Wrote {len(outputs)} rows to {path}")
            if cfg.push_to_hub:
                from datasets import Dataset

                Dataset.from_list(outputs).push_to_hub(cfg.push_to_hub, private=bool(cfg.private))
                print(f"Pushed to https://huggingface.co/datasets/{cfg.push_to_hub}")
            maybe_log_wandb(
                series["naming"]["wandb_project"],
                f"datagen-flores-calibration-{lang_code}",
                [lang_code, series["series"]],
                OmegaConf.to_container(cfg, resolve=True),
                {"dataset_id": dataset_id, "row_count": len(outputs), "probes": probes},
            )
        finally:
            Path(upload_path).unlink(missing_ok=True)

    hydra_main_fn()


if __name__ == "__main__":
    main()
