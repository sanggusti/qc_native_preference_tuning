"""Registry-driven benchmark translation via Adaption (issue #50).

Reads the benchmark, language and series entries composed by Hydra, selects
the fixed seeded train subset and the full test split, translates both with
the pinned blueprint route, verifies structure and row survival, emits the
canonical schema, pushes train/test splits to the Hub, and uploads the train
split raw for AutoScientist. Pure cores are stub-tested with no network.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.datagenerator.gate import (  # noqa: E402
    extract_final_number,
    source_numbers,
    verify_structure,
)

INSTRUCTION_ROW_ID = "__instruction__"

ISO_639_1 = {"en": "en", "id": "id", "jv": "jv", "su": "su"}


def select_train_indices(n_items: int, train_rows: int, seed: int) -> list[int]:
    """Seeded train subset indices, identical across languages for one series."""
    import random

    if train_rows > n_items:
        raise ValueError(f"train_rows {train_rows} exceeds source size {n_items}")
    return sorted(random.Random(seed).sample(range(n_items), train_rows))


def source_id_for(benchmark: str, split: str, row: dict[str, Any], index: int) -> str:
    """Stable id: the source id column when present, else benchmark-split-index."""
    if row.get("id") not in (None, ""):
        return str(row["id"])
    return f"{benchmark}-{split}-{index:05d}"


def build_blueprint(benchmark_entry: Any, language_entry: Any) -> str:
    """Render the benchmark blueprint with language name and register sentence."""
    get = (lambda key: benchmark_entry[key]) if isinstance(benchmark_entry, dict) else benchmark_entry.__getattr__
    lang = (lambda key: language_entry[key]) if isinstance(language_entry, dict) else language_entry.__getattr__
    return str(get("translate")["blueprint"]).format(
        language=lang("name"), register_instruction=lang("register_instruction")
    )


def blueprint_hash(blueprint: str) -> str:
    """Short hash identifying the exact rendered blueprint."""
    return hashlib.sha256(blueprint.encode("utf-8")).hexdigest()[:16]


def build_run_kwargs(
    translation: Any,
    blueprint: str,
    column_mapping: dict[str, str],
    max_rows: int | None,
    idempotency_key: str,
) -> dict[str, Any]:
    """Pinned Adaption run() kwargs from the series translation block."""
    get = translation.get if isinstance(translation, dict) else translation.__getattr__
    recipes = get("recipes")
    recipes = dict(recipes) if not isinstance(recipes, dict) else recipes
    brand_controls: dict[str, Any] = {
        "blueprint": blueprint,
        "hallucination_mitigation": bool(get("hallucination_mitigation")),
        "safety_categories": list(get("safety_categories") or []),
    }
    if get("length") is not None:
        brand_controls["length"] = get("length")
    job_specification: dict[str, Any] = {"idempotency_key": idempotency_key}
    if max_rows is not None:
        job_specification["max_rows"] = float(max_rows)
    return {
        "column_mapping": column_mapping,
        "recipe_specification": {"recipes": recipes},
        "job_specification": job_specification,
        "brand_controls": brand_controls,
        "training_type": "instruction_dataset",
        # Explicit null clears any expansion config (probe calls persist
        # theirs); verified live after a poisoned smoke run.
        "language_expansion": None,
    }


def expansion_probe_kwargs(iso_639_1: str) -> dict[str, Any]:
    """language_expansion payload probing one code (estimate only, free)."""
    return {"type": "translate", "languages": [iso_639_1], "sample_rate": 1.0}


def probe_language_expansion(
    client: Any, dataset_id: str, column_mapping: dict[str, str], codes: list[str]
) -> dict[str, dict[str, Any]]:
    """Probe the expansion route with estimate=True per code; record accepted set."""
    results = {}
    for code in codes:
        iso = ISO_639_1.get(code)
        if iso is None:
            results[code] = {"accepted": False, "detail": "no ISO 639-1 code"}
            continue
        try:
            response = client.datasets.run(
                dataset_id,
                column_mapping=column_mapping,
                language_expansion=expansion_probe_kwargs(iso),
                estimate=True,
            )
            results[code] = {"accepted": True, "detail": str(response)}
        except Exception as exc:  # noqa: BLE001 - probe records any rejection reason
            results[code] = {"accepted": False, "detail": str(exc)}
    return results


def wait_until_ready(
    client: Any, dataset_id: str, timeout: float = 1800.0, sleep_fn: Any = time.sleep
) -> None:
    """Poll get_status until row_count appears; fail loudly on failed status."""
    waited = 0.0
    while True:
        status = client.datasets.get_status(dataset_id)
        if getattr(status, "row_count", None) is not None:
            return
        if getattr(status, "status", None) == "failed":
            error = getattr(status, "error_data", None)
            raise RuntimeError(getattr(error, "message", str(error)))
        if waited >= timeout:
            raise TimeoutError(f"ingestion of {dataset_id} timed out after {timeout}s")
        sleep_fn(2)
        waited += 2


def check_run_succeeded(result: Any) -> None:
    """Raise on a failed run response, surfacing error_data.message."""
    if getattr(result, "status", None) == "failed":
        error = getattr(result, "error_data", None)
        raise RuntimeError(getattr(error, "message", str(error)))


def dataset_latest_run_id(client: Any, dataset_id: str) -> str | None:
    """Return the dataset's latest run id (None when the dataset has no runs)."""
    pages = client.datasets.list()
    pages = pages.iter_pages() if hasattr(pages, "iter_pages") else [pages]
    for page in pages:
        items = page._get_page_items() if hasattr(page, "_get_page_items") else list(page)
        for dataset in items:
            entry = dataset.model_dump() if hasattr(dataset, "model_dump") else dict(dataset)
            if entry.get("dataset_id") == dataset_id:
                return entry.get("run_id")
    return None


def check_run_is_latest(client: Any, dataset_id: str, run_id: str) -> None:
    """Refuse a download that would alias to another run's output.

    Verified live: download returns the dataset's latest run only and takes
    no run selector, so on a resumed multi-run dataset a deduped run would
    silently download a sibling column's translations.
    """
    latest = dataset_latest_run_id(client, dataset_id)
    if latest != run_id:
        raise RuntimeError(
            f"run {run_id} is not the latest run on dataset {dataset_id} "
            f"(latest is {latest}); the download would alias to another "
            "column. Re-run on a fresh upload instead of resuming."
        )


def translated_completion(row: dict[str, Any]) -> str:
    """Read the translated prompt column; raise when missing or null.

    Verified live: each run translates the mapped prompt column into
    enhanced_completion only (enhanced_prompt stays null with recipes off).
    """
    text = row.get("enhanced_completion")
    if text is None:
        raise KeyError(
            "adapted row lacks a translated enhanced_completion; "
            f"got {sorted(row)}, refusing to fall back to English text"
        )
    return str(text)


def index_by_id(rows: list[dict[str, Any]], id_key: str = "id") -> dict[str, str]:
    """Index translated completions by upload id (download order is unstable)."""
    return {str(row[id_key]): translated_completion(row) for row in rows}


def require_ids(*maps: dict[str, str], expected: list[str]) -> None:
    """Fail loudly when any expected id lacks a translation in any map."""
    for position, mapping in enumerate(maps):
        missing = [source_id for source_id in expected if source_id not in mapping]
        if missing:
            raise ValueError(f"run {position} is missing translations for {missing[:5]}")


def verify_row_survival(downloaded: int, uploaded: int) -> None:
    """Row survival must equal the input count; re-run, never trim."""
    if downloaded != uploaded:
        raise ValueError(f"row survival {downloaded} != input count {uploaded}; re-run")


def verify_instruction(text: str, scorer: str) -> None:
    """Translated instruction keeps the markers the scorer parses."""
    missing = []
    if "ANSWER:" not in text:
        missing.append("ANSWER:")
    if scorer == "choice" and ("{question}" not in text or "{choices}" not in text):
        missing.append("{question}/{choices}")
    if missing:
        raise ValueError(f"translated instruction lost {missing}")


def instruction_text(indexed: dict[str, str], scorer: str) -> str:
    """Extract and verify the translated instruction by its reserved id."""
    if INSTRUCTION_ROW_ID not in indexed:
        raise ValueError("download lacks the instruction row; re-run")
    text = indexed[INSTRUCTION_ROW_ID]
    verify_instruction(text, scorer)
    return text


def translator_version() -> str:
    """Platform version stamp for the translator_version column."""
    from importlib.metadata import version

    return f"adaption-datasets/{version('adaption')}"


def run_column_pair(
    client: Any,
    translation: Any,
    blueprint: str,
    dataset_id: str,
    upload: list[dict[str, str]],
    run_cap: int | None,
    key_prefix: str,
    preflight: Any,
) -> dict[str, dict[str, str]]:
    """Translate prompt columns on one dataset; returns per-column id maps.

    Each run translates its prompt column into enhanced_completion; completion
    maps to the neutral filler (same column twice is rejected live, and the
    sibling column would leak into the output).
    """
    translated: dict[str, dict[str, str]] = {}
    for column in ("question", "answer"):
        kwargs = build_run_kwargs(
            translation, blueprint, {"prompt": column, "completion": "context"},
            run_cap, f"{key_prefix}-{column}",
        )
        estimate = client.datasets.run(dataset_id, **kwargs, estimate=True)
        preflight(column, estimate)
        run_response = client.datasets.run(dataset_id, **kwargs)
        check_run_succeeded(run_response)
        check_run_succeeded(
            client.datasets.wait_for_completion(dataset_id, timeout=7200)
        )
        check_run_is_latest(client, dataset_id, run_response.run_id)
        downloaded = download_rows(client, dataset_id)
        verify_row_survival(len(downloaded), len(upload))
        translated[column] = index_by_id(downloaded)
    return translated


def missing_numerals(source: dict[str, Any], translated: dict[str, Any]) -> list[str]:
    """Source numerals absent from the translation (regen diagnostics)."""
    wanted = source_numbers(
        str(source.get("answer", "")) + " " + str(source.get("question", ""))
    )
    have = str(translated.get("answer", "")) + " " + str(translated.get("question", ""))
    return sorted({number for number in wanted if number not in have})


def normalize_numeral_format(text: str) -> str:
    """Rewrite localized numerals to English formatting, mechanically.

    Indonesian-style thousands dots (12.000) become commas and decimal commas
    (333.200,00) become dots. Correct English numerals (2,400; 2.5; 1,000.00)
    are untouched. Live: some rows keep localizing through regen; the repair
    is verified by re-running the structure check before anything is kept.
    """
    import re

    repaired = text
    for _ in range(5):
        rewritten = re.sub(r"(\d)\.(\d{3})(?=\D|$)", r"\1,\2", repaired)
        if rewritten == repaired:
            break
        repaired = rewritten
    repaired = re.sub(r"(\d),(\d{2})(?=\D|$)", r"\1.\2", repaired)
    return repaired


def regen_blueprint(blueprint: str, details: list[str]) -> str:
    """Append the failure reasons to the blueprint (methodology: regen once)."""
    return (
        blueprint
        + "\n\nPrevious translation rejected ("
        + "; ".join(details)
        + "). Fix exactly that; keep everything else identical."
    )


INSTRUCTION_MARKER_NOTES = {
    # Marker sentences for the dedicated instruction run only. They must not
    # appear in item blueprints: verified live, mentioning ANSWER: primes the
    # model to emit solutions into translated questions.
    "numeric_match": (
        "Keep the marker ANSWER: and the placeholder <number> byte-identical; "
        "never translate or localize them."
    ),
    "choice": (
        "Keep the marker ANSWER:, the placeholder $LETTER, and the template "
        "variables {letters}, {question} and {choices} byte-identical; never "
        "translate or localize them."
    ),
}


def instruction_blueprint(blueprint: str, scorer: str) -> str:
    """Item blueprint plus the marker note for the instruction run."""
    if scorer not in INSTRUCTION_MARKER_NOTES:
        raise ValueError(f"unknown scorer {scorer}")
    return blueprint + " " + INSTRUCTION_MARKER_NOTES[scorer]


def translate_instruction(
    client: Any, translation: Any, blueprint: str, entry: Any,
    bench_name: str, lang_code: str,
) -> tuple[str, str]:
    """Translate the benchmark instruction on a 1-row dataset.

    Returns (instruction, dataset_id). Retried once with the failure reason
    appended; raises when both attempts drop the markers.
    """
    name = entry["instruction_en"] if isinstance(entry, dict) else entry.instruction_en
    scorer = entry["scorer"] if isinstance(entry, dict) else entry.scorer
    upload = [{"id": INSTRUCTION_ROW_ID, "question": str(name),
               "answer": str(name), "context": "No additional context."}]
    current = instruction_blueprint(blueprint, scorer)
    last_error = ""
    for attempt in ("", "-retry"):
        path = write_upload_file(upload)
        try:
            dataset_id = client.datasets.upload_file(
                path, name=f"{bench_name}-{lang_code}-instruction{attempt}"
            ).dataset_id
            print(f"instruction dataset_id: {dataset_id}")
            wait_until_ready(client, dataset_id)
            maps = run_column_pair(
                client, translation, current, dataset_id, upload, 1,
                f"{bench_name}-{lang_code}-instruction-rows1{attempt}",
                lambda column, estimate: None,
            )
        finally:
            Path(path).unlink(missing_ok=True)
        try:
            return instruction_text(maps["question"], scorer), dataset_id
        except ValueError as exc:
            last_error = str(exc)
            current = regen_blueprint(current, [last_error])
    raise ValueError(f"instruction translation failed twice: {last_error}")


def checked_item(
    entry: Any,
    fields: Any,
    item: dict[str, Any],
    translated_columns: dict[str, dict[str, str]],
    instruction: str,
) -> dict[str, Any]:
    """Verify one translated item; returns derived fields plus ok/reasons."""
    source_id = item["source_id"]
    source_row = item["source_row"]
    prompt = translated_columns["question"][source_id]
    completion = translated_columns["answer"][source_id]
    answer_value = source_row[fields["answer"]]
    answer_list = (
        [str(a) for a in answer_value]
        if isinstance(answer_value, list)
        else [str(answer_value)]
    )
    source_choices = (
        [str(c) for c in source_row[fields["choices"]]]
        if "choices" in fields
        else []
    )
    translated_choices: list[str] = []
    choices_error = ""
    if entry["scorer"] == "choice":
        try:
            translated_choices = [str(c) for c in json.loads(completion)]
        except (json.JSONDecodeError, TypeError):
            choices_error = "choices_not_json_list"
    target_letter = (
        chr(ord("A") + source_choices.index(answer_list[0]))
        if entry["scorer"] == "choice"
        else ""
    )
    source = {
        "question": str(source_row[fields["question"]]),
        "answer": _completion_source_text(source_row, fields, entry["scorer"]),
        "choices": source_choices,
        "target": target_letter or "A",
        "instruction": instruction,
    }
    translated = {
        "question": prompt,
        "answer": completion,
        "choices": translated_choices,
        "target": target_letter or "A",
        "instruction": instruction,
    }
    ok, reasons = verify_structure(source, translated, entry["scorer"])
    if choices_error:
        ok, reasons = False, [*reasons, choices_error]
    return {
        "source_id": source_id, "ok": ok, "reasons": reasons,
        "prompt": prompt, "completion": completion,
        "answer_list": answer_list, "source_choices": source_choices,
        "translated_choices": translated_choices, "target_letter": target_letter,
        "source": source, "translated": translated,
    }


def canonical_row(
    source: dict[str, Any],
    prompt: str,
    completion: str,
    instruction: str,
    instruction_en: str,
    question_en: str,
    answer_en: str,
    language: str,
    register: str,
    blueprint_hash_value: str,
    translator: str,
    scorer: str,
    source_id: str,
) -> dict[str, Any]:
    """Emit one canonical schema row (methodology 5.3; quality columns added by the gate)."""
    row: dict[str, Any] = {
        "id": source_id,
        "question": prompt,
        "instruction": instruction,
        "instruction_en": instruction_en,
        "question_en": question_en,
        "answer_en": answer_en,
        "language": language,
        "register": register,
        "blueprint_hash": blueprint_hash_value,
        "translator_version": translator,
    }
    if scorer == "numeric_match":
        target = extract_final_number(completion)
        if target is None:
            raise ValueError(f"translated answer lost its #### number for {source_id}")
        row["target"] = target
    elif scorer == "choice":
        choices = json.loads(completion)
        index = list(source["choices"]).index(source["answer"][0])
        row["choices"] = choices
        row["target"] = chr(ord("A") + index)
    else:
        raise ValueError(f"unknown scorer for canonical row: {scorer!r}")
    return row


def training_pair(instruction: str, question: str, target: str) -> dict[str, str]:
    """Raw SFT row: translated prompt enters the gradient (train_on_inputs=true)."""
    return {"prompt": f"{instruction}\n\n{question}", "completion": target}


def dataset_card(
    benchmark: str,
    language: str,
    splits: dict[str, int],
    register: str,
    blueprint: str,
    run_ids: dict[str, str],
    source_license: str,
    gate_summary: str | None = None,
) -> str:
    """Render the dataset card (gate results filled when the gate stage runs)."""
    return "\n".join(
        [
            f"# {benchmark}-{language}",
            "",
            f"Translated {benchmark} ({language}, {register}).",
            f"Rows: {splits}. Blueprint sha {blueprint_hash(blueprint)}.",
            f"Translator runs: {run_ids} on {date.today().isoformat()}.",
            f"Source license: {source_license}.",
            f"Gate: {gate_summary or 'pending (gate stage #51)'}",
            "Do not train on the test split.",
        ]
    )


def write_upload_file(rows: list[dict[str, str]]) -> str:
    """Write upload rows as JSONL; returns the temp path (caller deletes)."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return handle.name


def download_rows(client: Any, dataset_id: str) -> list[dict[str, Any]]:
    """Download a run as JSONL rows via the binary body (SDK 0.10.0)."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as handle:
        local_path = handle.name
    try:
        client.datasets.download(dataset_id, file_format="jsonl").write_to_file(local_path)
        with open(local_path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    finally:
        Path(local_path).unlink(missing_ok=True)


def print_preflight(
    benchmark: str,
    language: str,
    series: str,
    split: str,
    max_rows: int | None,
    push_to_hub: str | None,
    estimate: Any,
) -> None:
    """Print the AGENTS.md pre-flight checklist before the first paid call."""
    print("Pre-flight: translate")
    print(f"  config: benchmark={benchmark} language={language} series={series}")
    print(f"  smoke: max_rows={max_rows} split={split}")
    print(f"  persistence: push_to_hub={push_to_hub}")
    print(f"  estimate: {estimate}")


def maybe_log_wandb(project: str, run_name: str, tags: list[str], config: Any, extra: dict[str, Any]) -> None:
    """Log the run via the shared tracking module (graceful when offline)."""
    from src.utils.tracking import finish, init_run, log_call, log_metrics

    handle = init_run(project, run_name, tags, config)
    log_metrics(handle, extra)
    log_call(handle, run_name, {"config": config}, extra)
    finish(handle)


def main() -> None:
    """Hydra entry point (deferred imports keep adaption/hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig, OmegaConf

    @hydra_main(version_base=None, config_path="../../configs", config_name="translate_benchmark")
    def hydra_main_fn(cfg: DictConfig) -> None:
        from adaption import Adaption
        from datasets import Dataset, DatasetDict, load_dataset

        series = cfg.series
        language = cfg.language
        entry = cfg.benchmark
        series_name = series["series"]
        bench_name = entry["name"]
        lang_code = language["code"]
        splits = ["train", "test"] if cfg.split == "both" else [cfg.split]
        blueprint = build_blueprint(entry, language)
        print(f"Blueprint sha {blueprint_hash(blueprint)} for {bench_name}/{lang_code}")
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        client = Adaption()
        translation = series["translation"]
        run_ids: dict[str, str] = {}
        pushed: dict[str, Any] = {}
        # One instruction translation per language on its own dataset: the
        # marker note it needs would prime solutions into item translations.
        instruction, instruction_dataset_id = translate_instruction(
            client, translation, blueprint, entry, bench_name, lang_code)
        print(f"instruction: {instruction[:80]}")
        run_ids["instruction"] = {"dataset_id": instruction_dataset_id}
        for split in splits:
            source = entry["train_source" if split == "train" else "eval_source"]
            fields = source["fields"]
            rows = load_dataset(source["hf_repo"], source["config"], split=source["split"])
            indices: Any = range(len(rows))
            if split == "train":
                indices = select_train_indices(
                    len(rows), series["train_rows"], series["sampling_seed"]
                )
            if cfg.max_rows is not None:
                indices = list(indices)[: cfg.max_rows]
            items: list[dict[str, Any]] = []
            for i in indices:
                source_row = dict(rows[i])
                source_id = source_id_for(bench_name, split, source_row, i)
                items.append({
                    "source_id": source_id,
                    "question": str(source_row[fields["question"]]),
                    "answer": _completion_source_text(source_row, fields, entry["scorer"]),
                    "source_row": source_row,
                })
            # The filler column stands in as the completion side of the mapping:
            # each run translates its prompt column, and the sibling column must
            # not be visible or the model translates it into the output too
            # (verified live: solutions leaked into translated questions).
            filler = "No additional context."
            upload = [
                {"id": item["source_id"], "question": item["question"],
                 "answer": item["answer"], "context": filler}
                for item in items
            ]
            upload_path = write_upload_file(upload)
            try:
                if cfg.dataset_id:
                    dataset_id = cfg.dataset_id
                    print(f"resuming on dataset {dataset_id} (upload skipped)")
                else:
                    dataset_id = client.datasets.upload_file(
                        upload_path, name=f"{bench_name}-{lang_code}-{split}"
                    ).dataset_id
                    print(f"dataset_id: {dataset_id} (server may rename the display name)")
                wait_until_ready(client, dataset_id)
                # The run cap must cover len(upload); a smaller cap drops rows.
                # The idempotency scope tracks the effective cap: same params must
                # map to the same key, or retries silently reuse a stale run.
                run_cap = len(upload) if cfg.max_rows is not None else None
                scope = run_cap if run_cap is not None else "full"
                key_prefix = f"{bench_name}-{lang_code}-{split}-rows{scope}"

                def preflight(column: str, estimate: Any, split: str = split) -> None:
                    if column == "question":
                        print_preflight(
                            bench_name, lang_code, series_name, split,
                            cfg.max_rows, cfg.push_to_hub, estimate,
                        )
                    else:
                        print(f"estimate [{column}]: {estimate}")

                translated_columns = run_column_pair(
                    client, translation, blueprint, dataset_id,
                    upload, run_cap, key_prefix, preflight,
                )
                # Snapshot raw translations: download is latest-run-only, so a
                # later crash would otherwise strand the superseded column.
                snapshot_dir = Path(cfg.output_dir) / "_snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = snapshot_dir / f"{bench_name}-{lang_code}-{split}.json"
                snapshot_path.write_text(
                    json.dumps(
                        {"dataset_id": dataset_id, "columns": translated_columns},
                        ensure_ascii=False),
                    encoding="utf-8")
                require_ids(translated_columns["question"], translated_columns["answer"],
                            expected=[item["source_id"] for item in items])
                records = {
                    item["source_id"]: checked_item(
                        entry, fields, item, translated_columns, instruction)
                    for item in items
                }
                failures = {
                    source_id: record["reasons"]
                    for source_id, record in records.items() if not record["ok"]
                }
                regen_dataset_id = ""
                if failures:
                    # Methodology regen ladder, first rung only: re-translate the
                    # failing rows once with the failure reason appended. NLLB
                    # fallback and exclusion need backends that do not exist yet.
                    details: list[str] = []
                    for source_id, reasons in failures.items():
                        record = records[source_id]
                        if "numeral_missing" in reasons:
                            missing = missing_numerals(record["source"], record["translated"])
                            details.append(f"{source_id}: numeral_missing {missing}")
                        else:
                            details.append(f"{source_id}: {','.join(reasons)}")
                    regen_rows = [
                        {"id": item["source_id"], "question": item["question"],
                         "answer": item["answer"], "context": filler}
                        for item in items if item["source_id"] in failures
                    ]
                    print(f"regen {len(regen_rows)} rows: {details}")
                    regen_path = write_upload_file(regen_rows)
                    try:
                        regen_dataset_id = client.datasets.upload_file(
                            regen_path, name=f"{bench_name}-{lang_code}-{split}-regen"
                        ).dataset_id
                        print(f"regen dataset_id: {regen_dataset_id}")
                        wait_until_ready(client, regen_dataset_id)
                        regen_maps = run_column_pair(
                            client, translation, regen_blueprint(blueprint, details),
                            regen_dataset_id, regen_rows, len(regen_rows),
                            f"{key_prefix}-regen", lambda column, estimate: None,
                        )
                    finally:
                        Path(regen_path).unlink(missing_ok=True)
                    require_ids(regen_maps["question"], regen_maps["answer"],
                                expected=[row["id"] for row in regen_rows])
                    for column in ("question", "answer"):
                        translated_columns[column].update(regen_maps[column])
                    records = {
                        item["source_id"]: checked_item(
                            entry, fields, item, translated_columns, instruction)
                        for item in items
                    }
                    still_failing = sorted(
                        source_id for source_id, record in records.items() if not record["ok"])
                    if still_failing:
                        # Last mechanical rung: some rows keep localizing numerals
                        # through regen. Repair the format deterministically and
                        # keep the row only if the structure check then passes.
                        repaired_ids = []
                        for source_id in still_failing:
                            record = records[source_id]
                            if "numeral_missing" not in record["reasons"]:
                                continue
                            trial = {
                                column: dict(translated_columns[column])
                                for column in ("question", "answer")
                            }
                            for column in ("question", "answer"):
                                trial[column][source_id] = normalize_numeral_format(
                                    trial[column][source_id])
                            item = next(
                                i for i in items if i["source_id"] == source_id)
                            fixed = checked_item(
                                entry, fields, item, trial, instruction)
                            if fixed["ok"]:
                                for column in ("question", "answer"):
                                    translated_columns[column][source_id] = trial[column][source_id]
                                records[source_id] = fixed
                                repaired_ids.append(source_id)
                        if repaired_ids:
                            print(f"numeral-format repaired: {repaired_ids}")
                        still_failing = sorted(
                            source_id for source_id, record in records.items()
                            if not record["ok"])
                    if still_failing:
                        reasons = {source_id: records[source_id]["reasons"]
                                   for source_id in still_failing}
                        raise ValueError(
                            f"rows failed after regen {reasons}; "
                            "NLLB fallback/exclusion not implemented")
                canonical = []
                for item in items:
                    record = records[item["source_id"]]
                    canonical.append(
                        canonical_row(
                            {"choices": record["source_choices"], "answer": record["answer_list"]},
                            record["prompt"], record["completion"],
                            instruction, entry["instruction_en"],
                            str(item["source_row"][fields["question"]]),
                            record["answer_list"][0],
                            lang_code, language["translation_register"],
                            blueprint_hash(blueprint), translator_version(),
                            entry["scorer"], item["source_id"],
                        )
                    )
                pushed[split] = canonical
                run_ids[split] = {"dataset_id": dataset_id, "columns": ["question", "answer"]}
                if regen_dataset_id:
                    run_ids[split]["regen_dataset_id"] = regen_dataset_id
                # Persist each split immediately: a later split must never waste
                # a completed one.
                out_dir = Path(cfg.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                split_path = out_dir / f"{bench_name}-{lang_code}-{split}.jsonl"
                with open(split_path, "w", encoding="utf-8") as handle:
                    for row in canonical:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"Wrote {len(canonical)} rows to {split_path}")
            finally:
                Path(upload_path).unlink(missing_ok=True)
        if cfg.push_to_hub:
            DatasetDict({split: Dataset.from_list(pushed[split]) for split in pushed}).push_to_hub(
                cfg.push_to_hub, private=bool(cfg.private)
            )
            print(f"Pushed to https://huggingface.co/datasets/{cfg.push_to_hub}")
        if "train" in pushed:
            raw_path = write_upload_file(
                [training_pair(r["instruction"], r["question"], str(r["target"])) for r in pushed["train"]]
            )
            try:
                raw_id = client.datasets.upload_file(
                    raw_path,
                    name=f"{bench_name}-{lang_code}-train-raw",
                    processing_mode="raw",
                    column_mapping={"prompt": "prompt", "completion": "completion"},
                ).dataset_id
                print(f"Raw train dataset for AutoScientist: {raw_id}")
            finally:
                Path(raw_path).unlink(missing_ok=True)
        maybe_log_wandb(
            series["naming"]["wandb_project"],
            f"datagen-{bench_name}-{lang_code}",
            [bench_name, lang_code, series_name],
            OmegaConf.to_container(cfg, resolve=True),
            {"run_ids": run_ids, "push_to_hub": cfg.push_to_hub},
        )

    hydra_main_fn()


def _completion_source_text(row: Any, fields: Any, scorer: str) -> str:
    get = fields.get if isinstance(fields, dict) else fields.__getattr__
    if scorer == "choice":
        return json.dumps(list(row[get("choices")]), ensure_ascii=False)
    return str(row[get("answer")])


if __name__ == "__main__":
    main()
