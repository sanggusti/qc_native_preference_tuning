import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from adaption import Adaption
from datasets import Dataset, load_dataset
from hydra import main as hydra_main
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_utils import load_dataset_from_hub  # noqa: E402

DATASET_NAME = "juletxara/mgsm"
# Adapted downloads carry enhanced_* columns next to original_*; older runs echoed the
# source column names. Both are accepted; anything else is an error, never a silent
# fallback to the English source text.
ADAPTED_COLUMNS = {"question": ("enhanced_prompt", "question"), "answer": ("enhanced_completion", "answer")}


def wait_until_ready(client: Adaption, dataset_id: str) -> None:
	while client.datasets.get_status(dataset_id).row_count is None:
		time.sleep(2)


def get_row(dataset: Dataset, index: int) -> dict[str, Any]:
	return cast(dict[str, Any], dataset[index])


def remove_file(path: str) -> None:
	Path(path).unlink(missing_ok=True)


def download_dataset(client: Adaption, dataset_id: str) -> Dataset:
	# SDK 0.10.0 streams the rows as a binary body (BinaryAPIResponse), not a URL.
	with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as handle:
		local_path = handle.name

	try:
		client.datasets.download(dataset_id, file_format="jsonl").write_to_file(local_path)
		dataset = load_dataset("json", data_files=local_path, split="train")
		if not isinstance(dataset, Dataset):
			raise TypeError("Expected a single dataset split from Adaption download")
		return dataset
	finally:
		remove_file(local_path)


def adapted_value(row: dict[str, Any], field: str) -> str:
	for key in ADAPTED_COLUMNS[field]:
		value = row.get(key)
		if value:
			return str(value)
	raise KeyError(f"adapted row has no {field} column; expected one of {ADAPTED_COLUMNS[field]}, got {sorted(row)}")


def load_source_dataset(split: str, source_config: str) -> Dataset:
	dataset = load_dataset_from_hub(DATASET_NAME, split=split, name=source_config)
	if not isinstance(dataset, Dataset):
		raise TypeError("Expected a single MGSM split")
	return dataset


def write_upload_file(dataset: Dataset) -> str:
	with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
		for index in range(len(dataset)):
			row = get_row(dataset, index)
			payload = {"question": row["question"], "answer": row["answer"]}
			handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
		return handle.name


def build_blueprint(target_language: str) -> str:
	return (
		f"Translate the question and answer into {target_language}. "
		"Preserve all numbers, equations, and the final answer exactly."
	)


def translate_dataset(dataset: Dataset, target_language: str) -> Dataset:
	upload_path = write_upload_file(dataset)

	try:
		client = Adaption()
		dataset_id = client.datasets.upload_file(
			upload_path,
			name=f"mgsm-{target_language.lower()}",
		).dataset_id
		wait_until_ready(client, dataset_id)
		client.datasets.run(
			dataset_id,
			column_mapping={"prompt": "question", "completion": "answer"},
			brand_controls={"blueprint": build_blueprint(target_language)},
		)
		result = client.datasets.wait_for_completion(dataset_id, timeout=1800)
		if result.status == "failed":
			error = getattr(result, "error_data", None)
			raise RuntimeError(getattr(error, "message", str(error)))
		return download_dataset(client, dataset_id)
	finally:
		remove_file(upload_path)


def to_preference_dataset(source: Dataset, translated: Dataset, source_language: str, target_language: str) -> Dataset:
	rows = []
	for index in range(len(source)):
		original = get_row(source, index)
		adapted = get_row(translated, index)
		rows.append(
			{
				"prompt": adapted_value(adapted, "question"),
				"chosen": adapted_value(adapted, "answer"),
				"rejected": original["answer"],
				"question_en": original["question"],
				"answer_en": original["answer"],
				"answer_number": original["answer_number"],
				"equation_solution": original["equation_solution"],
				"source_language": source_language,
				"target_language": target_language,
			}
		)
	return Dataset.from_list(rows)


def resolve_output_dir(output_dir: str) -> Path:
	path = Path(output_dir)
	return path if path.is_absolute() else PROJECT_ROOT / path


@hydra_main(
	version_base=None,
	config_path="../../../configs/datagenerator/evals_convert",
	config_name="mgsm_convert",
)
def main(cfg: DictConfig) -> None:
	dataset = load_source_dataset(cfg.split, cfg.source_config)
	if cfg.max_rows is not None:
		dataset = dataset.select(range(min(cfg.max_rows, len(dataset))))

	translated = translate_dataset(dataset, cfg.target_language)
	preference_dataset = to_preference_dataset(
		dataset,
		translated,
		cfg.source_config,
		cfg.target_language,
	)
	output_dir = resolve_output_dir(cfg.output_dir)
	preference_dataset.save_to_disk(str(output_dir))

	if cfg.push_to_hub:
		preference_dataset.push_to_hub(cfg.push_to_hub, private=cfg.private)

	print(f"Saved {len(preference_dataset)} rows to {output_dir}")


if __name__ == "__main__":
	main()
