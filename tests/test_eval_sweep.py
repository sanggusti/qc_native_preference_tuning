"""No-network tests for the eval sweep: filters, smoke gate, argv, export."""

import json
from types import SimpleNamespace

from pipeline.evals.run_cells import (
    cell_argv,
    cell_log_dir,
    export_cell_results,
    filter_cells,
    order_cells_checkpoint_first,
    run_name_for,
    should_abort_checkpoint,
    smoke_gate,
)
from pipeline.evals.serve_modal import build_vllm_command, resolve_endpoint
from pipeline.plan import eval_argv
from src.evals.tasks.translated_benchmark import record_to_sample, translated_benchmark
from src.utils.registry import load_series


def _cell(**overrides):
    base = {
        "benchmark": "gsm8k", "eval_language": "jv", "eval_variant": None,
        "condition": "native", "train_language": "jv", "backend": "autoscientist",
        "base_role": "primary", "replicate": 1, "tier": 0,
        "model": "m1", "eval_dataset": "d1",
    }
    base.update(overrides)
    return base


def test_filter_cells_each_dimension():
    cells = [_cell(), _cell(eval_language="id"), _cell(tier=2), _cell(condition="base")]
    assert len(filter_cells(cells, language="jv")) == 3
    assert len(filter_cells(cells, tier=0)) == 3
    assert len(filter_cells(cells, condition="base")) == 1
    assert len(filter_cells(cells, benchmark="medqa")) == 0
    assert len(filter_cells(cells, base="primary")) == 4


def test_order_cells_groups_models_with_en_id_first():
    cells = [
        _cell(model="m2", eval_language="su"),
        _cell(model="m1", eval_language="jv"),
        _cell(model="m1", eval_language="en"),
        _cell(model="m1", eval_language="id"),
    ]
    ordered = order_cells_checkpoint_first(cells)
    assert [c["model"] for c in ordered] == ["m1", "m1", "m1", "m2"]
    assert [c["eval_language"] for c in ordered[:3]] == ["en", "id", "jv"]


def test_should_abort_checkpoint_only_when_en_and_id_fail():
    assert should_abort_checkpoint({"en": False, "id": False})
    assert not should_abort_checkpoint({"en": True, "id": False})
    assert not should_abort_checkpoint({"en": False})


def test_smoke_gate_parsed_answers_and_choice_lines():
    good = [{"score": 1, "completion": "x\nANSWER: 7"}, {"score": 0, "completion": "y"}]
    passes, diag = smoke_gate(good, "numeric_match")
    assert passes and diag["accuracy"] == 0.5
    passes, _ = smoke_gate(good, "choice")
    assert passes
    zero = [{"score": 0, "completion": "nothing parsed here"}]
    assert not smoke_gate(zero, "numeric_match")[0]
    no_lines = [{"score": 1, "completion": "just a letter"}]
    assert not smoke_gate(no_lines, "choice")[0]
    assert not smoke_gate([], "numeric_match")[0]


def test_cell_log_dir_unique_per_cell():
    series = load_series("s01_language_medium")
    first = cell_log_dir("logs", series["series"], _cell())
    second = cell_log_dir("logs", series["series"], _cell(replicate=2))
    third = cell_log_dir("logs", series["series"], _cell(eval_variant="rt"))
    assert len({first, second, third}) == 3
    assert first == "logs/s01_language_medium/gsm8k/jv/native/primary-r1"


def test_cell_argv_matches_planner_plus_log_dir_and_limit():
    series = load_series("s01_language_medium")
    cell = _cell()
    argv = cell_argv(series, cell, "logs/cell", limit=20)
    assert argv[:2] == ["uv", "run"]
    assert argv[2 : 2 + len(eval_argv(series, cell))] == eval_argv(series, cell)
    assert "--log-dir" in argv and "logs/cell" in argv
    assert argv[-2:] == ["--limit", "20"]
    assert " ".join(argv).count("--model ") == 1


def test_resolve_endpoint_exact_single_default_and_missing():
    multi = {"m1": {"base_url": "u1"}, "m2": {"base_url": "u2"}}
    assert resolve_endpoint("m1", multi)["base_url"] == "u1"
    single = {"m1": {"base_url": "u1"}}
    assert resolve_endpoint("anything", single)["base_url"] == "u1"
    try:
        resolve_endpoint("zzz", multi)
    except ValueError as exc:
        assert "no endpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_vllm_command_shape():
    cmd = build_vllm_command("org/model", 8000)
    assert cmd[:4] == ["vllm", "serve", "org/model", "--host"]
    assert "--trust-remote-code" in cmd


def test_export_cell_results_from_fake_log(monkeypatch):
    def fake_sample(score, completion, source_id):
        return SimpleNamespace(
            scores={"match": SimpleNamespace(value=score)},
            output=SimpleNamespace(completion=completion),
            metadata={"source_id": source_id},
        )

    fake_log = SimpleNamespace(samples=[
        fake_sample(1, "reasoning here\nANSWER: 7", "s-1"),
        fake_sample(0, "no answer", "s-2"),
    ])
    import inspect_ai.log

    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda path: fake_log)
    result = export_cell_results(
        "fake.json", 2048, lid_fn=lambda texts: [("jav_Latn", 0.9) for _ in texts]
    )
    assert result["n"] == 2
    assert result["accuracy"]["match"] == 0.5
    assert result["samples"][0]["source_id"] == "s-1"
    assert result["samples"][0]["lid_label"] == "jav_Latn"
    assert result["samples"][0]["cap_hit_approx"] is False


def test_run_name_from_cell_fields():
    assert run_name_for(_cell()) == "eval-gsm8k-jv-native-primary-r1"


def test_instruction_variant_reads_instruction_en(tmp_path):
    rows = [{
        "id": "s-1", "question": "Q?", "target": "7",
        "instruction": "INSTR", "instruction_en": "INSTR_EN", "language": "jv",
    }]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    native = translated_benchmark(benchmark="gsm8k", language="jv", dataset_path=str(path))
    assert native.dataset[0].metadata["instruction"] == "INSTR"
    cot = translated_benchmark(
        benchmark="gsm8k", language="jv", dataset_path=str(path), instruction_variant="english_cot"
    )
    assert cot.dataset[0].metadata["instruction"] == "INSTR_EN"
    assert cot.dataset[0].metadata["instruction_variant"] == "english_cot"
    try:
        record_to_sample(rows[0], None, "german_cot")
    except ValueError as exc:
        assert "unknown instruction_variant" in str(exc)
    else:
        raise AssertionError("expected ValueError")
