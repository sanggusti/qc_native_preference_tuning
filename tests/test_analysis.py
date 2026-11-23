"""Tests for the analysis package on synthetic data with known effects."""

import json
import os
import random

import numpy as np
import pandas as pd
import pytest

from src.analysis import inference as inf
from src.analysis import report as rep
from src.analysis.logs import tidy_log, to_correct, to_dataframe


def _synthetic(seed=7, n_items=60, n_reps=3):
    rng = random.Random(seed)
    languages = {"en": 0.0, "jv": -0.3}
    conditions = {"base": 0.0, "native": 0.1, "english_anchor": 0.3}
    rows = []
    for item in range(n_items):
        difficulty = rng.gauss(0, 0.5)
        for lang, lang_eff in languages.items():
            for cond, cond_eff in conditions.items():
                replicates = [0] if cond == "base" else list(range(1, n_reps + 1))
                for replicate in replicates:
                    logit = 0.2 + lang_eff + cond_eff + difficulty
                    prob = 1 / (1 + np.exp(-logit))
                    rows.append({
                        "correct": 1 if rng.random() < prob else 0,
                        "language": lang,
                        "condition": cond,
                        "item": f"i{item}",
                        "itemlang": f"i{item}{lang}",
                        "run": f"{cond}-{replicate}",
                        "fertility": rng.gauss(2.0, 0.3),
                    })
    return pd.DataFrame(rows)


def test_wilson_interval_valid_and_shrinking():
    lo, hi = inf.wilson_interval(50, 100)
    assert 0 <= lo < 0.5 < hi <= 1
    narrow = inf.wilson_interval(500, 1000)
    assert narrow[1] - narrow[0] < hi - lo


def test_item_bootstrap_recovers_mean():
    result = inf.item_bootstrap([1.0] * 60 + [0.0] * 40, n_draws=1000, seed=1)
    assert abs(result["estimate"] - 0.6) < 1e-9
    assert result["lo"] < 0.6 < result["hi"]


def test_two_level_bootstrap_recovers_planted_contrast():
    df = _synthetic(n_items=800, n_reps=6)
    jv = df[df.language == "jv"]
    runs_a = [group["correct"].tolist() for _, group in jv[jv.condition == "english_anchor"].groupby("run")]
    runs_b = [group["correct"].tolist() for _, group in jv[jv.condition == "native"].groupby("run")]
    result = inf.two_level_bootstrap(runs_a, runs_b, n_draws=1000, seed=2)
    assert 0.02 < result["estimate"] < 0.08
    assert result["lo95"] > 0


def test_mcnemar_exact_known_cases():
    assert inf.mcnemar_exact(5, 5) == 1.0
    assert inf.mcnemar_exact(0, 10) < 0.01
    assert inf.mcnemar_exact(0, 0) == 1.0


def test_holm_and_bh_on_known_vectors():
    holm = inf.holm_adjusted([0.01, 0.02, 0.5])
    assert holm == [0.03, 0.04, 0.5]
    bh = inf.bh_adjusted([0.01, 0.02, 0.5])
    assert abs(bh[0] - 0.03) < 1e-9 and bh[2] == 0.5


def test_kendall_tau_known_cases():
    assert inf.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert inf.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_mixed_model_recovers_planted_contrast():
    df = _synthetic(n_items=600)
    fit = inf.fit_series_mixed_model(df, "en", "base")
    assert fit["converged"]
    names = fit["names"]
    anchor = inf.cell_coefficients(names, "jv", "english_anchor", "en", "base")
    native = inf.cell_coefficients(names, "jv", "native", "en", "base")
    contrast = inf.wald_contrast(fit, anchor - native)
    assert contrast["estimate"] > 0 and contrast["p"] < 0.05
    design, _, _ = inf.build_design(df, "en", "base")
    probs_anchor = inf.marginal_cell_probability(
        fit, design[(df.language == "jv").to_numpy() & (df.condition == "english_anchor").to_numpy()]
    )
    probs_native = inf.marginal_cell_probability(
        fit, design[(df.language == "jv").to_numpy() & (df.condition == "native").to_numpy()]
    )
    assert 0.02 < probs_anchor - probs_native < 0.08


def test_e7_verdict_all_three_labels():
    gaps = [
        {"label": "a", "estimate": 0.05, "interval_excludes_zero": True, "above_floor_both": True},
        {"label": "b", "estimate": 0.04, "interval_excludes_zero": True, "above_floor_both": True},
        {"label": "c", "estimate": 0.03, "interval_excludes_zero": True, "above_floor_both": True},
    ]
    same = inf.e7_verdict(gaps, {"a": 0.05, "b": 0.04, "c": 0.03})
    assert same["verdict"] == "exposure-consistent" and same["rank_agrees"]
    flipped = inf.e7_verdict(gaps, {"a": -0.05, "b": 0.04, "c": 0.03})
    assert flipped["verdict"] == "model-specific"
    few = inf.e7_verdict(gaps[:2], {"a": 0.05, "b": 0.04})
    assert few["verdict"] == "not testable at this precision"


def test_replicate_rule_bands_and_pause():
    assert inf.replicate_rule(1.0, 0.8, 1.2, 0.2, 1319)["k"] == 3
    assert inf.replicate_rule(2.0, 1.8, 2.0, 0.2, 1319)["k"] == 5
    assert inf.replicate_rule(2.5, 2.3, 2.5, 0.2, 1319)["k"] == 8
    assert inf.replicate_rule(3.0, 2.8, 3.0, 0.2, 1319)["paused"]
    straddle = inf.replicate_rule(1.4, 1.2, 1.7, 0.2, 1319)
    assert straddle["k"] == 4


def test_power_tables_match_methodology():
    assert abs(inf.mde_paired(0.2, 250) - 7.9) < 0.05
    # methodology table rounds inconsistently at the second cell; 0.1 tolerance
    assert abs(inf.mde_with_replicates(1.5, 3, 0.2, 1319) - 4.8) < 0.1


def test_share_absorbed():
    assert inf.share_absorbed(0.06, 0.02) == pytest.approx(2 / 3)
    assert inf.share_absorbed(0.0, 0.0) == 0.0


def test_to_correct_mapping():
    assert to_correct("C") == 1.0 and to_correct("I") == 0.0
    assert to_correct("correct") == 1.0 and to_correct(0.5) == 0.5
    assert to_correct(None) is None and to_correct("bogus") is None


def test_tidy_log_end_to_end_on_mockllm_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    from inspect_ai import eval as inspect_eval

    from src.evals.tasks.translated_benchmark import translated_benchmark

    dataset_path = tmp_path / "rows.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps({
                "id": f"s-{i}", "question": "2+2?", "target": "4",
                "instruction": "Do math. ANSWER: x", "language": "jv",
            })
            for i in range(2)
        ),
        encoding="utf-8",
    )
    logs = inspect_eval(
        translated_benchmark(benchmark="gsm8k", language="jv", dataset_path=str(dataset_path)),
        model="mockllm/model",
        limit=2,
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "success"
    quality_path = tmp_path / "quality.jsonl"
    quality_path.write_text(
        json.dumps({"source_id": "s-0", "leak_id": 0.1}) + "\n", encoding="utf-8"
    )
    from src.analysis.logs import load_quality_rows

    rows = tidy_log(
        logs[0].location, load_quality_rows(quality_path),
        lid_fn=lambda texts: [("jav_Latn", 0.9) for _ in texts], max_tokens=2048,
    )
    assert len(rows) == 2
    assert rows[0]["source_id"] == "s-0"
    assert rows[0]["language"] == "jv"
    assert rows[0]["quality"]["leak_id"] == 0.1
    assert rows[0]["output_language"] == "jav_Latn"
    assert set(rows[0]["correct"]) == {"match", "numeric_match_locale"}
    assert len(to_dataframe(rows)) == 2


def test_report_tables_order_and_statement():
    assert rep.TABLE_ORDER[0] == "covariates_gate"
    assert rep.TABLE_ORDER[-1] == "identification_statement"
    table = rep.render_markdown_table([{"a": 1, "b": 2}], ["a", "b"])
    assert "| a | b |" in table
    statement = rep.identification_statement({"primary_summary": "H1 holds", "e7": "consistent"})
    assert "H1 holds" in statement and "E10" in statement


def _figure_size(path):
    return os.path.getsize(path)


def test_varied_figures_all_render(tmp_path):
    labels = ["jv", "su", "min"]
    forest = rep.forest_plot(labels, [2.0, -1.0, 0.5], [0.5, -2.0, -1.0], [3.5, 0.0, 2.0], tmp_path / "f.png", equivalence_bound=3.0)
    scatter = rep.exposure_scatter(
        [{"label": label, "bpb": 1.0 + i, "accuracy": 0.5, "fertility": 2.0, "quality": 0.8} for i, label in enumerate(labels)],
        "bpb", tmp_path / "s.png",
    )
    dumbbell = rep.dumbbell_plot(labels, [0.4, 0.3, 0.2], [0.5, 0.45, 0.3], tmp_path / "d.png")
    heatmap = rep.accuracy_heatmap(["base", "native"], labels, [[0.4, 0.3, 0.2], [0.5, 0.45, 0.3]], tmp_path / "h.png")
    lollipop = rep.lollipop_plot(labels, [5.0, 7.0, 9.0], [2.0, 3.0, 4.0], tmp_path / "l.png")
    strip = rep.replicate_strip(labels, [[0.4, 0.42, 0.38], [0.3, 0.31], [0.2, 0.22, 0.21]], tmp_path / "s2.png")
    for path in (forest, scatter, dumbbell, heatmap, lollipop, strip):
        assert _figure_size(path) > 5000
