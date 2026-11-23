"""Tests for TOST equivalence on synthetic bootstrap and contrast inputs."""

import random

import yaml

from src.analysis.equivalence import (
    load_equivalence_bound,
    summarize_primary_family,
    tost_from_bootstrap,
    tost_from_contrast,
)


def _draws(mean, sd, n=5000, seed=3):
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


def test_zero_effect_small_noise_is_equivalent():
    result = tost_from_bootstrap(_draws(0.0, 0.01), 0.03)
    assert result["verdict"] == "equivalent"
    assert result["tost_p"] < 0.05
    assert result["lo90"] >= -0.03 and result["hi90"] <= 0.03


def test_six_point_effect_is_different():
    result = tost_from_bootstrap(_draws(0.06, 0.01), 0.03)
    assert result["verdict"] == "different"
    assert result["lo95"] > 0


def test_zero_effect_large_noise_is_undetermined():
    result = tost_from_bootstrap(_draws(0.0, 0.05), 0.03)
    assert result["verdict"] == "undetermined"


def test_contrast_path_matches_bootstrap_path():
    bootstrap = tost_from_bootstrap(_draws(0.0, 0.01), 0.03)
    contrast = tost_from_contrast(0.0, 0.01, 0.03)
    assert contrast["verdict"] == bootstrap["verdict"] == "equivalent"
    different = tost_from_contrast(0.06, 0.01, 0.03)
    assert different["verdict"] == "different"
    assert different["p_lower"] < 0.05 or different["p_upper"] < 0.05


def test_bound_comes_from_series_file_not_literals():
    bound, target = load_equivalence_bound()
    with open("configs/series/s01_language_medium.yaml", encoding="utf-8") as handle:
        series = yaml.safe_load(handle)
    assert bound == series["analysis"]["equivalence_bound_points"] == 3.0
    assert target == series["analysis"]["target_mde_points"] == 5.0
    narrow = tost_from_contrast(0.0, 0.01, 0.005)
    wide = tost_from_contrast(0.0, 0.01, 0.10)
    assert (narrow["verdict"], wide["verdict"]) == ("undetermined", "equivalent")


def test_summary_table_holm_and_both_verdicts():
    contrasts = [
        {
            "benchmark": "gsm8k", "language": "jv", "kind": "Delta_en",
            "estimate": 0.06, "lo95": 0.04, "hi95": 0.08,
            "lo90": 0.045, "hi90": 0.075, "p_difference": 0.001,
        },
        {
            "benchmark": "gsm8k", "language": "su", "kind": "Delta_en",
            "estimate": 0.01, "lo95": -0.01, "hi95": 0.03,
            "lo90": -0.005, "hi90": 0.025, "p_difference": 0.4,
        },
    ]
    table = summarize_primary_family(contrasts, 0.03)
    assert [row["difference"] for row in table] == ["different", "not_different"]
    assert [row["equivalence"] for row in table] == ["different", "equivalent"]
    assert table[0]["p_holm"] < 0.05 < table[1]["p_holm"]
