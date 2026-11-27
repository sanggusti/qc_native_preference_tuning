"""No-network tests for the translation quality gate with fixtures and stubs."""

import pytest

from pipeline.datagenerator.gate import (
    admission_verdict,
    auroc,
    calibrate_length_band,
    calibrate_lid_threshold,
    decide_row_action,
    hard_checks,
    is_untranslated_echo,
    leak_id_share,
    length_ratio_ok,
    lid_item_verdict,
    orchestrate_gate,
    paired_bootstrap_lower_bound,
    qualifying_sentences,
    risk_score,
    run_judge,
    select_review_sample,
    verify_structure,
)

CFG = {
    "length_ratio_min": 0.5,
    "length_ratio_max": 2.0,
    "echo_chrf_max": 90.0,
    "sentence_min_chars": 20,
    "lid_prob_min": 0.5,
    "lid_verdict_level": "sentence",
    "lid_non_target_max_share": 0.2,
    "lid_reject_codes": ["ind_Latn", "eng_Latn"],
    "refusal_patterns": ["i cannot translate"],
    "metatext_patterns": ["note:"],
    "leak_penalty": 50.0,
    "judge_prompt": "Score {language}:",
    "judge_sample_decile": 0.1,
    "judge_sample_random": 0.1,
    "review_top_risk": 50,
    "review_random": 100,
    "review_seed": 47,
    "admit_chrf_floor": 30.0,
    "admit_lid_share": 0.90,
    "admit_register_share": 0.90,
    "admit_fallback_share": 0.05,
    "admit_max_answer_changed": 0,
    "admit_max_major_critical": 3,
}


class StubLid:
    def __init__(self, label="jav_Latn", prob=0.9):
        self.label_value = label
        self.prob = prob

    def label(self, texts):
        return [(self.label_value, self.prob) for _ in texts]


def _numeric_pair():
    source = {
        "question": "Budi has 3 apples and buys 4 more. How many in total?",
        "answer": "He has 3 + 4 apples. #### 7",
    }
    translated = {
        "question": "Budi duwe 3 apel lan tuku 4 maneh. Piro kabehe?",
        "answer": "Dheweke duwe 3 + 4 apel. #### 7",
        "instruction": "Jawab. End with ANSWER: <number>.",
    }
    return source, translated


def _choice_pair():
    source = {
        "question": "Which organelle produces energy?",
        "choices": ["Nucleus", "Mitochondria", "Ribosome", "Membrane"],
    }
    translated = {
        "question": "Organel apa kang ngasilake energi?",
        "choices": ["Nukleus", "Mitokondria", "Ribosom", "Membran"],
        "target": "B",
        "instruction": "Pilih siji. End with ANSWER: <letter>.",
    }
    return source, translated


def test_verify_structure_numeric_pass_and_failures():
    source, translated = _numeric_pair()
    assert verify_structure(source, translated, "numeric_match") == (True, [])
    bad = dict(translated, answer="Dheweke duwe apel. #### 8")
    ok, reasons = verify_structure(source, bad, "numeric_match")
    assert not ok and "final_number_changed" in reasons
    bad = dict(translated, answer="Dheweke duwe 3 + 4 apel.")
    assert "delimiter_missing" in verify_structure(source, bad, "numeric_match")[1]
    bad = dict(translated, instruction="Jawab.")
    assert "answer_marker_missing" in verify_structure(source, bad, "numeric_match")[1]


def test_source_numbers_ignores_letter_glued_ghosts():
    from pipeline.datagenerator.gate import source_numbers

    assert "9" not in source_numbers("10 (for Mico) + 10 9for Marco) = 20")
    assert source_numbers("$80,000 and 2.5% plus 1500") == ["80,000", "2.5", "1500"]


def test_numeral_format_check_rejects_localized_numbers():
    source = {"question": "It costs 1500 rupiah.", "answer": "#### 1500"}
    translated = {
        "question": "Regane 1.500 rupiah.",
        "answer": "#### 1.500",
        "instruction": "x ANSWER: y",
    }
    ok, reasons = verify_structure(source, translated, "numeric_match")
    assert not ok and "numeral_missing" in reasons


def test_verify_structure_choice_pass_and_failures():
    source, translated = _choice_pair()
    assert verify_structure(source, translated, "choice") == (True, [])
    bad = dict(translated, choices=["Nukleus", "Mitokondria"])
    assert "choice_count_changed" in verify_structure(source, bad, "choice")[1]
    bad = dict(translated, target="Z")
    assert "target_out_of_range" in verify_structure(source, bad, "choice")[1]
    bad = dict(translated, target="BB")
    assert "target_not_letter" in verify_structure(source, bad, "choice")[1]


def test_verify_structure_unknown_scorer_raises():
    with pytest.raises(ValueError, match="unknown scorer"):
        verify_structure({}, {}, "essay")


def test_length_ratio_ok_boundaries():
    assert length_ratio_ok("abcd", "abcdef", 0.5, 2.0)
    assert not length_ratio_ok("abcd", "abcdefghi", 0.5, 2.0)
    assert length_ratio_ok("", "", 0.5, 2.0)
    assert not length_ratio_ok("", "x", 0.5, 2.0)


def test_echo_flags_near_identical_text():
    assert is_untranslated_echo("Budi has 3 apples", "Budi has 3 apples", 90.0)
    assert not is_untranslated_echo("Budi duwe 3 apel lan tuku", "Budi has 3 apples and buys", 90.0)


def test_qualifying_sentences_apply_length_and_noise_floors():
    assert qualifying_sentences("ok this sentence is long enough yes", 20) != []
    assert qualifying_sentences("too short", 20) == []
    assert qualifying_sentences("1500 2.5 3 + 4 = 7 enough words here", 10) == []


def test_lid_item_verdict_sentence_and_item_paths():
    text = (
        "Ukara kapisan iki dawa banget kanggo lolos. "
        "Ukara kapindho uga dawa banget tenan. "
        "Ukara katelu isih dawa banget uga."
    )
    ok, diag = lid_item_verdict(text, "jav_Latn", StubLid("jav_Latn", 0.9), 20, 0.5, 0.2, ["ind_Latn"])
    assert ok and diag["n_qualifying"] == 3
    ok, _ = lid_item_verdict(text, "jav_Latn", StubLid("ind_Latn", 0.9), 20, 0.5, 0.2, ["ind_Latn"])
    assert not ok
    ok, _ = lid_item_verdict("Siji.", "jav_Latn", StubLid("jav_Latn", 0.9), 20, 0.5, 0.2, [])
    assert ok
    ok, _ = lid_item_verdict("Siji.", "jav_Latn", StubLid("jav_Latn", 0.1), 20, 0.5, 0.2, [])
    assert not ok


def test_lid_item_verdict_item_level_skips_sentence_path():
    text = (
        "Ukara kapisan iki dawa banget kanggo lolos. "
        "Ukara kapindho uga dawa banget tenan. "
        "Ukara katelu isih dawa banget uga."
    )
    ok, diag = lid_item_verdict(
        text, "jav_Latn", StubLid("jav_Latn", 0.9), 20, 0.5, 0.2, ["ind_Latn"], "item")
    assert ok and diag["label"] == "jav_Latn"
    ok, _ = lid_item_verdict(
        text, "jav_Latn", StubLid("ind_Latn", 0.9), 20, 0.5, 0.2, ["ind_Latn"], "item")
    assert not ok


def test_hard_checks_pass_good_row_and_block_bad_rows():
    source, translated = _numeric_pair()
    result = hard_checks(source, translated, "numeric_match", "jav_Latn", StubLid(), CFG)
    assert not result["blocked"]
    bad = dict(translated, question="I cannot translate this request at all here")
    result = hard_checks(source, bad, "numeric_match", "jav_Latn", StubLid(), CFG)
    assert result["blocked"] and result["refusal"]
    bad = dict(translated, question=str(source["question"]))
    result = hard_checks(source, bad, "numeric_match", "jav_Latn", StubLid(), CFG)
    assert result["blocked"] and result["echo"]


def test_risk_score_orders_clean_above_noisy():
    assert risk_score([90.0, 90.0, 90.0, 100.0, 95.0], 0.0, 50.0) < 10.0
    assert risk_score([20.0, 20.0], 0.5, 50.0) > 80.0
    assert risk_score([], 0.0, 50.0) == 50.0


def test_leak_id_share_boundaries():
    assert leak_id_share(["saya", "makan"], ["saya", "makan"]) == 1.0
    assert leak_id_share(["kula", "nedha"], ["saya", "makan"]) == 0.0
    assert leak_id_share([], ["saya"]) == 0.0


def test_paired_bootstrap_lower_bound_pass_and_fail():
    assert paired_bootstrap_lower_bound([60.0] * 30, [50.0] * 30, 0.9, 200, 47) > 0
    assert paired_bootstrap_lower_bound([30.0] * 30, [50.0] * 30, 0.9, 200, 47) < 0


def test_auroc_perfect_inverted_degenerate():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 1]) == 0.5


def test_calibrate_lid_threshold_keeps_95_percent():
    probs = [i / 100 for i in range(100)]
    threshold = calibrate_lid_threshold(probs)
    assert sum(p >= threshold for p in probs) >= 95
    assert calibrate_lid_threshold([]) == 0.0


def test_calibrate_length_band_percentiles():
    low, high = calibrate_length_band([1.0] * 100)
    assert (low, high) == (1.0, 1.0)
    assert calibrate_length_band([]) == (0.5, 2.0)


def test_select_review_sample_top_and_random_disjoint():
    rows = [{"source_id": f"s-{i:03d}", "risk": float(i)} for i in range(200)]
    sample = select_review_sample(rows, "risk", "source_id", 50, 100, 47)
    assert sample["top_risk"][0] == "s-199"
    assert len(sample["top_risk"]) == 50 and len(sample["random"]) == 100
    assert not set(sample["top_risk"]) & set(sample["random"])
    again = select_review_sample(rows, "risk", "source_id", 50, 100, 47)
    assert again == sample


def test_decide_row_action_ladder():
    assert decide_row_action(False, 0) == "accept"
    assert decide_row_action(True, 0) == "regenerate"
    assert decide_row_action(True, 1) == "nllb"
    assert decide_row_action(True, 2) == "exclude"


def _passing_metrics():
    return {
        "bootstrap_lower_bound": 1.5,
        "adapt_chrf": 45.0,
        "own_chrf": 45.0,
        "indonesian_chrf": 30.0,
        "lid_share": 0.95,
        "struct_share": 1.0,
        "register_share": 0.95,
        "fallback_share": 0.02,
        "human": {"answer_changed": 0, "major_or_critical": 1},
    }


def test_admission_verdict_all_pass_and_each_rule_flips():
    assert admission_verdict(_passing_metrics(), CFG)["admitted"]
    flips = [
        ("bootstrap_lower_bound", -1.0),
        ("adapt_chrf", 20.0),
        ("lid_share", 0.5),
        ("struct_share", 0.99),
        ("register_share", 0.5),
        ("fallback_share", 0.2),
    ]
    for key, value in flips:
        metrics = _passing_metrics()
        metrics[key] = value
        assert not admission_verdict(metrics, CFG)["admitted"]
    metrics = _passing_metrics()
    metrics["own_chrf"] = 20.0
    assert not admission_verdict(metrics, CFG)["admitted"]
    metrics = _passing_metrics()
    metrics["human"] = {"answer_changed": 1, "major_or_critical": 0}
    assert not admission_verdict(metrics, CFG)["admitted"]


def test_run_judge_skipped_without_fn_and_samples_with_stub():
    rows = [
        {"source_id": f"s-{i:03d}", "risk": float(i), "source_text": "en", "translated_text": "L"}
        for i in range(100)
    ]
    assert run_judge(rows, None, "Score {language}:", "jv", 0.1, 0.1, 47) == {}
    scores = run_judge(
        rows, lambda prompts: [{"esa_score": 70} for _ in prompts],
        "Score {language}:", "jv", 0.1, 0.1, 47,
    )
    assert len(scores) == 20
    assert "s-099" in scores


def test_orchestrate_gate_end_to_end_with_stubs():
    source, good = _numeric_pair()
    bad = dict(good, question="I cannot translate this request at all here")
    rows = [
        {"source_id": "s-001", **good},
        {"source_id": "s-002", **bad},
    ]
    sources = {"s-001": source, "s-002": source}
    report = orchestrate_gate(
        rows, sources, {}, {}, "numeric_match", "jav_Latn", StubLid(), None, CFG, "jv",
    )
    assert report["n_rows"] == 2
    assert report["blocked"] == 1
    assert report["blocked_by_cause"].get("refusal") == 1
    assert report["actions"] == {"accept": 1, "regenerate": 1}
    assert len(report["review_sample"]["top_risk"]) == 2
    assert report["judge_scores"] == {}
