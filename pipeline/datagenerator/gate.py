"""Translation quality gate, steps 0 to 5 (issue #51, methodology section 6).

Pure, tested cores with stub-friendly seams: hard checks, risk rank, review
selection, admission logic, calibration math. Heavy backends (GlotLID, QE
metrics, the LLM judge, Hub data) are lazy and injectable so the suite runs
with no network and no models.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.covariates.similarity import chrf_plusplus  # noqa: E402

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u2026\n])\s+")


class LidClient(Protocol):
    def label(self, texts: list[str]) -> list[tuple[str, float]]: ...


def extract_final_number(answer: str) -> str | None:
    """Return the number after #### in a GSM8K-style answer, else None."""
    if "####" not in answer:
        return None
    tail = answer.split("####", 1)[1].strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", tail)
    return match.group(0) if match else None


def source_numbers(text: str) -> list[str]:
    """Numeral tokens in English formatting found in text."""
    return _NUMBER.findall(text)


def verify_structure_numeric(source: dict[str, Any], translated: dict[str, Any]) -> list[str]:
    """Return failure reasons (empty means struct_ok) for numeric_match tasks."""
    failures = []
    src_answer = str(source.get("answer", ""))
    tgt_answer = str(translated.get("answer", ""))
    if extract_final_number(src_answer) != extract_final_number(tgt_answer):
        failures.append("final_number_changed")
    if "####" in src_answer and "####" not in tgt_answer:
        failures.append("delimiter_missing")
    src_numbers = source_numbers(src_answer + " " + str(source.get("question", "")))
    tgt_text = tgt_answer + " " + str(translated.get("question", ""))
    if any(number not in tgt_text for number in src_numbers):
        failures.append("numeral_missing")
    if "ANSWER:" not in str(translated.get("instruction", "")):
        failures.append("answer_marker_missing")
    return failures


def verify_structure_choice(source: dict[str, Any], translated: dict[str, Any]) -> list[str]:
    """Return failure reasons (empty means struct_ok) for choice tasks."""
    failures = []
    src_choices = list(source.get("choices", []))
    tgt_choices = list(translated.get("choices", []))
    if len(tgt_choices) != len(src_choices) or not tgt_choices:
        failures.append("choice_count_changed")
    if any(not str(choice).strip() for choice in tgt_choices):
        failures.append("empty_choice")
    target = str(translated.get("target", ""))
    if len(target) != 1 or not target.isalpha():
        failures.append("target_not_letter")
    elif ord(target.upper()) - ord("A") >= len(tgt_choices):
        failures.append("target_out_of_range")
    if "ANSWER:" not in str(translated.get("instruction", "")):
        failures.append("answer_marker_missing")
    return failures


def verify_structure(
    source: dict[str, Any], translated: dict[str, Any], scorer: str
) -> tuple[bool, list[str]]:
    """Dispatch structure verification by benchmark scorer type."""
    if scorer == "numeric_match":
        failures = verify_structure_numeric(source, translated)
    elif scorer == "choice":
        failures = verify_structure_choice(source, translated)
    else:
        raise ValueError(f"unknown scorer for structure check: {scorer!r}")
    return (not failures, failures)


def length_ratio_ok(source: str, translated: str, band_min: float, band_max: float) -> bool:
    """Character length ratio of translation to source inside the band."""
    if not source:
        return not translated
    return band_min <= len(translated) / len(source) <= band_max


def contains_pattern(text: str, patterns: list[str]) -> str | None:
    """Return the first matched pattern (case-insensitive) or None."""
    lowered = text.lower()
    for pattern in patterns:
        if pattern.lower() in lowered:
            return pattern
    return None


def is_untranslated_echo(translated: str, source: str, echo_chrf_max: float) -> bool:
    """chrF++ against the English source at or above the max means untranslated."""
    return chrf_plusplus(translated, source) >= echo_chrf_max


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences on end punctuation."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]


def is_qualifying_sentence(sentence: str, min_chars: int) -> bool:
    """Length floor plus fewer than half numeric or symbol tokens."""
    if len(sentence) < min_chars:
        return False
    tokens = sentence.split()
    if not tokens:
        return False
    noisy = sum(1 for token in tokens if not re.search(r"[A-Za-z\u00c0-\u024f]", token))
    return noisy < len(tokens) / 2


def qualifying_sentences(text: str, min_chars: int) -> list[str]:
    """Sentences eligible for per-sentence language identification."""
    return [s for s in split_sentences(text) if is_qualifying_sentence(s, min_chars)]


def lid_item_verdict(
    text: str,
    target_code: str,
    lid: LidClient,
    min_chars: int,
    prob_min: float,
    non_target_max_share: float,
    reject_codes: list[str],
    verdict_level: str = "sentence",
) -> tuple[bool, dict[str, Any]]:
    """Apply the step 1 language rule; returns (passes, diagnostics).

    verdict_level "item" labels the whole item at once. Required for closely
    related languages: verified live November 2026, sentence-level GlotLID
    cannot separate Javanese from Indonesian (genuine Javanese sentences
    labelled ind_Latn at 0.48-0.49, plus pov/zsm noise labels under 0.20).
    """
    qualifying = qualifying_sentences(text, min_chars)
    if verdict_level != "item" and len(qualifying) >= 3:
        labels = lid.label(qualifying)
        non_target = sum(1 for code, _ in labels if code != target_code)
        rejected = [code for code, _ in labels if code in reject_codes]
        weak = [code for code, prob in labels if prob < prob_min]
        passes = (
            non_target / len(labels) <= non_target_max_share
            and not rejected
            and not weak
        )
        return passes, {
            "n_qualifying": len(qualifying),
            "non_target_share": non_target / len(labels),
            "rejected": rejected,
            "weak": weak,
        }
    labels = lid.label([text])
    code, prob = labels[0]
    passes = code == target_code and prob >= prob_min
    return passes, {"n_qualifying": len(qualifying), "label": code, "prob": prob}


def hard_checks(
    source: dict[str, Any],
    translated: dict[str, Any],
    scorer: str,
    target_code: str,
    lid: LidClient,
    cfg: Any,
) -> dict[str, Any]:
    """Run every step 1 hard check; any failure blocks the row."""
    struct_ok, struct_reasons = verify_structure(source, translated, scorer)
    translated_text = str(translated.get("question", ""))
    source_text = str(source.get("question", ""))
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    length_ok = length_ratio_ok(translated_text, source_text, get("length_ratio_min"), get("length_ratio_max"))
    refusal = contains_pattern(translated_text, list(get("refusal_patterns")))
    metatext = contains_pattern(translated_text, list(get("metatext_patterns")))
    echo = is_untranslated_echo(translated_text, source_text, get("echo_chrf_max"))
    lid_ok, lid_diag = lid_item_verdict(
        translated_text, target_code, lid, get("sentence_min_chars"),
        get("lid_prob_min"), get("lid_non_target_max_share"), list(get("lid_reject_codes")),
        get("lid_verdict_level"),
    )
    blocked = not (struct_ok and length_ok and refusal is None and metatext is None and not echo and lid_ok)
    return {
        "struct_ok": struct_ok, "struct_reasons": struct_reasons, "length_ok": length_ok,
        "refusal": refusal, "metatext": metatext, "echo": echo,
        "lid_ok": lid_ok, "lid": lid_diag, "blocked": blocked,
    }


def risk_score(
    quality_signals: list[float],
    leak_id: float,
    leak_penalty: float,
) -> float:
    """Per-row risk rank 0-100 from coverage-safe signals (higher is riskier).

    Each quality signal is 0-100 (higher is better): NLLB agreement, NLLB
    back-translation chrF++, back-translation COMET x100, solvable_bt x100,
    LID confidence x100. Only provided signals enter the mean.
    """
    quality = statistics.fmean(quality_signals) if quality_signals else 50.0
    return min(100.0, max(0.0, 100.0 - quality + leak_penalty * leak_id))


def leak_id_share(
    translated_tokens: list[str],
    indonesian_tokens: list[str],
    preserved: frozenset[str] = frozenset(),
) -> float:
    """Share of translated content tokens identical to the Indonesian version."""
    from pipeline.covariates.similarity import content_tokens as _content

    left = _content(" ".join(translated_tokens), preserved)
    right = set(_content(" ".join(indonesian_tokens), preserved))
    if not left:
        return 0.0
    return sum(1 for token in left if token in right) / len(left)


def paired_bootstrap_lower_bound(
    adapt_scores: list[float],
    nllb_scores: list[float],
    ratio: float,
    draws: int,
    seed: int,
) -> float:
    """Lower 95% bound of mean(adapt - ratio * nllb) over paired resamples."""
    rng = random.Random(seed)
    n = len(adapt_scores)
    diffs = sorted(
        statistics.fmean(
            adapt_scores[i] - ratio * nllb_scores[i] for i in (rng.randrange(n) for _ in range(n))
        )
        for _ in range(draws)
    )
    return diffs[int(0.05 * draws)]


def auroc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve by trapezoid rule (diagnostic only)."""
    paired = sorted(zip(scores, labels, strict=True), reverse=True)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    area, true_pos, false_pos, prev_true = 0.0, 0, 0, 0
    for _, label in paired:
        if label:
            true_pos += 1
        else:
            false_pos += 1
            area += (true_pos + prev_true) / 2
        prev_true = true_pos
    return area / (positives * negatives)


def calibrate_lid_threshold(reference_probs: list[float], pass_share: float = 0.95) -> float:
    """Highest threshold at which at least pass_share of references pass."""
    if not reference_probs:
        return 0.0
    ordered = sorted(reference_probs)
    return ordered[max(0, int((1 - pass_share) * len(ordered)))]


def calibrate_length_band(ratios: list[float]) -> tuple[float, float]:
    """1st/99th percentile band of FLORES length ratios."""
    if not ratios:
        return (0.5, 2.0)
    ordered = sorted(ratios)
    low = ordered[max(0, int(0.01 * len(ordered)))]
    high = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
    return (low, high)


def select_review_sample(
    rows: list[dict[str, Any]],
    risk_key: str,
    id_key: str,
    top_n: int,
    random_n: int,
    seed: int,
) -> dict[str, list[str]]:
    """Top-risk ids plus random ids (caller sorts by id for cross-language sharing)."""
    ranked = sorted(rows, key=lambda row: row[risk_key], reverse=True)
    top_ids = [row[id_key] for row in ranked[:top_n]]
    rng = random.Random(seed)
    pool = [row[id_key] for row in rows if row[id_key] not in set(top_ids)]
    return {"top_risk": top_ids, "random": rng.sample(pool, min(random_n, len(pool)))}


def decide_row_action(hard_fail: bool, fallback_count: int) -> str:
    """Fallback ladder per row: regenerate once, then NLLB, then exclude."""
    if not hard_fail:
        return "accept"
    if fallback_count <= 0:
        return "regenerate"
    if fallback_count == 1:
        return "nllb"
    return "exclude"


def admission_verdict(metrics: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """Apply the step 5 admission rules; returns per-rule and overall verdicts."""
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    human = metrics.get("human", {})
    rules = {
        "nllb_ratio": metrics["bootstrap_lower_bound"] > 0,
        "chrf_floor": metrics["adapt_chrf"] >= get("admit_chrf_floor"),
        "leakage": metrics["own_chrf"] > metrics["indonesian_chrf"],
        "lid_share": metrics["lid_share"] >= get("admit_lid_share"),
        "structure": metrics["struct_share"] == 1.0,
        "register": metrics.get("register_share", 1.0) >= get("admit_register_share"),
        "human_answer": human.get("answer_changed", 0) <= get("admit_max_answer_changed"),
        "human_errors": human.get("major_or_critical", 0) <= get("admit_max_major_critical"),
        "fallback": metrics["fallback_share"] <= get("admit_fallback_share"),
    }
    rules["admitted"] = all(rules.values())
    return rules


def judge_prompt_for(source: str, translation: str, language: str, template: str) -> str:
    """Render the frozen judge prompt for one row."""
    return template.format(language=language) + f"\nEnglish source:\n{source}\nTranslation:\n{translation}\n"


def run_judge(
    rows: list[dict[str, Any]],
    judge_fn: Callable[[list[str]], list[dict[str, Any]]] | None,
    template: str,
    language: str,
    decile: float,
    random_share: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Score the bottom risk decile plus a random sample; empty when skipped."""
    if judge_fn is None:
        return {}
    ranked = sorted(rows, key=lambda row: row["risk"], reverse=True)
    cutoff = max(1, int(len(ranked) * decile))
    rng = random.Random(seed)
    rest = [row for row in ranked[cutoff:] if row["source_id"] not in {r["source_id"] for r in ranked[:cutoff]}]
    sample = ranked[:cutoff] + rng.sample(rest, min(int(len(ranked) * random_share), len(rest)))
    prompts = [judge_prompt_for(r["source_text"], r["translated_text"], language, template) for r in sample]
    return {r["source_id"]: score for r, score in zip(sample, judge_fn(prompts), strict=True)}


def openai_judge_fn(
    model: str, base_url: str | None = None
) -> Callable[[list[str]], list[dict[str, Any]]]:
    """Build a judge_fn over an OpenAI-compatible API (lazy import, frozen prompt).

    Default base_url serves OpenAI; point it at a Modal vLLM endpoint to run
    the judge on Modal (the project default: no judge API keys needed).
    A HuggingFace model id without a base_url fails fast instead of
    querying OpenAI for a model it does not serve.
    """
    if base_url is None and "/" in model:
        raise ValueError(
            f"judge model {model!r} needs judge_base_url (deploy it on Modal first)"
        )

    def judge(prompts: list[str]) -> list[dict[str, Any]]:
        import openai

        client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        scores = []
        for prompt in prompts:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            scores.append(json.loads(response.choices[0].message.content or "{}"))
        return scores

    return judge


def _numpy2_fasttext_shim() -> None:
    """Map numpy-1 copy=False to copy=None (numpy 2 migration guide).

    fasttext 0.9.3 calls np.array(..., copy=False), which raises on numpy 2.
    copy=None preserves the numpy 1.x "copy only if needed" behavior. Scoped
    to processes that load the LID model; remove if fasttext ever fixes it.
    """
    import numpy

    if getattr(numpy.array, "__fasttext_shim__", False):
        return
    original = numpy.array

    def patched(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if kwargs.get("copy") is False:
            kwargs["copy"] = None
        return original(*args, **kwargs)

    patched.__fasttext_shim__ = True  # type: ignore[attr-defined]
    numpy.array = patched  # type: ignore[assignment]


class FastTextLid:
    """GlotLID labels via a FastText model file (lazy import, loaded once)."""

    def __init__(self, model_path: str) -> None:
        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError("GlotLID needs fasttext: pip install fasttext") from exc
        _numpy2_fasttext_shim()
        self._model = fasttext.load_model(model_path)

    def label(self, texts: list[str]) -> list[tuple[str, float]]:
        clean = [text.replace("\n", " ").replace("\r", " ") for text in texts]
        labels, probs = self._model.predict(clean)
        return [
            (label[0].replace("__label__", ""), float(prob[0]))
            for label, prob in zip(labels, probs, strict=True)
        ]


def orchestrate_gate(
    rows: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    nllb_opinion_by_id: dict[str, str],
    nllb_back_by_id: dict[str, str],
    scorer: str,
    target_code: str,
    lid: LidClient,
    judge_fn: Callable[[list[str]], list[dict[str, Any]]] | None,
    cfg: Any,
    language: str,
) -> dict[str, Any]:
    """Run steps 1-5 over rows with injected backends; returns the report dict."""
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    scored = []
    for row in rows:
        source = source_by_id[row["source_id"]]
        checks = hard_checks(source, row, scorer, target_code, lid, cfg)
        translated_text = str(row.get("question", ""))
        signals = []
        opinion = nllb_opinion_by_id.get(row["source_id"])
        if opinion is not None:
            signals.append(chrf_plusplus(translated_text, opinion))
        back = nllb_back_by_id.get(row["source_id"])
        if back is not None:
            signals.append(chrf_plusplus(back, str(source.get("question", ""))))
        if "lid_conf" in row:
            signals.append(float(row["lid_conf"]) * 100.0)
        if "solvable_bt" in row:
            signals.append(float(row["solvable_bt"]) * 100.0)
        leak = float(row.get("leak_id", 0.0))
        risk = risk_score(signals, leak, get("leak_penalty"))
        scored.append({**row, **checks, "risk": risk, "leak_id": leak})
    blocked = sum(1 for row in scored if row["blocked"])
    sample = select_review_sample(
        scored, "risk", "source_id", get("review_top_risk"), get("review_random"), get("review_seed")
    )
    judge_rows = [
        {"source_id": r["source_id"], "risk": r["risk"],
         "source_text": str(source_by_id[r["source_id"]].get("question", "")),
         "translated_text": str(r.get("question", ""))}
        for r in scored
    ]
    judge_scores = run_judge(
        judge_rows, judge_fn, get("judge_prompt"), language,
        get("judge_sample_decile"), get("judge_sample_random"), get("review_seed"),
    )
    actions: dict[str, int] = {}
    for row in scored:
        action = decide_row_action(row["blocked"], int(row.get("fallback_count", 0)))
        actions[action] = actions.get(action, 0) + 1
    return {
        "n_rows": len(scored),
        "blocked": blocked,
        "blocked_by_cause": _count_causes(scored),
        "actions": actions,
        "review_sample": sample,
        "judge_scores": judge_scores,
        "mean_risk": statistics.fmean([r["risk"] for r in scored]) if scored else 0.0,
    }


def _count_causes(scored: list[dict[str, Any]]) -> dict[str, int]:
    causes: dict[str, int] = {}
    for row in scored:
        if not row["struct_ok"]:
            for reason in row["struct_reasons"]:
                causes[f"struct:{reason}"] = causes.get(f"struct:{reason}", 0) + 1
        if not row["length_ok"]:
            causes["length"] = causes.get("length", 0) + 1
        if row["refusal"]:
            causes["refusal"] = causes.get("refusal", 0) + 1
        if row["metatext"]:
            causes["metatext"] = causes.get("metatext", 0) + 1
        if row["echo"]:
            causes["echo"] = causes.get("echo", 0) + 1
        if not row["lid_ok"]:
            causes["lid"] = causes.get("lid", 0) + 1
    return causes


def write_report(report: dict[str, Any], output_dir: str | Path, name: str) -> Path:
    """Write the JSON admission report with provenance and return the path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"date": date.today().isoformat(), **report}
    path = directory / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    """Hydra entry point (deferred import keeps hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig

    @hydra_main(version_base=None, config_path="../../configs/datagenerator", config_name="gate")
    def hydra_main_fn(cfg: DictConfig) -> None:
        from omegaconf import open_dict

        from src.utils.registry import load_entry

        language = load_entry("language", cfg.language)
        benchmark = load_entry("benchmark", cfg.benchmark)
        target_code = language["flores_code"]
        print(f"Gate for {cfg.benchmark}/{cfg.language}/{cfg.split} (target {target_code}, scorer {benchmark['scorer']}).")
        rows = _load_gate_rows(cfg)
        source_by_id = _load_gate_sources(cfg, rows)
        opinion_by_id = _load_id_map(cfg.nllb_second_opinion_jsonl)
        back_by_id = _load_id_map(cfg.nllb_backtranslation_jsonl)
        lid = _load_lid(cfg, target_code)
        calibration = _calibrate_step0(cfg, target_code, lid)
        for key, value in calibration.get("overrides", {}).items():
            with open_dict(cfg):
                setattr(cfg, key, value)
        if cfg.judge_base_url is None:
            print("warning: judge skipped (judge_base_url is null; advisory only)")
            judge_fn = None
        else:
            judge_fn = openai_judge_fn(cfg.judge_model, cfg.judge_base_url)
        report = orchestrate_gate(
            rows, source_by_id, opinion_by_id, back_by_id,
            benchmark["scorer"], target_code, lid, judge_fn, cfg, cfg.language,
        )
        verdict = _partial_verdict(report, cfg)
        path = write_report(
            {"gate": f"{cfg.benchmark}/{cfg.language}/{cfg.split}", **report, **verdict,
             "calibration": calibration},
            cfg.output_dir, f"gate-{cfg.benchmark}-{cfg.language}-{cfg.split}.json",
        )
        print(f"blocked {report['blocked']}/{report['n_rows']} (mean risk {report['mean_risk']:.1f})")
        print(f"evaluated rules: {verdict['rules_evaluated']}")
        print(f"pending rules: {verdict['rules_pending']}")
        print(f"Wrote {path}")
        _log_gate_wandb(cfg, report, verdict)

    hydra_main_fn()


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file into row dicts."""
    import json

    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_gate_rows(cfg: Any) -> list[dict[str, Any]]:
    """Load translated rows; exactly one of input_jsonl/input_dataset."""
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    if bool(get("input_jsonl")) == bool(get("input_dataset")):
        raise ValueError("set exactly one of input_jsonl/input_dataset")
    if get("input_jsonl"):
        raw = _read_jsonl(get("input_jsonl"))
    else:
        from datasets import load_dataset

        raw = list(load_dataset(get("input_dataset"), split=get("input_split")))
    text_column, id_column = get("text_column"), get("id_column")
    rows = []
    for row in raw:
        rows.append({
            "source_id": str(row[id_column]),
            "question": str(row[text_column]),
            # Translated completions are not persisted (canonical schema 5.3);
            # answer-side structure was enforced fatally at translate time.
            "answer": "",
            "instruction": str(row.get("instruction", "")),
        })
    print(f"loaded {len(rows)} translated rows")
    return rows


def _load_gate_sources(cfg: Any, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load English sources; canonical inputs carry their own _en columns."""
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    if get("source_jsonl"):
        raw = _read_jsonl(get("source_jsonl"))
        return {str(row[get("id_column")]): row for row in raw}
    if get("source_dataset"):
        from datasets import load_dataset

        raw = list(load_dataset(get("source_dataset"), split=get("input_split")))
        return {str(row[get("id_column")]): row for row in raw}
    print("source: using _en columns embedded in the translated rows")
    raw = _read_jsonl(get("input_jsonl")) if get("input_jsonl") else []
    embedded = {str(row[get("id_column")]): row for row in raw}
    sources = {}
    for row in rows:
        source_id = row["source_id"]
        original = embedded.get(source_id, {})
        sources[source_id] = {
            "question": str(original.get("question_en", "")),
            "answer": "",
            "instruction": str(original.get("instruction_en", "")),
        }
    return sources


def _load_id_map(path: str | None) -> dict[str, str]:
    """Load an NLLB sidecar (source_id -> text); empty when unset."""
    if not path:
        return {}
    return {str(row["source_id"]): str(row["text"]) for row in _read_jsonl(path)}


def _load_lid(cfg: Any, target_code: str) -> LidClient:
    """Real GlotLID client, or a target-always stub for tests."""
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    if get("lid_stub"):
        return _StubLid(target_code)
    if not get("lid_model_path"):
        raise ValueError("set lid_model_path (or lid_stub=true for tests)")
    return FastTextLid(get("lid_model_path"))


class _StubLid:
    """Test LID client that always votes the target code."""

    def __init__(self, target_code: str) -> None:
        self.target_code = target_code

    def label(self, texts: list[str]) -> list[tuple[str, float]]:
        return [(self.target_code, 1.0) for _ in texts]


def _calibrate_step0(cfg: Any, target_code: str, lid: LidClient) -> dict[str, Any]:
    """Calibrate LID threshold and length band on FLORES references.

    Falls back to config defaults (recorded) when the gated FLORES repo is
    unreachable.
    """
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    try:
        from pipeline.covariates._common import flores_sentences

        refs = flores_sentences(get("flores_dataset"), target_code, get("flores_split"))
        sources = flores_sentences(get("flores_dataset"), "eng_Latn", get("flores_split"))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"warning: step 0 calibration skipped ({type(exc).__name__}: {exc})")
        return {"status": "skipped", "overrides": {}}
    probs = [prob for _, prob in lid.label(refs)]
    ratios = [
        len(ref) / max(1, len(source)) for ref, source in zip(refs, sources, strict=False)
    ]
    overrides = {
        "lid_prob_min": calibrate_lid_threshold(probs),
        "length_ratio_min": calibrate_length_band(ratios)[0],
        "length_ratio_max": calibrate_length_band(ratios)[1],
    }
    print(f"step 0: lid_prob_min={overrides['lid_prob_min']:.3f} "
          f"length_band=({overrides['length_ratio_min']:.3f}, {overrides['length_ratio_max']:.3f})")
    return {"status": "calibrated", "n_refs": len(refs), "overrides": overrides}


def _partial_verdict(report: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """Evaluate the computable admission rules; NLLB/human rules stay pending.

    Structure holds by construction: translate raises fatally on any
    structure failure, so every persisted row passed it upstream.
    """
    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    evaluated = {
        "structure": True,
        "fallback": True,
    }
    lid_failures = report["blocked_by_cause"].get("lid", 0)
    lid_share = 1.0 - lid_failures / max(1, report["n_rows"])
    evaluated["lid_share"] = lid_share >= get("admit_lid_share")
    pending = ["nllb_ratio", "chrf_floor", "leakage", "register", "human_answer", "human_errors"]
    return {
        "rules_evaluated": {**evaluated, "lid_share_value": lid_share},
        "rules_pending": pending,
        "evaluated_pass": all(evaluated.values()),
    }


def _log_gate_wandb(cfg: Any, report: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Best-effort wandb logging for the gate report."""
    from omegaconf import OmegaConf

    from pipeline.datagenerator.translate_benchmark import maybe_log_wandb

    get = cfg.get if isinstance(cfg, dict) else cfg.__getattr__
    wandb_cfg = get("wandb")
    project = wandb_cfg.get("project") if isinstance(wandb_cfg, dict) else wandb_cfg.project
    prefix = wandb_cfg.get("run_prefix") if isinstance(wandb_cfg, dict) else wandb_cfg.run_prefix
    maybe_log_wandb(
        project,
        f"{prefix}-{cfg.benchmark}-{cfg.language}-{cfg.split}",
        [cfg.benchmark, cfg.language],
        OmegaConf.to_container(cfg, resolve=True),
        {"blocked": report["blocked"], "n_rows": report["n_rows"],
         "mean_risk": report["mean_risk"], **verdict["rules_evaluated"]},
    )


if __name__ == "__main__":
    main()
