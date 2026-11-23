"""TOST equivalence tests for the primary contrasts (issue #64, 10.5).

Each primary contrast ends as different, equivalent within the bound, or
undetermined. The bound comes from the series file
(analysis.equivalence_bound_points), never from a literal.
"""

from __future__ import annotations

import statistics
from typing import Any

from scipy import stats as scipy_stats

from src.analysis.inference import holm_adjusted


def tost_verdict(
    lo90: float, hi90: float, lo95: float, hi95: float, bound: float
) -> str:
    """Verdict from the 90% and 95% intervals against the bound (10.5)."""
    if lo90 >= -bound and hi90 <= bound:
        return "equivalent"
    if lo95 > 0 or hi95 < 0:
        return "different"
    return "undetermined"


def tost_from_bootstrap(draws: list[float], bound: float) -> dict[str, Any]:
    """TOST verdict, one-sided p-values and intervals from bootstrap draws."""
    ordered = sorted(draws)
    n = len(ordered)
    p_lower = sum(1 for draw in draws if draw <= -bound) / n
    p_upper = sum(1 for draw in draws if draw >= bound) / n
    result = {
        "estimate": statistics.fmean(draws),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "tost_p": max(p_lower, p_upper),
        "lo90": ordered[int(0.05 * n)],
        "hi90": ordered[min(n - 1, int(0.95 * n))],
        "lo95": ordered[int(0.025 * n)],
        "hi95": ordered[min(n - 1, int(0.975 * n))],
    }
    result["verdict"] = tost_verdict(
        result["lo90"], result["hi90"], result["lo95"], result["hi95"], bound
    )
    return result


def tost_from_contrast(estimate: float, se: float, bound: float) -> dict[str, Any]:
    """TOST verdict from a mixed-model contrast estimate and standard error."""
    if se <= 0:
        raise ValueError("standard error must be positive")
    p_lower = float(scipy_stats.norm.sf((estimate + bound) / se))
    p_upper = float(scipy_stats.norm.cdf((estimate - bound) / se))
    result = {
        "estimate": estimate,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "tost_p": max(p_lower, p_upper),
        "lo90": estimate - 1.645 * se,
        "hi90": estimate + 1.645 * se,
        "lo95": estimate - 1.96 * se,
        "hi95": estimate + 1.96 * se,
    }
    result["verdict"] = tost_verdict(
        result["lo90"], result["hi90"], result["lo95"], result["hi95"], bound
    )
    return result


def load_equivalence_bound(series_name: str = "s01_language_medium") -> tuple[float, float]:
    """Bound and target detectable difference from the series file."""
    from src.utils.registry import load_entry

    analysis = load_entry("series", series_name)["analysis"]
    return float(analysis["equivalence_bound_points"]), float(analysis["target_mde_points"])


def summarize_primary_family(
    contrasts: list[dict[str, Any]], bound: float
) -> list[dict[str, Any]]:
    """Per benchmark+language table: estimate, intervals, both verdicts.

    Each contrast carries estimate, lo95, hi95, lo90, hi90 and the difference
    p-value; Holm adjusts the family and the equivalence verdict from 10.5
    sits next to the difference verdict.
    """
    adjusted = holm_adjusted([contrast["p_difference"] for contrast in contrasts])
    table = []
    for contrast, p_holm in zip(contrasts, adjusted, strict=True):
        table.append({
            "benchmark": contrast["benchmark"],
            "language": contrast["language"],
            "contrast": contrast.get("kind", ""),
            "estimate": contrast["estimate"],
            "lo95": contrast["lo95"],
            "hi95": contrast["hi95"],
            "p_holm": p_holm,
            "difference": "different" if p_holm < 0.05 else "not_different",
            "equivalence": tost_verdict(
                contrast["lo90"], contrast["hi90"],
                contrast["lo95"], contrast["hi95"], bound,
            ),
        })
    return table
