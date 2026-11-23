"""Estimation for S01: intervals, contrasts, corrections, rules (issue #56).

Intervals and assumption-light checks are exact finite-sample procedures.
The logistic mixed model is a Laplace-approximated GLMM with the
pre-registered random structure (item, item-by-language, run), fit by
penalized block-coordinate PIRLS in numpy: statsmodels' BayesMixedGLM does
not converge on this crossed structure even on clean synthetic data, and R
is not available here, so the pre-registered fallback (drop v_il on
non-convergence) is implemented around a fitter that does converge.
Corrections use statsmodels.multitest; McNemar uses scipy.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def item_bootstrap(
    values: list[float], n_draws: int = 10000, seed: int = 47, ci: float = 0.95
) -> dict[str, float]:
    """Percentile bootstrap CI over items."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choice(values) for _ in range(n)) for _ in range(n_draws))
    tail = (1 - ci) / 2
    return {
        "estimate": statistics.fmean(values),
        "lo": means[int(tail * n_draws)],
        "hi": means[min(n_draws - 1, int((1 - tail) * n_draws))],
    }


def two_level_bootstrap(
    runs_a: list[list[float]],
    runs_b: list[list[float]],
    n_draws: int = 10000,
    seed: int = 47,
) -> dict[str, float]:
    """Contrast CI resampling runs, then items within runs (10.2)."""
    rng = random.Random(seed)

    def cell_mean(runs: list[list[float]]) -> float:
        sampled_runs = [rng.choice(runs) for _ in range(len(runs))]
        pooled = [rng.choice(run) for run in sampled_runs for _ in range(len(run))]
        return statistics.fmean(pooled)

    diffs = sorted(cell_mean(runs_a) - cell_mean(runs_b) for _ in range(n_draws))
    frac_le_zero = sum(1 for d in diffs if d <= 0) / n_draws
    frac_ge_zero = sum(1 for d in diffs if d >= 0) / n_draws
    return {
        "estimate": statistics.fmean([v for run in runs_a for v in run])
        - statistics.fmean([v for run in runs_b for v in run]),
        "lo95": diffs[int(0.025 * n_draws)],
        "hi95": diffs[min(n_draws - 1, int(0.975 * n_draws))],
        "lo90": diffs[int(0.05 * n_draws)],
        "hi90": diffs[min(n_draws - 1, int(0.95 * n_draws))],
        "p_two_sided": min(1.0, 2 * min(frac_le_zero, frac_ge_zero)),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant counts (single runs only)."""
    if b + c == 0:
        return 1.0
    return float(scipy_stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def holm_adjusted(p_values: list[float]) -> list[float]:
    """Holm adjusted p-values for the primary family."""
    return list(multipletests(p_values, method="holm")[1])


def bh_adjusted(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values for exploratory contrasts."""
    return list(multipletests(p_values, method="fdr_bh")[1])


def kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall tau-b for two orderings (E7 rank agreement)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("kendall needs two equal-length lists of length >= 2")
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if dx == 0:
                    tied_x += 1
                if dy == 0:
                    tied_y += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    )
    if denominator == 0:
        return 1.0 if discordant == 0 else -1.0
    return (concordant - discordant) / denominator


def mde_paired(q: float, n: int) -> float:
    """Minimum detectable difference in points for paired items (10.5)."""
    return 2.8 * math.sqrt(q / n) * 100


def mde_with_replicates(s_r_points: float, k: int, q: float, n: int) -> float:
    """MDE in points including the run-to-run variance term (10.5)."""
    s_r = s_r_points / 100
    return 2.8 * math.sqrt(2 * s_r * s_r / k + q / n) * 100


def replicate_k_needed(s_r_points: float, q: float, n: int, target: float = 0.05) -> int | None:
    """Smallest k meeting the 5-point target, or None when impossible."""
    denominator = (target / 2.8) ** 2 - q / n
    if denominator <= 0:
        return None
    needed = math.ceil(2 * (s_r_points / 100) ** 2 / denominator)
    return max(3, needed)


def replicate_rule(
    s_r: float, s_r_lo: float, s_r_hi: float, q: float, n: int
) -> dict[str, Any]:
    """Apply the 10.5 replicate rule; pause when s_r exceeds 2.5."""
    needed = replicate_k_needed(s_r_hi, q, n)
    if needed is None or needed > 8 or s_r_hi > 2.5:
        return {"k": None, "paused": True, "reason": "s_r above 2.5: platform not deterministic enough"}
    return {"k": needed, "paused": False, "reason": f"k={needed} covers s_r interval upper bound {s_r_hi:.3f}"}


def e7_verdict(
    gaps: list[dict[str, Any]], contrast_signs: dict[str, float], tau_threshold: float = 0.733
) -> dict[str, Any]:
    """E7 sign and rank reading over qualifying gaps (10.4)."""
    qualifying = [
        gap for gap in gaps
        if gap["interval_excludes_zero"] and gap["above_floor_both"]
    ]
    if len(qualifying) < 3:
        return {"verdict": "not testable at this precision", "n_qualifying": len(qualifying)}
    signs_agree = all(
        (gap["estimate"] > 0) == (contrast_signs[gap["label"]] > 0) for gap in qualifying
    )
    tau = kendall_tau(
        [gap["estimate"] for gap in qualifying],
        [contrast_signs[gap["label"]] for gap in qualifying],
    )
    verdict = "exposure-consistent" if signs_agree else "model-specific"
    return {
        "verdict": verdict,
        "n_qualifying": len(qualifying),
        "signs_agree": signs_agree,
        "kendall_tau": tau,
        "rank_agrees": tau >= tau_threshold,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def build_design(
    df: pd.DataFrame,
    ref_language: str,
    ref_condition: str,
    covariates: tuple[str, ...] = (),
    include_base: bool = False,
    ref_base: str = "primary",
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Fixed design (saturated language x condition) plus random-effect codes.

    Columns: intercept, language dummies, condition dummies, their products,
    then optional standardized covariates and base terms. Groups: item,
    item-by-language, run.
    """
    languages = sorted(set(df["language"]) - {ref_language})
    conditions = sorted(set(df["condition"]) - {ref_condition})
    columns: dict[str, np.ndarray] = {"intercept": np.ones(len(df))}
    for lang in languages:
        columns[f"lang[{lang}]"] = (df["language"].to_numpy() == lang).astype(float)
    for cond in conditions:
        columns[f"cond[{cond}]"] = (df["condition"].to_numpy() == cond).astype(float)
    for lang in languages:
        for cond in conditions:
            columns[f"lang[{lang}]:cond[{cond}]"] = (
                (df["language"].to_numpy() == lang) & (df["condition"].to_numpy() == cond)
            ).astype(float)
    scales: dict[str, tuple[float, float]] = {}
    for covariate in covariates:
        values = df[covariate].to_numpy(dtype=float)
        mean, sd = float(values.mean()), float(values.std()) or 1.0
        scales[covariate] = (mean, sd)
        columns[f"cov[{covariate}]"] = (values - mean) / sd
    if include_base:
        bases = sorted(set(df["base"]) - {ref_base})
        for base in bases:
            columns[f"base[{base}]"] = (df["base"].to_numpy() == base).astype(float)
        for lang in sorted(set(df["language"])):
            for base in bases:
                columns[f"lang[{lang}]:base[{base}]"] = (
                    (df["language"].to_numpy() == lang) & (df["base"].to_numpy() == base)
                ).astype(float)
    names = [name for name in columns if name == "intercept" or np.std(columns[name]) > 0]
    columns = {name: columns[name] for name in names}
    groups = {
        name: pd.factorize(df[key].astype(str))[0].astype(int)
        for name, key in (
            ("item", "item"),
            ("itemlang", "itemlang"),
            ("run", "run"),
        )
    }
    return np.column_stack([columns[name] for name in names]), names, groups


def fit_glmm(
    y: np.ndarray,
    design: np.ndarray,
    groups: dict[str, np.ndarray],
    max_sweeps: int = 100,
    tol: float = 1e-6,
    ridge: float = 1e-4,
) -> dict[str, Any]:
    """Penalized block-coordinate PIRLS for logistic random intercepts.

    Fixed effects by Newton steps on a p-by-p system; each random-effect
    grouping by vectorized 1D Newton sweeps; variances by moment updates.
    Standard errors come from the fixed block of the penalized Hessian at
    convergence (random effects treated as offsets).
    """
    n, n_params = design.shape
    beta = np.zeros(n_params)
    random_effects = {name: np.zeros(int(codes.max()) + 1) for name, codes in groups.items()}
    variances = dict.fromkeys(groups, 0.5)
    identity = np.eye(n_params)
    deviance = float("inf")
    converged = False
    for _ in range(max_sweeps):
        eta = design @ beta + sum(
            random_effects[name][codes] for name, codes in groups.items()
        )
        mu = _sigmoid(eta)
        weights = np.maximum(mu * (1 - mu), 1e-8)
        working = eta + (y - mu) / weights
        offset = sum(random_effects[name][codes] for name, codes in groups.items())
        lhs = design.T @ (weights[:, None] * design) + ridge * identity
        rhs = design.T @ (weights * (working - offset))
        beta = np.linalg.solve(lhs, rhs)
        for name, codes in groups.items():
            levels = random_effects[name]
            for _ in range(3):
                eta = design @ beta + sum(
                    random_effects[other][other_codes]
                    for other, other_codes in groups.items()
                )
                mu = _sigmoid(eta)
                weights = np.maximum(mu * (1 - mu), 1e-8)
                sum_score = np.bincount(codes, weights=(y - mu), minlength=len(levels))
                sum_w = np.bincount(codes, weights=weights, minlength=len(levels))
                levels += (sum_score - levels / variances[name]) / (sum_w + 1 / variances[name])
            eta = design @ beta + sum(
                random_effects[other][other_codes] for other, other_codes in groups.items()
            )
            mu = _sigmoid(eta)
            weights = np.maximum(mu * (1 - mu), 1e-8)
            sum_w = np.bincount(codes, weights=weights, minlength=len(levels))
            posterior_var = 1.0 / (sum_w + 1.0 / variances[name])
            variances[name] = max(float(np.mean(levels * levels + posterior_var)), 1e-8)
        eta = design @ beta + sum(
            random_effects[name][codes] for name, codes in groups.items()
        )
        mu = np.clip(_sigmoid(eta), 1e-12, 1 - 1e-12)
        new_deviance = float(-2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
        if abs(new_deviance - deviance) < tol * max(1.0, abs(new_deviance)):
            deviance = new_deviance
            converged = True
            break
        deviance = new_deviance
    weights = np.maximum(_sigmoid(design @ beta) * (1 - _sigmoid(design @ beta)), 1e-8)
    cov = np.linalg.inv(design.T @ (weights[:, None] * design) + ridge * identity)
    return {
        "beta": beta,
        "se": np.sqrt(np.diag(cov)),
        "cov": cov,
        "variances": variances,
        "converged": converged,
        "deviance": deviance,
    }


def fit_series_mixed_model(
    df: pd.DataFrame,
    ref_language: str,
    ref_condition: str,
    covariates: tuple[str, ...] = (),
    include_base: bool = False,
) -> dict[str, Any]:
    """Fit the pre-registered structure, dropping v_il once on non-convergence."""
    y = df["correct"].to_numpy(dtype=float)
    design, names, groups = build_design(df, ref_language, ref_condition, covariates, include_base)
    fit = fit_glmm(y, design, groups)
    fit["names"] = names
    fit["dropped"] = None
    if not fit["converged"]:
        reduced = {name: codes for name, codes in groups.items() if name != "itemlang"}
        fit = fit_glmm(y, design, reduced)
        fit["names"] = names
        fit["dropped"] = "itemlang"
    return fit


def wald_contrast(fit: dict[str, Any], vector: np.ndarray) -> dict[str, float]:
    """Wald estimate, SE, z and two-sided p for a linear combination of beta."""
    estimate = float(vector @ fit["beta"])
    se = float(math.sqrt(max(vector @ fit["cov"] @ vector, 0.0)))
    z = estimate / se if se > 0 else 0.0
    return {"estimate": estimate, "se": se, "z": z, "p": float(2 * scipy_stats.norm.sf(abs(z)))}


def cell_coefficients(
    names: list[str], language: str, condition: str, ref_language: str, ref_condition: str
) -> np.ndarray:
    """Coefficient vector for one language-by-condition cell mean."""
    vector = np.zeros(len(names))
    index = {name: position for position, name in enumerate(names)}

    def add(term: str) -> None:
        if term in index:
            vector[index[term]] = 1.0

    add("intercept")
    if language != ref_language:
        add(f"lang[{language}]")
    if condition != ref_condition:
        add(f"cond[{condition}]")
    if language != ref_language and condition != ref_condition:
        add(f"lang[{language}]:cond[{condition}]")
    return vector


def marginal_cell_probability(fit: dict[str, Any], design_rows: np.ndarray) -> float:
    """Mean predicted probability over rows at zero random effects."""
    return float(np.mean(_sigmoid(design_rows @ np.asarray(fit["beta"]))))


def share_absorbed(unadjusted: float, adjusted: float) -> float:
    """Share of a gap absorbed by covariate adjustment (E8)."""
    if unadjusted == 0:
        return 0.0
    return 1.0 - adjusted / unadjusted
