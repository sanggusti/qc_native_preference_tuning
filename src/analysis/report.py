"""Report tables, varied figures, and the identification statement (issue #56).

Tables follow methodology section 13 in order. Figures use matplotlib (Agg):
each estimand family gets its own chart type (forest, scatter, dumbbell,
heatmap, lollipop, strip) rather than one repeated bar chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

TABLE_ORDER = [
    "covariates_gate",
    "platform_check",
    "practitioner",
    "cross_language",
    "identification",
    "exposure_plots",
    "identification_statement",
]


def render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a GitHub markdown table with fixed column order."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def forest_plot(
    labels: list[str],
    estimates: list[float],
    lower: list[float],
    upper: list[float],
    path: str | Path,
    title: str = "Contrasts with 95% intervals (accuracy points)",
    equivalence_bound: float | None = None,
) -> Path:
    """Horizontal forest plot with a zero line and optional equivalence band."""
    order = sorted(range(len(labels)), key=lambda i: estimates[i])
    labels = [labels[i] for i in order]
    estimates = [estimates[i] for i in order]
    lower = [lower[i] for i in order]
    upper = [upper[i] for i in order]
    positions = list(range(len(labels)))
    _, ax = plt.subplots(figsize=(7, max(2.5, 0.55 * len(labels) + 1)))
    if equivalence_bound is not None:
        ax.axvspan(-equivalence_bound, equivalence_bound, color="green", alpha=0.08, label="equivalence bound")
    ax.errorbar(
        estimates, positions,
        xerr=[np.subtract(estimates, lower), np.subtract(upper, estimates)],
        fmt="o", capsize=3, color="black", ecolor="#444444",
    )
    ax.axvline(0, color="red", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("accuracy points")
    ax.set_title(title)
    if equivalence_bound is not None:
        ax.legend(loc="lower right", fontsize=8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def exposure_scatter(
    points: list[dict[str, Any]],
    x_key: str,
    path: str | Path,
    title: str = "Accuracy against exposure",
) -> Path:
    """Scatter of accuracy vs exposure; size is fertility, color is quality."""
    xs = [point[x_key] for point in points]
    ys = [point["accuracy"] for point in points]
    max_fertility = max([point.get("fertility", 1.0) for point in points] or [1.0])
    sizes = [30 + 300 * point.get("fertility", 1.0) / max_fertility for point in points]
    colors = [point.get("quality", 0.5) for point in points]
    _, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.8, edgecolors="black")
    for point in points:
        ax.annotate(str(point["label"]), (point[x_key], point["accuracy"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(x_key)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label="translation quality")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def dumbbell_plot(
    labels: list[str],
    before: list[float],
    after: list[float],
    path: str | Path,
    title: str = "Base vs native accuracy per language",
    before_label: str = "base",
    after_label: str = "native",
) -> Path:
    """Paired dumbbell lines connecting before/after accuracy per language."""
    positions = list(range(len(labels)))
    _, ax = plt.subplots(figsize=(6.5, max(2.5, 0.55 * len(labels) + 1)))
    for position, low, high in zip(positions, before, after, strict=True):
        ax.plot([low, high], [position, position], color="#888888", linewidth=2, zorder=1)
    ax.scatter(before, positions, color="#d55e00", s=60, label=before_label, zorder=2)
    ax.scatter(after, positions, color="#0072b2", s=60, label=after_label, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def accuracy_heatmap(
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float]],
    path: str | Path,
    title: str = "Accuracy by condition and language",
) -> Path:
    """Annotated heatmap of accuracy over conditions by languages."""
    data = np.asarray(values, dtype=float)
    _, ax = plt.subplots(figsize=(max(5, 0.9 * len(col_labels) + 2), max(3, 0.6 * len(row_labels) + 1)))
    image = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row, _ in enumerate(row_labels):
        for col, _ in enumerate(col_labels):
            ax.text(col, row, f"{data[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    plt.colorbar(image, ax=ax, label="accuracy")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def lollipop_plot(
    labels: list[str],
    gaps: list[float],
    bounds: list[float],
    path: str | Path,
    title: str = "Native gap vs round-trip bound per language",
) -> Path:
    """Lollipops for the native gap with the translator-loss bound marked."""
    positions = list(range(len(labels)))
    _, ax = plt.subplots(figsize=(6.5, max(2.5, 0.55 * len(labels) + 1)))
    ax.hlines(positions, [0] * len(labels), gaps, color="#0072b2", linewidth=2)
    ax.plot(gaps, positions, "o", color="#0072b2", markersize=7, label="native gap A(en) - A(L)")
    ax.plot(bounds, positions, "x", color="#d55e00", markersize=8, label="round-trip bound d(L)")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("accuracy points")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def replicate_strip(
    labels: list[str],
    replicates: list[list[float]],
    path: str | Path,
    title: str = "Replicate spread per cell (shows s_r)",
) -> Path:
    """Strip of per-replicate cell means with the cell mean marked."""
    _, ax = plt.subplots(figsize=(6.5, max(2.5, 0.55 * len(labels) + 1)))
    rng = np.random.default_rng(47)
    for position, (_, values) in enumerate(zip(labels, replicates, strict=True)):
        jitter = rng.normal(0, 0.06, len(values))
        ax.scatter(values, np.full(len(values), position) + jitter, color="#0072b2", s=40, alpha=0.7)
        ax.plot([float(np.mean(values))], [position], marker="|", color="black", markersize=18, markeredgewidth=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("cell accuracy")
    ax.set_title(title)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def identification_statement(results: dict[str, Any]) -> str:
    """One paragraph stating what is and is not identified, from verdicts."""
    parts = [
        "Within-language medium contrasts (E1-E4) are identified as causal effects of a "
        "manipulable choice, paired on items and replicated over runs.",
        f"Primary verdicts: {results.get('primary_summary', 'pending')}.",
        f"E7 base-model replication: {results.get('e7', 'pending')}.",
        f"E8 covariate adjustment: {results.get('e8', 'pending')}.",
        f"Contamination (E9 difference-in-differences): {results.get('e9', 'pending')}.",
        "Cross-language levels (E6) are descriptive of this base model, translator and "
        "tokenizer, not effects of language structure; E10 structural claims are retired.",
    ]
    return " ".join(parts)
