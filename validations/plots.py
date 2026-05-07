"""Plotting utilities for Pareto-front and metric comparison."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "nsga2": ("NSGA-II", "steelblue", "o"),
    "spea2": ("SPEA-2", "tomato", "s"),
    "uclm_nsga2": ("UCLM NSGA-II", "seagreen", "^"),
}
ALGORITHMS = ["nsga2", "spea2", "uclm_nsga2"]
METRICS = ["hypervolume", "spread", "spacing", "time"]
LOWER_IS_BETTER = {"spacing", "time"}


def pareto_2d(
    fronts: Dict[str, list],
    title: str = "Cost/Satisfaction Pareto Front Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Cost/satisfaction scatter plot for all comparable algorithms."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, front in fronts.items():
        label, color, marker = STYLES.get(key, (key, None, "o"))
        ax.scatter(
            [_cost(s) for s in front],
            [_satisfaction(s) for s in front],
            label=label,
            alpha=0.75,
            color=color,
            marker=marker,
            s=28,
        )

    ax.set_xlabel("Cost (lower is better)")
    ax.set_ylabel("Satisfaction (higher is better)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def pareto_3d(
    fronts: Dict[str, list],
    title: str = "3-Objective Pareto Front",
    save_path: Optional[str] = None,
) -> None:
    """3-D plot for algorithms that expose fairness variance."""
    comparable = {
        key: front
        for key, front in fronts.items()
        if front and hasattr(front[0], "fairness_variance")
    }
    if not comparable:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for key, front in comparable.items():
        label, color, marker = STYLES.get(key, (key, None, "o"))
        ax.scatter(
            [_cost(s) for s in front],
            [_satisfaction(s) for s in front],
            [s.fairness_variance for s in front],
            label=label,
            alpha=0.75,
            color=color,
            marker=marker,
            s=24,
        )

    ax.set_xlabel("Cost (lower)", labelpad=8)
    ax.set_ylabel("Satisfaction (higher)", labelpad=8)
    ax.set_zlabel("Fairness variance (lower)", labelpad=8)
    ax.set_title(title)
    ax.legend()
    _save_or_show(fig, save_path)


def metric_bars(
    all_results: List[Dict[str, Any]],
    metric: str = "hypervolume",
    save_path: Optional[str] = None,
) -> None:
    datasets = [r["dataset"] for r in all_results]
    algorithms = ["nsga2", "spea2", "uclm_nsga2"]

    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.3), 5))

    offsets = np.linspace(-width, width, len(algorithms))
    for offset, key in zip(offsets, algorithms):
        label, color, _marker = STYLES[key]
        means = [r[f"{key}_summary"][f"{metric}_mean"] for r in all_results]
        stds = [r[f"{key}_summary"][f"{metric}_std"] for r in all_results]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=label,
            alpha=0.85,
            capsize=4,
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(f"{metric.replace('_', ' ').capitalize()} comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def metric_boxplots(
    all_results: List[Dict[str, Any]],
    metric: str,
    save_path: Optional[str] = None,
) -> None:
    """Show run-to-run variability for one metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    values = []
    labels = []
    colors = []
    for key in ALGORITHMS:
        label, color, _marker = STYLES[key]
        vals = [run[metric] for result in all_results for run in result[key]]
        values.append(vals)
        labels.append(label)
        colors.append(color)

    box = ax.boxplot(values, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_title(f"{metric.replace('_', ' ').capitalize()} variability across runs")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def ranking_heatmap(
    all_results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Rank algorithms per dataset and metric; rank 1 is best."""
    row_labels = []
    rows = []
    for result in all_results:
        for metric in METRICS:
            means = {
                key: result[f"{key}_summary"][f"{metric}_mean"]
                for key in ALGORITHMS
            }
            reverse = metric not in LOWER_IS_BETTER
            ordered = sorted(means, key=means.get, reverse=reverse)
            ranks = {key: ordered.index(key) + 1 for key in ALGORITHMS}
            rows.append([ranks[key] for key in ALGORITHMS])
            row_labels.append(f"{result['dataset']} / {metric}")

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.42)))
    mat = np.array(rows)
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=1, vmax=len(ALGORITHMS))

    ax.set_xticks(np.arange(len(ALGORITHMS)))
    ax.set_xticklabels([STYLES[key][0] for key in ALGORITHMS], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Algorithm rank by dataset and metric")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Rank (1 is best)")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def hv_time_scatter(
    all_results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Plot quality/runtime tradeoff using per-run values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ALGORITHMS:
        label, color, marker = STYLES[key]
        times = [run["time"] for result in all_results for run in result[key]]
        hvs = [run["hypervolume"] for result in all_results for run in result[key]]
        ax.scatter(times, hvs, label=label, color=color, marker=marker, alpha=0.75, s=36)

    ax.set_xlabel("Runtime in seconds (lower is better)")
    ax.set_ylabel("Hypervolume (higher is better)")
    ax.set_title("Quality vs runtime tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def normalized_radar(
    all_results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Radar chart of average normalized presentation metrics."""
    axes_metrics = ["hypervolume", "spread", "spacing", "time"]
    scores = {key: [] for key in ALGORITHMS}

    for metric in axes_metrics:
        means = {
            key: float(np.mean([r[f"{key}_summary"][f"{metric}_mean"] for r in all_results]))
            for key in ALGORITHMS
        }
        vals = np.array(list(means.values()), dtype=float)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        denom = hi - lo or 1.0
        for key in ALGORITHMS:
            raw = (means[key] - lo) / denom
            scores[key].append(1.0 - raw if metric in LOWER_IS_BETTER else raw)

    labels = ["Hypervolume", "Spread", "Spacing", "Runtime"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for key in ALGORITHMS:
        label, color, _marker = STYLES[key]
        vals = scores[key] + scores[key][:1]
        ax.plot(angles, vals, label=label, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Normalized overall profile")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def win_count_bars(
    all_results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Count how often each algorithm has the best mean value."""
    wins = {key: 0 for key in ALGORITHMS}
    for result in all_results:
        for metric in METRICS:
            means = {
                key: result[f"{key}_summary"][f"{metric}_mean"]
                for key in ALGORITHMS
            }
            best = min(means, key=means.get) if metric in LOWER_IS_BETTER else max(means, key=means.get)
            wins[best] += 1

    labels = [STYLES[key][0] for key in ALGORITHMS]
    colors = [STYLES[key][1] for key in ALGORITHMS]
    values = [wins[key] for key in ALGORITHMS]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylabel("Number of metric wins")
    ax.set_title("Best-performing algorithm count")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def presentation_notes(all_results: List[Dict[str, Any]], save_path: str) -> None:
    """Write speaker notes for the generated presentation figures."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    datasets = ", ".join(result["dataset"] for result in all_results)
    lines = [
        "# Speaker Notes",
        "",
        "## Slide: Experimental Setup",
        f"- We compare three algorithms on the shared MONRP objectives: cost and satisfaction.",
        f"- Datasets included in this run: {datasets}.",
        "- Hypervolume and spread are interpreted as higher is better; spacing and runtime are lower is better.",
        "- The custom NSGA-II and SPEA-2 also optimize fairness variance, but UCLM NSGA-II is 2-objective, so the main comparison uses the shared 2D space.",
        "",
        "## Slide: 2D Pareto Front",
        "- This plot shows the tradeoff between implementation cost and stakeholder satisfaction.",
        "- Points farther toward low cost and high satisfaction are preferable.",
        "- A wider front means the algorithm gives decision makers more alternatives rather than a single answer.",
        "",
        "## Slide: 3D Pareto Front",
        "- This slide is only for the custom NSGA-II and SPEA-2, because they include the fairness objective.",
        "- The third axis shows fairness variance; lower variance means satisfaction is distributed more evenly across stakeholders.",
        "- Use this slide to explain the extension beyond the baseline UCLM formulation.",
        "",
        "## Slide: Metric Bar Charts",
        "- Hypervolume summarizes both convergence and coverage of the Pareto front.",
        "- Spread shows how much of the tradeoff surface is covered.",
        "- Spacing shows how evenly solutions are distributed; lower spacing is better.",
        "- Runtime compares computational cost under the same population, generation, and seed settings.",
        "",
        "## Slide: Metric Boxplots",
        "- Boxplots show stability across independent runs.",
        "- A high median with a compact box is desirable for hypervolume.",
        "- For spacing and runtime, lower medians and compact boxes indicate more reliable behavior.",
        "",
        "## Slide: Rank Heatmap",
        "- Each cell is the rank for one algorithm on one dataset and metric.",
        "- Rank 1 is the best for that row.",
        "- This gives a compact overview of consistency instead of focusing on a single dataset.",
        "",
        "## Slide: Hypervolume vs Runtime",
        "- This plot shows the quality-efficiency tradeoff.",
        "- The ideal region is upper-left: high hypervolume with low runtime.",
        "- It helps justify whether a quality improvement is worth the extra computational time.",
        "",
        "## Slide: Normalized Profile Radar",
        "- Each axis is normalized so the best observed algorithm scores near the outside.",
        "- This is a high-level summary, not a replacement for the raw metric charts.",
        "- Use it to explain overall strengths and weaknesses quickly.",
        "",
        "## Slide: Win Count",
        "- This counts how often each algorithm has the best mean metric value across datasets.",
        "- It is useful as a closing summary, but should be interpreted together with the metric magnitudes.",
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _cost(solution) -> float:
    return float(solution.cost if hasattr(solution, "cost") else solution.total_cost)


def _satisfaction(solution) -> float:
    return float(
        solution.satisfaction
        if hasattr(solution, "satisfaction")
        else solution.total_satisfaction
    )


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
