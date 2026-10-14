#!/usr/bin/env python3
"""Build the manuscript figures from frozen, explicitly listed values."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def _box(ax, xy, width, height, title, body, color):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + 0.025, y + height - 0.055, title, fontsize=11, weight="bold", color=color)
    ax.text(x + 0.025, y + height - 0.105, body, fontsize=8.4, va="top", linespacing=1.35)


def _arrow(ax, start, end, color="#334155"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def framework_figure():
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.025, 0.61), 0.19, 0.27, "Frozen contract", "Split and target\nCheckpoint and objective\nOperator set and budget\nImmutable hashes", "#1d4ed8")
    _box(ax, (0.28, 0.61), 0.19, 0.27, "MedTAWS", "UCB-style parent draw\nOne workflow modification\nFail-closed execution\nReceipt and tree update", "#0f766e")
    _box(ax, (0.535, 0.61), 0.19, 0.27, "Workflow ablation", "MedTAWS\nScore-softmax\nGreedy\nUniform", "#b45309")
    _box(ax, (0.79, 0.61), 0.18, 0.27, "Frozen workflow", "Prompt schedule\nMask recurrence\nAggregation\nExecution order", "#7c3aed")

    _arrow(ax, (0.215, 0.745), (0.28, 0.745))
    _arrow(ax, (0.47, 0.745), (0.535, 0.745))
    _arrow(ax, (0.725, 0.745), (0.79, 0.745))

    _box(ax, (0.14, 0.13), 0.21, 0.29, "Trajectory group", "Same 3D volume\nK corrective-click seeds\nThreshold masks\nGeometry-aware rewards", "#1d4ed8")
    _box(ax, (0.405, 0.13), 0.21, 0.29, "Relative alignment", "Within-case advantages\nBernoulli mask log-prob\nReference KL\nDice-CE anchor", "#0f766e")
    _box(ax, (0.67, 0.13), 0.21, 0.29, "Frozen checkpoint", "Prompt encoder adapted\nMask decoder adapted\nImage encoder frozen\nProvenance committed", "#b45309")
    _arrow(ax, (0.35, 0.275), (0.405, 0.275))
    _arrow(ax, (0.615, 0.275), (0.67, 0.275))
    _arrow(ax, (0.88, 0.275), (0.88, 0.61), color="#7c3aed")

    ax.text(0.025, 0.95, "Discrete workflow optimization", fontsize=13, weight="bold", color="#0f172a")
    ax.text(0.025, 0.49, "Continuous prompt-sensitive weight adaptation", fontsize=13, weight="bold", color="#0f172a")
    ax.text(0.5, 0.035, "Final evaluation requires base / MedTAWS / GRPO-style / combined arms under one frozen protocol", ha="center", fontsize=9.5, color="#334155")
    fig.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"framework_overview.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def result_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    names = ["Supervised", "GRPO-style"]
    means = [0.57283, 0.58932]
    colors = ["#64748b", "#0f766e"]
    bars = ax.bar(names, means, color=colors, width=0.62)
    ax.scatter([-0.13, 0.13], [0.56454, 0.58113], color="#0f172a", s=24, zorder=3)
    ax.scatter([0.87, 1.13], [0.59415, 0.58450], color="#0f172a", s=24, zorder=3)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.008,
            f"{value:.5f}",
            ha="center",
            fontsize=9,
            weight="bold",
            color="white",
        )
    ax.set_ylim(0.52, 0.62)
    ax.set_ylabel("Reload-audit Dice")
    ax.set_title("Matched 118-scan / 10-epoch comparison")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    labels = ["32/e5", "64/e5", "96/e5", "118/e5", "118/e8", "118/e10"]
    values = [0.44198, 0.51247, 0.53777, 0.54698, 0.56923, 0.58932]
    ax.plot(labels, values, marker="o", linewidth=2.2, color="#1d4ed8", markersize=6)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.007, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0.40, 0.62)
    ax.set_ylabel("Two-seed mean Dice")
    ax.set_xlabel("Training scans / epochs")
    ax.set_title("Development progression (variables not isolated)")
    ax.grid(axis="y", alpha=0.22)

    fig.suptitle("Retrospective RC development evidence: not subject-disjoint", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"development_results.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    framework_figure()
    result_figure()
