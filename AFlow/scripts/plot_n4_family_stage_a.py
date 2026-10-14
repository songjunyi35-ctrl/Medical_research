"""Plot saved preregistered family comparisons; never run or select a model."""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aflow_matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ("No additional N4", "2026-09-05_no_n4_clean_teacher_replication", "#1976a3", -0.11),
    ("N4 with Otsu fit mask", "2026-09-05_n4_otsu_clean_replication", "#cf7130", 0.11),
)


def main():
    reports = []
    for name, directory, color, offset in METHODS:
        path = ROOT / "archive/research_records" / directory / "RAW_INNER.json"
        if not path.is_file():
            raise FileNotFoundError("All frozen family readouts must finish before plotting")
        reports.append((name, json.loads(path.read_text()), color, offset))
    output = ROOT / "archive/reviews/2026-09-05_30h_chain_audit"
    destinations = [output / ("N4_FAMILY_STAGE_A_COMPARISON." + extension) for extension in ("pdf", "png")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserve existing result figures")
    positions = [(fold, style) for fold in ("e", "f") for style in ("identity", "flip8")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.7), sharey=True, constrained_layout=True)
    for axis, metric, title in zip(axes, ("scan_dice", "subject_dice", "point_hits"),
                                   ("Scan Dice difference", "Subject Dice difference", "Point-hit count difference")):
        axis.axvline(0, color="#444444", linewidth=1)
        axis.axvline(-1 if metric == "point_hits" else -0.003, color="#999999", linestyle="--", linewidth=1)
        for name, report, color, offset in reports:
            for index, (fold, style) in enumerate(positions):
                delta = report["comparisons"][fold][style]["delta"]
                value = delta[metric]
                y = index + offset
                if metric != "point_hits":
                    low, high = delta[metric + "_95ci"]
                    axis.hlines(y, low, high, color=color, linewidth=2)
                axis.plot(value, y, "o", color=color, markersize=5,
                          label=name if index == 0 else None)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Candidate minus shared canonical control")
        axis.grid(axis="x", alpha=0.18)
    axes[0].set_yticks(range(4), ["E / identity 1x", "E / fixed Flip8", "F / identity 1x", "F / fixed Flip8"])
    axes[0].invert_yaxis()
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Post-test development: E = 42 scans / 17 patients; F = 42 scans / 19 patients\n"
                 "Intervals: saved patient-cluster 95% bootstrap; shared controls, no independent test claim", fontsize=11)
    for path in destinations:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    print("\n".join(str(path) for path in destinations))


if __name__ == "__main__":
    main()
