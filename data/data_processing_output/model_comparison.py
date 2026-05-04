import csv
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, '/home/jp/a_school/BP')
from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

PATH = os.path.dirname(os.path.abspath(__file__))

BASE_INPUT_DIR_SAME = os.path.join(PATH, "same_data_comparison")
BASE_INPUT_DIR_INLIERS = os.path.join(PATH, "filtered_inliers")

SAME_DATA_FILE = os.path.join(BASE_INPUT_DIR_SAME, "same_data_model_comparison.csv")
INLIERS_FILE = os.path.join(BASE_INPUT_DIR_INLIERS, "inliers_model_comparison.csv")

OUTPUT_DIR = os.path.join(PATH, "err_err_plots", "combined")
AXIS_MAX = 10.0

COLORS = {
    "linear": "blue",
    "gamma": "green",
    "poly3c": "red",
}

LABELS = {
    "linear": "Lineární",
    "gamma": "Gamma",
    "poly3c": "Polynomický",
}

MARKERS = {
    "inliers": "o",
    "same_data": "X",
}

DATASET_LABELS = {
    "inliers": "Inliers",
    "same_data": "Stejná data",
}


def open_file(filename):
    results = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def compute_ranking_score(results):
    scores = {}

    for r in results:
        glyph = r["glyph_type"]
        model = r["model"]

        u = float(r["avg_unsigned"]) * 100
        s = abs(float(r["avg_signed"])) * 100
        R = np.sqrt(u * u + s * s)

        if glyph not in scores:
            scores[glyph] = {}

        scores[glyph][model] = R

    return scores

def export_to_file(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["glyph_type", "model", "data", "avg_unsigned", "avg_signed", "R"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def error_error_plot_combined(same_results, inlier_results, glyph, output):
    fig, ax = plt.subplots(figsize=(8, 6))

    datasets = {
        "inliers": inlier_results,
        "same_data": same_results,
    }

    scores = {
        "inliers": compute_ranking_score(inlier_results),
        "same_data": compute_ranking_score(same_results),
    }

    content = []

    for dataset_name, results in datasets.items():
        for model in COLORS.keys():
            for r in results:
                if r["glyph_type"] == glyph and r["model"] == model:
                    u = float(r["avg_unsigned"]) * 100
                    s = float(r["avg_signed"]) * 100
                    s_abs = abs(s)

                    R = scores[dataset_name].get(glyph, {}).get(model, None)

                    content.append({
                        "glyph_type": glyph,
                        "model": model,
                        "data": dataset_name,
                        "avg_unsigned": u,
                        "avg_signed": s,
                        "R": R
                    })

                    ax.scatter(
                        u,
                        s_abs,
                        color=COLORS[model],
                        marker=MARKERS[dataset_name],
                        label=LABELS[model],
                        alpha=0.75,
                        s=75
                    )

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xlabel("Průměrná neznaménková chyba - $u$", fontsize=12)
    ax.set_ylabel("Průměrná znaménková chyba - $|s|$", fontsize=12)

    ax.set_xlim(0, AXIS_MAX)
    ax.set_ylim(0, AXIS_MAX)
    ax.set_aspect("equal", adjustable="box")

    #ax.set_title(glyph.capitalize().replace("_", " "))
    ax.grid(True, linestyle="--", alpha=0.2)

    # Contours R
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, 300),
        np.linspace(ymin, ymax, 300)
    )
    Z = np.sqrt(X**2 + Y**2)

    levels = [1, 2, 4, 6, 8, 10]
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=1, alpha=0.35)
    ax.clabel(cs, inline=True, fontsize=8, fmt="R=%g")

    handles, labels = ax.get_legend_handles_labels()

    # legenda modelů podle barev
    model_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=COLORS[model],
            markeredgecolor=COLORS[model],
            markersize=8,
            label=LABELS[model],
            linestyle='None',
        )
        for model in COLORS.keys()
    ]

    # legenda datasetů podle markerů
    dataset_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='gray',
            markerfacecolor='gray',
            markeredgecolor='gray',
            markersize=8,
            linestyle='None',
            label='Filtrovaná'
        ),
        Line2D(
            [0], [0],
            marker='X',
            color='gray',
            markerfacecolor='gray',
            markeredgecolor='gray',
            markersize=8,
            linestyle='None',
            label='Stejná'
        )
    ]

    legend1 = ax.legend(
        handles=model_handles,
        title="Modely",
        fontsize="small",
        title_fontsize="small",
        loc="upper left"
    )

    ax.add_artist(legend1)

    ax.legend(
        handles=dataset_handles,
        title="Data",
        fontsize="small",
        title_fontsize="small",
        loc="upper right"
    )

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot to", output)

    return content


same_results = open_file(SAME_DATA_FILE)
inlier_results = open_file(INLIERS_FILE)

all_glyphs = list({**SIMPLE_GLYPHS, **ADVANCED_GLYPHS}.keys())

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_content = []

for g in all_glyphs:
    content = error_error_plot_combined(
        same_results,
        inlier_results,
        g,
        os.path.join(OUTPUT_DIR, f"combined_error_error_{g}.png")
    )
    all_content.extend(content)

export_to_file(all_content, os.path.join(OUTPUT_DIR, f"combined_error_error_all_glyphs.csv"))
