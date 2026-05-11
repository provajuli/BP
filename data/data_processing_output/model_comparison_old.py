import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PATH, "..", ".."))

sys.path.insert(0, PROJECT_ROOT)

print("Current PATH:", PATH)
print("Project root:", PROJECT_ROOT)

from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS
# -------------------- CONFIG --------------------
#MODE = "same_data"  
MODE = "model_inliers"

BASE_INPUT_DIR_SAME = os.path.join(
    PATH,
    "same_data_comparison"
)

BASE_INPUT_DIR_INLIERS = os.path.join(
    PATH,
    "filtered_inliers"
)

INPUT_FILES = {
    "same_data": "same_data_model_comparison.csv",
    "model_inliers": "inliers_model_comparison.csv",
}

OUTPUT_DIR = os.path.join(PATH, "err_err_plots", MODE)

FILENAME = os.path.join(BASE_INPUT_DIR_SAME, INPUT_FILES[MODE]) if MODE == "same_data" else os.path.join(BASE_INPUT_DIR_INLIERS, INPUT_FILES[MODE])


# -------------------- LOAD DATA --------------------
def open_file(filename=FILENAME):
    results = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


COLORS = {
    "linear": "blue",
    "gamma": "green",
    "poly3c": "red",
    "pchip": "orange"
}

LABELS = {
    "linear": "Linear",
    "gamma": "Gamma",
    "poly3c": "Polynomial",
}


# -------------------- METRICS --------------------
def compute_global_max(results):
    global_max = 0.0
    for r in results:
        x = float(r["avg_unsigned"]) * 100
        y = abs(float(r["avg_signed"])) * 100
        global_max = max(global_max, x, y)
    return global_max


def compute_ranking_score(results):
    """
    R = sqrt(u^2 + s^2)
    """
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


# -------------------- PLOT --------------------
def error_error_plot(results, glyph, title, output, global_max):
    fig, ax = plt.subplots(figsize=(8, 6))

    scores = compute_ranking_score(results)

    for model in COLORS.keys():
        for r in results:
            if r['glyph_type'] == glyph and r['model'] == model:
                u = float(r["avg_unsigned"]) * 100
                s = float(r["avg_signed"]) * 100
                s_abs = abs(float(r["avg_signed"])) * 100

                R = scores.get(glyph, {}).get(model, None)

                ax.scatter(
                    u,
                    s_abs,
                    color=COLORS[model],
                    label=f"{LABELS[model]} (R={R:.2f}, u={u:.2f}, s={s:.2f})",
                    alpha=0.7,
                    s=75
                )

    # osy
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xlabel("Average Unsigned Error - $u$")
    ax.set_ylabel("Average Signed Error - $|s|$")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal", adjustable="box")

    #ax.title.set_text(glyph.capitalize().replace("_", " "))

    ax.grid(True, linestyle="--", alpha=0.2)

    # -------------------- CONTOURS (R) --------------------
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

    # legenda bez duplicit
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize="small",
        title="Models",
        title_fontsize="medium",
        loc="upper left",
    )

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot to", output)


# -------------------- MAIN --------------------
results = open_file()
global_max = compute_global_max(results)
scores = compute_ranking_score(results)

all_glyphs = list({**SIMPLE_GLYPHS, **ADVANCED_GLYPHS}.keys())

os.makedirs(OUTPUT_DIR, exist_ok=True)

for g in all_glyphs:
    error_error_plot(
        results,
        g,
        g.capitalize(),
        f"{OUTPUT_DIR}/{MODE}_error_error_{g}.png",
        global_max
    )

# -------------------- PRINT RANKING --------------------
print("\n=== MODEL RANKING (lower R = better) ===")

for glyph, model_scores in scores.items():
    ranking = sorted(model_scores.items(), key=lambda x: x[1])

    print(f"\n{glyph}:")
    for i, (model, score) in enumerate(ranking, start=1):
        print(f"  {i}. {model:<10} R={score:.2f}")