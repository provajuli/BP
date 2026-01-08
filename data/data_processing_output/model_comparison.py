import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# ERR ERR plots

# !!! V BP nutno zminit v jake metrice model selhava !!!

sys.path.insert(0, '/home/jp/a_school/BP')
from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

PATH = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(PATH, 'filtered_inliers/inliers_model_comparison.csv')
OUTPUT_DIR = os.path.join(PATH, 'err_err_plots')


# zpracuju csv soubor s vysledky
def open_file(filename = FILENAME):
    results = []
    with open (filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


COLORS = {
    "linear": "blue",
    "gamma": "green",
    "poly3c": "red",
    "piecewise": "orange"
}


def compute_global_max(results):
    global_max = 0.0
    for r in results:
        x = float(r["avg_unsigned"]) * 100
        y = abs(float(r["avg_signed"])) * 100
        global_max = max(global_max, x, y)
    return global_max


def error_error_plot(results, glyph, title, output, global_max):
    fig, ax = plt.subplots(figsize=(8, 6))

    areas = compute_area(results)

    for model in COLORS.keys():
        for r in results:
            if r['glyph_type'] == glyph and r['model'] == model:
                avg_unsigned_mult = float(r['avg_unsigned']) * 100
                avg_signed_mult = abs(float(r['avg_signed'])) * 100

                area = areas.get(glyph, {}).get(model, None)
                area_txt = f"{area:.2f}"

                ax.scatter(
                    avg_unsigned_mult,
                    avg_signed_mult,
                    color=COLORS[model],
                    label=(
                        f"{model.capitalize()} "
                        f"(S={area_txt}, "
                        f"u={avg_unsigned_mult:.2f}, "
                        f"s={avg_signed_mult:.2f})"
                    ),
                    alpha=0.6,
                    s=48
                )

    # osy v nule
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # popisky os
    ax.set_xlabel("Unsigned average distance from curve/beak_point")
    ax.set_ylabel("Signed average absolute distance from curve/beak_point")

    # ticks
    ax.tick_params(axis="both", which="major", pad=6)

    # limity
    ax.set_xlim(0, global_max + 0.5)
    ax.set_ylim(0, global_max + 0.5)
    ax.set_aspect("equal", adjustable="box")

    ax.grid(True, linestyle="--", alpha=0.2)

    # titulek + legenda
    ax.set_title(f"Error for {title} Glyph")

    # --- vrstevnice konstantního obsahu A = x*y ---
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    X, Y = np.meshgrid(np.linspace(xmin, xmax, 300),
                    np.linspace(ymin, ymax, 300))
    Z = X * Y

    levels = [0.5, 1, 2, 4, 8]
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=1, alpha=0.35)
    ax.clabel(cs, inline=True, fontsize=8, fmt="S=%g")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="best",
        fontsize="small",
        title="Models",
        title_fontsize="medium",
    )

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot to", output)


# vrati slovnik s key - glyph, value - ( key: model, value: square area)
def compute_area(results):
    dict = {}
    
    for  r in results:
        glyph = r['glyph_type']
        model = r['model']

        unsigned = float(r['avg_unsigned'])
        signed = abs(float(r['avg_signed']))

        area = abs((unsigned * 100) * (signed * 100))

        if glyph not in dict:
            dict[glyph] = {}

        dict[glyph][model] = area
    
    return dict

results = open_file()
global_max = compute_global_max(results)
all_glyphs = list({**SIMPLE_GLYPHS, **ADVANCED_GLYPHS}.keys())

for g in all_glyphs:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    error_error_plot(results, g, g.capitalize(), f"{OUTPUT_DIR}/error_error_{g}.png", global_max)

areas = compute_area(results)

for glyph, model_values in areas.items():
    for model, area in model_values.items():
        print(glyph, model, f"{area:.2f}")