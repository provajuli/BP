import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/jp/a_school/BP')
from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

PATH = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(PATH, 'filtered_inliers/in_model_comparison.csv')
SIMPLE_OUTPUT_PLOT = os.path.join(PATH, 'error_error_comparison_1.png')
ADVANCED_OUTPUT_PLOT = os.path.join(PATH, 'error_error_comparison_2.png')


def get_global_limits(results):
    x = []
    y = []
    for r in results:
        x_ = float(r['avg_unsigned']) * 100
        y_ = float(r['avg_signed']) * 100

        x.append(x_)
        y.append(y_)

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    return (x_min - 0.05, x_max + 0.05), (y_min - 0.5, y_max + 0.5) 


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
}

# v jednom plotu budou simple nebo advanced glyphs
# glyphy budou odlisene barvou 
# modely budou odlisene tvarem scatter points
# TODO: pridat contour lines, upravit legend
def error_error_plot(results, glyph, title, output, x_lim, y_lim):
    plt.figure (figsize=(8, 6))

    for model in COLORS.keys():
        x_vals = []
        y_vals = []
        for r in results:
            if r['glyph_type'] == glyph and r['model'] == model :
                avg_unsigned = r['avg_unsigned']
                avg_signed = r['avg_signed']

                # normalizace dat
                x_vals.append(float(avg_unsigned))
                y_vals.append(float(avg_signed))

                plt.scatter(float(avg_unsigned) * 100, float(avg_signed) * 100, 
                            color=COLORS[model],
                            marker='^',
                            label=f"{model.capitalize()} Model")

    plt.xlabel("Unsigned Euclidean Error Average per Point")
    plt.ylabel("Signed Euclidean Error Average per Point")
    # TODO: nastavit limity podle normalizovanych hodnot, mozna si ty mini cisla vynasobim konstantou
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.title(f"Error Comparison for {title} Glyph")
    plt.legend(loc='best', fontsize='small', title="Models", title_fontsize='medium')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot to", output)

# vrati slovnik s key - glyph, value - ( key: model, value: square area)
def compute_area(results):
    dict = {}
    
    for  r in results:
        glyph = r['glyph_type']
        model = r['model']

        unsigned = float(r['avg_unsigned'])
        signed = float(r['avg_signed'])

        area = abs((unsigned * 100) * (signed * 100))

        if glyph not in dict:
            dict[glyph] = {}

        dict[glyph][model] = area
    
    return dict

results = open_file()
all_glyphs = list({**SIMPLE_GLYPHS, **ADVANCED_GLYPHS}.keys())

# mozna si to plotnu pro jednotlive glyphy, uvidime... 
global_xlim, global_ylim = get_global_limits(results, all_glyphs)

for g in all_glyphs:
    error_error_plot(results, g, g.capitalize(), f"error_error_comparison_{g}.png", global_xlim, global_ylim)

areas = compute_area(results)

for glyph, model_values in areas.items():
    for model, area in model_values.items():
        print(glyph, model, area)