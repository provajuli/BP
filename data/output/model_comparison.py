import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

#sys.path.insert(0, '/home/jp/a_school/BP')
#from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

PATH = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(PATH, 'model_comparison.csv')
SIMPLE_OUTPUT_PLOT = os.path.join(PATH, 'error_error_comparison_1.png')
ADVANCED_OUTPUT_PLOT = os.path.join(PATH, 'error_error_comparison_2.png')

# zpracuju csv soubor s vysledky
def open_file(filename = FILENAME):
    results = []
    with open (filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

MARKERS = {
    "linear": "o",
    "gamma": "s",
    "poly3c": "^",
}

# toto asi potreba nebude, jen potrebuju seznam glyphu... import z app.glyph_set opravit
COLORS = {
    "circle": "r",
    "square": "b",
    "star": "green",
    "polygon": "purple",
    "line": "gray",
    "beer": "yellow",
    "circular_progressbar": "pink",
    "flower": "green",
    "tree_growth": "brown",
    "sun": "orange"
}

# v jednom plotu budou simple nebo advanced glyphs
# glyphy budou odlisene barvou 
# modely budou odlisene tvarem scatter points
# TODO: pridat contour lines, upravit legend
def error_error_plot(results, glyph, title, output):
    plt.figure (figsize=(8, 6))

    for model, marker in MARKERS.items():
        x_vals = []
        y_vals = []
        for r in results:
            if r['glyph_type'] == glyph and r['model'] == model:
                avg_unsigned = r['avg_unsigned']
                avg_signed = r['avg_signed']

                # normalizace dat
                x_vals.append(float(avg_unsigned))
                y_vals.append(float(avg_signed))

                plt.scatter(float(avg_unsigned), float(avg_signed), 
                            color=COLORS[glyph], marker=marker, s=80,
                            label=f"{model.capitalize()} Model")
                
    plt.xlabel("Unsigned Euclidean Error Average per Point")
    plt.ylabel("Signed Euclidean Error Average per Point")
    # TODO: nastavit limity podle normalizovanych hodnot, mozna si ty mini cisla vynasobim konstantou
    plt.title(f"Error Comparison for {title} Glyph")
    plt.legend(loc='best', fontsize='small', title="Models", title_fontsize='medium')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot to", output)

results = open_file()

# mozna si to plotnu pro jednotlive glyphy, uvidime... 
for g in COLORS.keys():
    error_error_plot(results, g, g.capitalize(), f"error_error_comparison_{g}.png")