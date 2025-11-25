import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/jp/a_school/BP')
from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

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
def error_error_plot(results, glyph_dict, title, output):
    plt.figure (figsize=(8, 6))

    glyph_types = glyph_dict.keys()

    for r in results:
        if(r['glyph_type'] in glyph_types):
            glyph = r['glyph_type']
            model = r['model']
            
            # TODO: NORMALIZOVAT PRES POCET TRIALU
            # TODO: upravit pls legendu, to je hruza :,)
            unsigned_sum = float(r['unsigned_euclidean_sum']) / float(r['n_points'])
            signed_sum = float(r['signed_euclidean_sum']) / float(r['n_points'])

            scatter_point_x, scatter_point_y = unsigned_sum, signed_sum

            plt.scatter(scatter_point_x, scatter_point_y, 
                        color=COLORS[glyph], 
                        marker=MARKERS[model],
                        label=f"{glyph} - {model}")
            
    plt.xlabel("Unsigned Euclidean Error Sum")
    plt.ylabel("Signed Euclidean Error Sum")
    # TODO: nastavit limity podle normalizovanych hodnot
    plt.xlim(0, 0.04)
    plt.ylim(-0.03, 0.03)
    plt.title(f"Error Comparison for {title} Glyphs")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot to", output)

results = open_file()
#for row in results:
#    print(row)

error_error_plot(results, SIMPLE_GLYPHS, "Simple", SIMPLE_OUTPUT_PLOT)

error_error_plot(results, ADVANCED_GLYPHS, "Advanced", ADVANCED_OUTPUT_PLOT)