import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/jp/a_school/BP')
from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

PATH = os.path.dirname(os.path.abspath(__file__))

FILENAME = os.path.join(PATH, 'model_comparison.csv')
OUTPUT_PLOT = os.path.join(PATH, 'error_error_comparison.png')

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
    "line": "gray"
}


# v jednom plotu budou simple nebo advanced glyphs
# glyphy budou odlisene barvou 
# modely budou odlisene tvarem scatter points
# TODO: pridat contour lines 
def error_error_plot(results):
    plt.figure (figsize=(8, 6))

    simple_glyphs = SIMPLE_GLYPHS.keys()

    for r in results:
        if(r['glyph_type'] in simple_glyphs):
            glyph = r['glyph_type']
            model = r['model']
            
            scatter_point_x, scatter_point_y = float(r['unsigned_euclidean_sum']), float(r['signed_euclidean_sum'])

            plt.scatter(scatter_point_x, scatter_point_y, 
                        color=COLORS[glyph], 
                        marker=MARKERS[model],
                        label=f"{glyph} - {model}")
            
    plt.xlabel("Unsigned Euclidean Error Sum")
    plt.ylabel("Signed Euclidean Error Sum")
    plt.title("Error Comparison for Simple Glyphs")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot to", OUTPUT_PLOT)

results = open_file()
#for row in results:
#    print(row)

error_error_plot(results)