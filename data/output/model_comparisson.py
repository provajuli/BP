import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/jp/a_school/BP')
from app.glyph_set import SIMPLE_GLYPHS, ADVANCED_GLYPHS

PATH = os.path.dirname(os.path.abspath(__file__))

FILENAME = os.path.join(PATH, 'model_comparisson.csv')

# zpracuju csv soubor s vysledky
def open_file(filename = FILENAME):
    results = []
    with open (filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

# v jednom plotu budou simple nebo advanced glyphs
# glyphy budou odlisene barvou 
# modely budou odlisene tvarem scatter points
def error_error_plot(results):
    plt.figure (figsize=(8, 6))

    simple_glyphs = SIMPLE_GLYPHS.keys()
    
    for r in results:
        if(r['glyph_type'] in simple_glyphs):
            scatter_point_x, scatter_point_y = float(r['unsigned_euclidean_sum']), float(r['signed_euclidean_sum'])
            
            model = r['model']
            print(model)

            plt.scatter(scatter_point_x, scatter_point_y, color='b', marker='o')
    #plt.show()

results = open_file()
#for row in results:
#    print(row)

for g in SIMPLE_GLYPHS.keys():
    error_error_plot(results)