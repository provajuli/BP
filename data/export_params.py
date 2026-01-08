import json
import os
import numpy as np

from data_processing import open_data_file, normalize_data, fit_poly3c_model, fit_gamma_model

OUTLIERS_PCT = 5.0
INPUT_FILE = "data_processing_input/filtered_results.csv"
OUTPUT_FILE_POLY3 = "data_processing_output/poly3c_params_by_glyph.json"
OUTPUT_FILE_GAMMA = "data_processing_output/gamma_params_by_glyph.json"

def export_poly3c_params(csv_in, json_out, outliers_pct = OUTLIERS_PCT):
    glyph_types, a, b, c = open_data_file(csv_in)

    A = normalize_data(a)
    B = normalize_data(b)
    C = normalize_data(c)

    params = {}

    for g in np.unique(glyph_types):
        m = (glyph_types == g)
        (b_fit, c_fit), mask_in = fit_poly3c_model(A[m], B[m], C[m], outliers_pct=outliers_pct)
        params[g] = {
            "b": float(b_fit),
            "c": float(c_fit),
            "n_total": int(np.sum(m)),
            "n_inliers": int(np.sum(mask_in)),
        }

    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("[ok] saved poly3c params to", json_out)

def export_gamma_params(csv_in, json_out, outliers_pct=OUTLIERS_PCT):
    glyph_types, a, b, c = open_data_file(csv_in)

    A = normalize_data(a)
    B = normalize_data(b)
    C = normalize_data(c)

    params = {}

    for g in np.unique(glyph_types):
        m = (glyph_types == g)

        gamma_fit, mask_in = fit_gamma_model(
            A[m], B[m], C[m], outliers_pct=outliers_pct
        )

        params[g] = {
            "gamma": float(gamma_fit),
            "n_total": int(np.sum(m)),
            "n_inliers": int(np.sum(mask_in)),
        }

    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("[ok] saved gamma params to", json_out)

if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__))
    csv_in = os.path.join(PATH, INPUT_FILE)
    json_out_poly3 = os.path.join(PATH, OUTPUT_FILE_POLY3)
    export_poly3c_params(csv_in, json_out_poly3, outliers_pct=5.0)

    json_out_gamma = os.path.join(PATH, OUTPUT_FILE_GAMMA)
    export_gamma_params(csv_in, json_out_gamma, outliers_pct=OUTLIERS_PCT)

