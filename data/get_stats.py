import numpy as np
import os
import csv

PATH = os.path.dirname(os.path.abspath(__file__))
print(PATH)

def cubic_constrained_function(x, b_par, c_par):
    x01 = x / 100
    return b_par*x01 + c_par*(x01**2) + (1.0 - b_par - c_par)*(x01**3)


def remove_outliers_by_abs_e(e_vals, t_vals, outliers_pct):
    """
    Odstraní outliery podle |e| (percentilový trim)
    """
    if outliers_pct <= 0.0:
        return e_vals, t_vals

    abs_e = np.abs(e_vals)
    thresh = np.percentile(abs_e, 100.0 - outliers_pct)

    mask = abs_e <= thresh
    return e_vals[mask], t_vals[mask]


def compute_perceptual_errors(trials, b, c):
    """
    trials ... list dictů s klíči sizeA, sizeB, sizeC
    """
    e_vals = []
    t_vals = []

    for r in trials:
        A = float(r["sizeA"])
        B = float(r["sizeB"])
        C = float(r["sizeC"])

        pA = cubic_constrained_function(A, b, c)
        pB = cubic_constrained_function(B, b, c)
        pC = cubic_constrained_function(C, b, c)

        # midpoint error
        e = pB - 0.5 * (pA + pC)
        e_vals.append(e)

        # relativní poloha
        if abs(pC - pA) > 1e-9:
            t = (pB - pA) / (pC - pA)
            t_vals.append(t)

    e_vals = np.array(e_vals)
    t_vals = np.array(t_vals)

    e_vals, t_vals = remove_outliers_by_abs_e(e_vals, t_vals, outliers_pct=5.0)

    return {
        "mean_abs_e": np.mean(np.abs(e_vals)),
        "std_e": np.std(e_vals),
        "mean_t": np.mean(t_vals),
        "std_t": np.std(t_vals),
        "n": len(e_vals),
    }


EXPERIMENT_FILE = os.path.join(PATH, "data_processing_input/filtered_results.csv")

def open_trials(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


POLY3C_PARAMS = {
    "star": {"b": 0.379, "c": 1.767},
}


trials = open_trials(EXPERIMENT_FILE)

print("\n=== Perceptual error metrics STAR - USER DATA ===\n")
for glyph, params in POLY3C_PARAMS.items():
    glyph_trials = [r for r in trials if r["glyph_type"] == glyph]

    stats = compute_perceptual_errors(
        glyph_trials,
        b=params["b"],
        c=params["c"],
    )

    print(
        f"{glyph:10s} | "
        f"mean(|e|)={stats['mean_abs_e']:.4f} | "
        f"std(e)={stats['std_e']:.4f} | "
        f"mean(t)={stats['mean_t']:.3f} | "
        f"std(t)={stats['std_t']:.3f} | "
        f"n={stats['n']}"
    )


EXPERIMENT_FILE_POLY3 = os.path.join(PATH, "user_data_sets/fit_poly3/results.csv")

poly3_trials = open_trials(EXPERIMENT_FILE_POLY3)

print("\n=== Perceptual error metrics STAR - USER DATA FIT TO POLY3 ===\n")
for glyph, params in POLY3C_PARAMS.items():
    glyph_trials = [r for r in poly3_trials if r["glyph_type"] == glyph]

    stats = compute_perceptual_errors(
        glyph_trials,
        b=params["b"],
        c=params["c"],
    )

    print(
        f"{glyph:10s} | "
        f"mean(|e|)={stats['mean_abs_e']:.4f} | "
        f"std(e)={stats['std_e']:.4f} | "
        f"mean(t)={stats['mean_t']:.3f} | "
        f"std(t)={stats['std_t']:.3f} | "
        f"n={stats['n']}"
    )