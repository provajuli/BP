import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =========================
# CONFIG
# =========================
PATH = os.path.dirname(os.path.abspath(__file__))

# sem dej CSV z experimentu s FIT_TO_POLY3_MODE=True
# (tj. to, co ukládáš do data/user_data_sets/fit_poly3/results.csv)
INPUT_FILE = os.path.join(PATH, "user_data_sets/fit_poly3/results.csv")

OUT_DIR = os.path.join(PATH, "data_processing_output", "fit_poly3_eval")
PLOTS_DIR = os.path.join(OUT_DIR, "beak_plots")
CSV_DIR = os.path.join(OUT_DIR, "csv")

OUTLIERS_PCT = 5.0  # pro "bez outliers"; pro "s outliers" se použije 0.0

GAMMA_MIN = 0.05
GAMMA_MAX = 3.0

CC_THETA0 = np.array([1.0, 0.0], dtype=float)
CC_LOW    = np.array([0.0, -3.0], dtype=float)
CC_HIGH   = np.array([1.2,  3.0], dtype=float)

EPS = 1e-9


# =========================
# IO + NORMALIZE
# =========================
def open_data_file(filename):
    glyph_types, A, B, C = [], [], [], []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            glyph_types.append(row["glyph_type"].strip().lower())
            A.append(float(row["sizeA"]))
            B.append(float(row["sizeB"]))
            C.append(float(row["sizeC"]))
    return np.array(glyph_types), np.array(A), np.array(B), np.array(C)


def normalize_01(x):
    return np.clip(np.asarray(x, float) / 100.0, 0.0, 1.0)


# =========================
# MODELS
# =========================
def gamma_function(x, gamma):
    x = np.asarray(x, float)
    g = max(float(gamma), 1e-6)
    return np.power(np.clip(x, 0, 1), g)

def predict_b_gamma(gamma, a, c):
    middle = (a**gamma + c**gamma) / 2.0
    return middle ** (1.0/gamma)

def residuals_gamma(gamma, a, b, c):
    return predict_b_gamma(gamma, a, c) - b

def fit_gamma(a, b, c):
    res = least_squares(
        residuals_gamma,
        x0=1.0,
        bounds=(GAMMA_MIN, GAMMA_MAX),
        args=(a, b, c),
        max_nfev=5000
    )
    return float(res.x[0])

# ---- poly3 constrained ----
def cubic_constrained_function(x, b_par, c_par):
    x = np.asarray(x, float)
    return b_par*x + c_par*(x**2) + (1.0 - b_par - c_par)*(x**3)

def inv_cubic_constrained(y, b_par, c_par, eps=1e-9):
    y = np.atleast_1d(np.asarray(y, float)).ravel()
    out = np.empty_like(y)

    A = (1.0 - b_par - c_par)
    B = c_par
    C = b_par

    def fval(xx):
        return b_par*xx + c_par*xx**2 + (1.0-b_par-c_par)*xx**3

    for i, yi in enumerate(y):
        yi = float(np.clip(yi, 0.0, 1.0))
        roots = np.roots([A, B, C, -yi])
        real = roots[np.isreal(roots)].real.astype(float)

        if real.size == 0:
            xi = 0.0
        else:
            in01 = real[(real >= -eps) & (real <= 1.0 + eps)]
            if in01.size > 0:
                cand = np.clip(in01, 0.0, 1.0)
                xi = cand[np.argmin(np.abs(fval(cand) - yi))]
            else:
                xi = np.clip(real[np.argmin(np.minimum(np.abs(real - 0.0), np.abs(real - 1.0)))], 0.0, 1.0)

        out[i] = np.clip(xi, 0.0, 1.0)

    return out.reshape(y.shape)

def predict_b_cubic_constrained(b_par, c_par, A, C):
    mid = 0.5 * (cubic_constrained_function(A, b_par, c_par) +
                 cubic_constrained_function(C, b_par, c_par))
    return inv_cubic_constrained(mid, b_par, c_par)

def residuals_cubic_constrained(theta, A, B, C):
    b_par = float(theta[0])
    c_par = float(theta[1])
    return predict_b_cubic_constrained(b_par, c_par, A, C) - B

def fit_cubic_constrained(a, b, c):
    res = least_squares(
        residuals_cubic_constrained,
        x0=CC_THETA0,
        bounds=(CC_LOW, CC_HIGH),
        args=(np.asarray(a,float), np.asarray(b,float), np.asarray(c,float)),
        max_nfev=8000
    )
    return float(res.x[0]), float(res.x[1])


# =========================
# BEAKS + DISTANCES
# =========================
def beak_points(A, B, C, f, return_all=False):
    fA = f(A)
    fC = f(C)
    beak_y = 0.5 * (fA + fC)
    beak_x = B
    if return_all:
        return beak_x, beak_y, fA, fC
    return beak_x, beak_y

def euclidean_distance_from_curve(beak_x, beak_y, curve_x, curve_y, select_signed=False):
    beak_x = np.asarray(beak_x, float)
    beak_y = np.asarray(beak_y, float)
    curve_x = np.asarray(curve_x, float)
    curve_y = np.asarray(curve_y, float)

    n_curve = len(curve_x)
    distances = np.empty(len(beak_x), float)

    for i, (xb, yb) in enumerate(zip(beak_x, beak_y)):
        j = 0
        while j < n_curve and curve_x[j] < xb:
            j += 1

        cand = []
        if j < n_curve: cand.append(j)
        if j-1 >= 0:    cand.append(j-1)

        best = None
        best_sign = 1
        for k in cand:
            dx = xb - curve_x[k]
            dy = yb - curve_y[k]
            d = float(np.sqrt(dx*dx + dy*dy))

            if select_signed:
                sign = 1 if dy >= 0 else -1

            if (best is None) or (d < best):
                best = d
                if select_signed:
                    best_sign = sign

        distances[i] = 0.0 if best is None else best * (best_sign if select_signed else 1.0)

    return distances

def inliers_mask(distances, outliers_pct):
    n = len(distances)
    if outliers_pct <= 0.0:
        return np.ones(n, dtype=bool)

    keep = 1.0 - (outliers_pct / 100.0)
    n_keep = int(keep * n)

    order = np.argsort(distances)
    keep_idx = order[:n_keep]

    mask = np.zeros(n, dtype=bool)
    mask[keep_idx] = True
    return mask

def mask_for_model(A, B, C, f, outliers_pct):
    bx, by = beak_points(A, B, C, f)
    x = np.linspace(0, 1, 20000)
    y = f(x)
    dist = euclidean_distance_from_curve(bx, by, x, y, select_signed=False)
    return inliers_mask(dist, outliers_pct)

def compute_beak_error(A, B, C, f, outliers_pct):
    bx, by = beak_points(A, B, C, f)
    x = np.linspace(0, 1, 20000)
    y = f(x)

    unsigned = euclidean_distance_from_curve(bx, by, x, y, select_signed=False)
    signed   = euclidean_distance_from_curve(bx, by, x, y, select_signed=True)

    mask = inliers_mask(unsigned, outliers_pct)

    unsigned_sum = float(np.sum(unsigned[mask]))
    signed_sum   = float(np.sum(signed[mask]))
    n = int(np.sum(mask))

    return {
        "unsigned_sum": unsigned_sum,
        "signed_sum": signed_sum,
        "n": n,
        "avg_unsigned": unsigned_sum / max(n, 1),
        "avg_signed": signed_sum / max(n, 1),
    }

def beak_plot(ax, f, A, B, C, mask, title, plot_outliers):
    x = np.linspace(0, 1, 20000)
    y = f(x)
    ax.plot(x, y, lw=2, label="model")

    bx, by, fA, fC = beak_points(A, B, C, f, return_all=True)

    if mask is None:
        mask = np.ones(len(bx), dtype=bool)

    # inliers
    for Ai, Bi, Ci, yAi, yBi, yCi in zip(A[mask], bx[mask], C[mask], fA[mask], by[mask], fC[mask]):
        ax.plot([Ai, Bi], [yAi, yBi], color="gray", lw=0.3)
        ax.plot([Ci, Bi], [yCi, yBi], color="gray", lw=0.3)
        ax.scatter([Ai, Ci], [yAi, yCi], color="black", s=8)
        ax.scatter([Bi], [yBi], color="blue", s=14)

    # outliers
    if plot_outliers:
        out = ~mask
        for Ai, Bi, Ci, yAi, yBi, yCi in zip(A[out], bx[out], C[out], fA[out], by[out], fC[out]):
            ax.plot([Ai, Bi], [yAi, yBi], color="red", lw=0.3, alpha=0.4)
            ax.plot([Ci, Bi], [yCi, yBi], color="red", lw=0.3, alpha=0.4)
            ax.scatter([Ai, Ci], [yAi, yCi], color="darkorange", s=8, alpha=0.4)
            ax.scatter([Bi], [yBi], color="red", s=14, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(ls="--", alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (perceived)")


# =========================
# PIPELINE PER GLYPH
# =========================
def fit_gamma_model(A, B, C, outliers_pct):
    # fit all
    g0 = fit_gamma(A, B, C)
    f0 = lambda x: gamma_function(x, g0)

    if outliers_pct <= 0.0:
        return g0, np.ones(len(A), bool)

    mask = mask_for_model(A, B, C, f0, outliers_pct)
    g1 = fit_gamma(A[mask], B[mask], C[mask])
    f1 = lambda x: gamma_function(x, g1)
    mask = mask_for_model(A, B, C, f1, outliers_pct)
    return g1, mask

def fit_poly3c_model(A, B, C, outliers_pct):
    b0, c0 = fit_cubic_constrained(A, B, C)
    f0 = lambda x: cubic_constrained_function(x, b0, c0)

    if outliers_pct <= 0.0:
        return (b0, c0), np.ones(len(A), bool)

    mask = mask_for_model(A, B, C, f0, outliers_pct)
    b1, c1 = fit_cubic_constrained(A[mask], B[mask], C[mask])
    f1 = lambda x: cubic_constrained_function(x, b1, c1)
    mask = mask_for_model(A, B, C, f1, outliers_pct)
    return (b1, c1), mask

def run_for_glyph(glyph, glyph_types, A, B, C, outliers_pct, plot_outliers):
    m = glyph_types == glyph
    A_g, B_g, C_g = A[m], B[m], C[m]

    # --- Linear
    f_lin = lambda x: x
    mask_lin = np.ones(len(A_g), bool) if outliers_pct <= 0 else mask_for_model(A_g, B_g, C_g, f_lin, outliers_pct)
    err_lin = compute_beak_error(A_g, B_g, C_g, f_lin, outliers_pct)

    # --- Gamma
    gamma, mask_g = fit_gamma_model(A_g, B_g, C_g, outliers_pct)
    f_g = lambda x: gamma_function(x, gamma)
    err_g = compute_beak_error(A_g, B_g, C_g, f_g, outliers_pct)

    # --- Poly3C
    (b_cc, c_cc), mask_cc = fit_poly3c_model(A_g, B_g, C_g, outliers_pct)
    f_cc = lambda x: cubic_constrained_function(x, b_cc, c_cc)
    err_cc = compute_beak_error(A_g, B_g, C_g, f_cc, outliers_pct)

    # --- Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    beak_plot(axes[0], f_lin, A_g, B_g, C_g, mask_lin,
              title=f"{glyph} | Linear | inliers={mask_lin.sum()}/{len(mask_lin)}",
              plot_outliers=plot_outliers)

    beak_plot(axes[1], f_g, A_g, B_g, C_g, mask_g,
              title=f"{glyph} | Gamma γ={gamma:.3f} | inliers={mask_g.sum()}/{len(mask_g)}",
              plot_outliers=plot_outliers)

    beak_plot(axes[2], f_cc, A_g, B_g, C_g, mask_cc,
              title=f"{glyph} | Poly3C b={b_cc:.3f}, c={c_cc:.3f} | inliers={mask_cc.sum()}/{len(mask_cc)}",
              plot_outliers=plot_outliers)

    fig.suptitle(f"FIT_TO_POLY3 dataset — {glyph} — outliers_pct={outliers_pct}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tag = "with_outliers" if plot_outliers else "no_outliers"
    outpath = os.path.join(PLOTS_DIR, f"beak_{tag}_{glyph}.png")
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # return metrics row(s)
    return [
        (glyph, "linear",  err_lin,  {}),
        (glyph, "gamma",   err_g,   {"gamma": gamma}),
        (glyph, "poly3c",  err_cc,  {"b": b_cc, "c": c_cc}),
    ]


# =========================
# MAIN
# =========================
def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

def write_metrics_csv(rows, filename):
    path = os.path.join(CSV_DIR, filename)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "glyph_type", "model",
            "unsigned_sum", "signed_sum",
            "n_used", "avg_unsigned", "avg_signed",
            "param_gamma", "param_b", "param_c"
        ])
        for glyph, model, err, params in rows:
            w.writerow([
                glyph, model,
                f"{err['unsigned_sum']:.6f}",
                f"{err['signed_sum']:.6f}",
                err["n"],
                f"{err['avg_unsigned']:.6f}",
                f"{err['avg_signed']:.6f}",
                f"{params.get('gamma', np.nan)}",
                f"{params.get('b', np.nan)}",
                f"{params.get('c', np.nan)}",
            ])
    print("[ok] wrote", path)

def main():
    ensure_dirs()
    glyph_types, a, b, c = open_data_file(INPUT_FILE)
    print(f"[info] Loaded {len(a)} rows from {INPUT_FILE}")

    A = normalize_01(a)
    B = normalize_01(b)
    C = normalize_01(c)

    glyphs = sorted(np.unique(glyph_types))

    # 1) WITH OUTLIERS (fit+metrics na všech bodech, outliers_pct=0)
    rows_with = []
    for g in glyphs:
        rows_with.extend(run_for_glyph(g, glyph_types, A, B, C, outliers_pct=0.0, plot_outliers=True))
    write_metrics_csv(rows_with, "fit_poly3_with_outliers.csv")

    # 2) WITHOUT OUTLIERS (robust: mask+refit+metrics na inlierech)
    rows_no = []
    for g in glyphs:
        rows_no.extend(run_for_glyph(g, glyph_types, A, B, C, outliers_pct=OUTLIERS_PCT, plot_outliers=False))
    write_metrics_csv(rows_no, "fit_poly3_no_outliers.csv")

    print("[done] plots in:", PLOTS_DIR)

if __name__ == "__main__":
    main()
