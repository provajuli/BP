import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import csv
import os
from scipy.interpolate import PchipInterpolator
import pandas as pd


PATH = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(PATH, "data_processing_input/filtered_results_no_close_AC.csv")
IMAGES_OUTPUT_DIR = os.path.join(PATH, "data_processing_output/beak_plots")
OUTLIER_OUTPUT_DIR = os.path.join(PATH, "data_processing_output/unfiltered_outliers")
INLIER_OUTPUT_DIR = os.path.join(PATH, "data_processing_output/filtered_inliers")
SAME_DATA_OUTPUT_DIR = os.path.join(PATH, "data_processing_output/same_data_comparison")

OUTLIER_OUTPUT_FILE_CSV = os.path.join(OUTLIER_OUTPUT_DIR, "outliers_model_comparison.csv")
INLIER_OUTPUT_FILE_CSV = os.path.join(INLIER_OUTPUT_DIR, "inliers_model_comparison.csv")
SAME_DATA_OUTPUT_FILE_CSV = os.path.join(SAME_DATA_OUTPUT_DIR, "same_data_model_comparison.csv")


####################################################
# --------------------PREPINACE---------------------
####################################################
SAVE_OUTLIER_BEAK_PLOTS = True
SAVE_INLIER_BEAK_PLOTS = True
OUTLIERS_PCT = 5.0
SAVE_RESULTS_CSV_OUT = False
SAVE_RESULTS_CSV_IN = True
SAVE_RESULTS_CSV_SAME_DATA = True
SAVE_SAME_DATA_BEAK_PLOTS = True
PLOT_GLYPH_DEVIATIONS = True
GLYPH_DEVIATION_FILTER_OUTLIERS = True

ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 0.03

GLYPH_TYPES = []
EPS = 1e-9

# -------------------------------------------------------------
def open_data_file(filename=INPUT_FILE):
    A, B, C = [], [], []
    glyph_types = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            glyph_types.append(row["glyph_type"])
            A.append(float(row["sizeA"]))
            B.append(float(row["sizeB"]))
            C.append(float(row["sizeC"]))
    return (np.array(glyph_types), np.array(A), np.array(B), np.array(C))


def normalize_data(x):
    return np.clip(x / 100.0, 0.0, 1.0)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def lsq_fit(fun, x0, bounds=(-np.inf, np.inf), args=(), max_nfev=5000, robust=False):
    kwargs = dict(
        fun=fun,
        x0=x0,
        bounds=bounds,
        args=args,
        max_nfev=max_nfev
    )
    if robust:
        kwargs["loss"] = ROBUST_LOSS
        kwargs["f_scale"] = ROBUST_F_SCALE
    return least_squares(**kwargs)


#############################################################
# --------------------GAMMA MODEL FUNKCE---------------------
#############################################################
GAMMA_MIN = 0.05
GAMMA_MAX = 3.0

def gamma_function(x, gamma):
    x = np.asarray(x, float)
    g = max(float(gamma), 1e-6)
    return np.power(np.clip(x, 0, 1), g)


def predict_b_gamma(gamma, a, c):
    middle = (a**gamma + c**gamma) / 2
    return middle ** (1.0 / gamma)


def residuals_gamma(gamma, a, b, c):
    return predict_b_gamma(gamma, a, c) - b


def fit_gamma(a, b, c, robust=False):
    res = lsq_fit(
        residuals_gamma,
        x0=1.0,
        bounds=(GAMMA_MIN, GAMMA_MAX),
        args=(a, b, c),
        max_nfev=5000,
        robust=robust
    )
    return float(res.x[0])


def gamma_stats_per_glyph_type(glyph_types, a, b, c):
    rows = []
    g_all = fit_gamma(a, b, c)
    rows.append(("All", len(b), g_all))

    for t in np.unique(glyph_types):
        idx = (glyph_types == t)
        a_t = a[idx]
        b_t = b[idx]
        c_t = c[idx]
        g_t = fit_gamma(a_t, b_t, c_t)
        rows.append((t.capitalize(), len(b_t), g_t))
    return rows


def plot_gamma_model(rows, outpath):
    x = np.linspace(0, 1, 200)
    plt.figure(figsize=(8, 6))
    for label, n, gamma in rows:
        y = x**gamma
        plt.plot(x, y, label=f"{label} (gamma={gamma:.2f}, N={n})")
    plt.xlabel("Physical size (normalized 0-1)")
    plt.ylabel("Perceived size (normalized 0-1)")
    plt.title("Perceived Size vs Physical Size — Gamma model")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()


#####################################################################
# --------------------POLY3 OMEZENA MODEL FUNKCE---------------------
#####################################################################
CC_THETA0 = np.array([1.0, 0.0], dtype=float)
CC_LOW    = np.array([0.0, -3], dtype=float)
CC_HIGH   = np.array([1.5,  3], dtype=float)

def cubic_constrained_function(x, b_par, c_par):
    x = np.asarray(x, float)
    return b_par * x + c_par * (x**2) + (1.0 - b_par - c_par) * (x**3)


def inv_cubic_constrained(y, b_par, c_par, eps=1e-9):
    y = np.atleast_1d(np.asarray(y, float)).ravel()
    out = np.empty_like(y)

    A = (1.0 - b_par - c_par)
    B = c_par
    C = b_par

    def fval(xx):
        return b_par * xx + c_par * xx**2 + (1.0 - b_par - c_par) * xx**3

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
                xi = np.clip(
                    real[np.argmin(np.minimum(np.abs(real - 0.0), np.abs(real - 1.0)))],
                    0.0,
                    1.0
                )

        out[i] = np.clip(xi, 0.0, 1.0)

    return out.reshape(y.shape)


def predict_b_cubic_constrained(b_par, c_par, A, C):
    mid = 0.5 * (
        cubic_constrained_function(A, b_par, c_par) +
        cubic_constrained_function(C, b_par, c_par)
    )
    return inv_cubic_constrained(mid, b_par, c_par)


def residuals_cubic_constrained(theta, A, B, C):
    b_par = float(theta[0])
    c_par = float(theta[1])
    return predict_b_cubic_constrained(b_par, c_par, A, C) - B


def fit_cubic_constrained(a, b, c, robust=False):
    res = lsq_fit(
        residuals_cubic_constrained,
        x0=CC_THETA0,
        bounds=(CC_LOW, CC_HIGH),
        args=(np.asarray(a, float),
              np.asarray(b, float),
              np.asarray(c, float)),
        max_nfev=5000,
        robust=robust
    )
    return res.x


def cubic_constrained_stats_by_glyph(glyph_types, a, b, c):
    rows = []
    b0, c0 = fit_cubic_constrained(a, b, c)
    rows.append(("All", len(b), 0.0, b0, c0, 1.0 - b0 - c0))
    for t in np.unique(glyph_types):
        m = (glyph_types == t)
        bt, ct = fit_cubic_constrained(a[m], b[m], c[m])
        rows.append((t.capitalize(), int(np.sum(m)), 0.0, bt, ct, 1.0 - bt - ct))
    return rows


def plot_cubic_constrained_model(rows, outpath):
    x = np.linspace(0, 1, 200)
    plt.figure(figsize=(8, 6))
    for label, n, a0, b, c, d in rows:
        y = cubic_constrained_function(x, b, c)
        plt.plot(x, y, label=f"{label} (b={b:.2f}, c={c:.2f}, N={n})")
    plt.xlabel("x (normalized)")
    plt.ylabel("y (perceived)")
    plt.title("Poly3 constrained: y = b x + c x² + (1-b-c) x³")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()

#####################################################
# --------------------BEAK PLOTS---------------------
#####################################################
def linear_function(x):
    return np.asarray(x, float)


def beak_points(A, B, C, function, return_all=False):
    fA = function(A)
    fC = function(C)
    beak_y = (fA + fC) / 2
    beak_x = B
    if return_all:
        return beak_x, beak_y, fA, fC
    return beak_x, beak_y


def plot_curve(function):
    x = np.linspace(0, 1, 20000)
    y = function(x)
    return x, y


def euclidean_distance_from_curve(beak_x, beak_y, curve_x, curve_y, select_signed=False):
    beak_x = np.asarray(beak_x, float)
    beak_y = np.asarray(beak_y, float)
    curve_x = np.asarray(curve_x, float)
    curve_y = np.asarray(curve_y, float)

    n_curve = len(curve_x)
    n_beaks = len(beak_x)

    distances = np.empty(n_beaks, float)

    for i, (xb, yb) in enumerate(zip(beak_x, beak_y)):
        j = 0
        while j < n_curve and curve_x[j] < xb:
            j += 1

        cand = []
        if j < n_curve:
            cand.append(j)
        if j - 1 >= 0:
            cand.append(j - 1)

        best = None
        best_sign = 1
        for k in cand:
            dx = xb - curve_x[k]
            dy = yb - curve_y[k]
            d = np.sqrt(dx * dx + dy * dy)

            if select_signed:
                sign = 1 if dy >= 0 else -1

            if (best is None) or (d < best):
                best = d
                if select_signed:
                    best_sign = sign

        distances[i] = 0.0 if best is None else best * (best_sign if select_signed else 1)
    return distances


def inliers_mask(distances, outliers_pct=OUTLIERS_PCT):
    distances = np.asarray(distances, float)
    n_points = len(distances)

    if outliers_pct <= 0.0:
        mask = np.ones(n_points, dtype=bool)
        sorted_idx = np.arange(n_points)
        return mask, sorted_idx

    keep = 1.0 - (outliers_pct / 100.0)
    n_keep = max(1, int(keep * n_points))

    sorted_idx = np.argsort(distances)
    inlier_idx = sorted_idx[:n_keep]

    mask = np.zeros(n_points, dtype=bool)
    mask[inlier_idx] = True
    return mask, sorted_idx


def filter_outliers(A, B, C, function, outliers_pct=OUTLIERS_PCT):
    beak_x, beak_y = beak_points(A, B, C, function, return_all=False)

    x = np.linspace(0, 1, 20000)
    y = function(x)

    distances = euclidean_distance_from_curve(beak_x, beak_y, x, y, select_signed=False)
    mask_inliers, _ = inliers_mask(distances, outliers_pct=outliers_pct)

    return A[mask_inliers], B[mask_inliers], C[mask_inliers], mask_inliers


def compute_beak_error(A, B, C, function, outliers=True, outliers_pct=OUTLIERS_PCT):
    beak_x, beak_y = beak_points(A, B, C, function, return_all=False)

    x = np.linspace(0, 1, 20000)
    y = function(x)

    unsigned_euclidean_distances = euclidean_distance_from_curve(beak_x, beak_y, x, y, select_signed=False)
    signed_euclidean_distances = euclidean_distance_from_curve(beak_x, beak_y, x, y, select_signed=True)

    if outliers:
        n_all = int(len(beak_x))
        unsigned_euclidean_sum = np.sum(unsigned_euclidean_distances)
        signed_euclidean_sum = np.sum(signed_euclidean_distances)

        return dict(
            unsigned_euclidean_sum=unsigned_euclidean_sum,
            signed_euclidean_sum=signed_euclidean_sum,
            n=n_all,
            avg_unsigned=unsigned_euclidean_sum / n_all,
            avg_signed=signed_euclidean_sum / n_all
        )

    else:
        mask_inliers, _ = inliers_mask(unsigned_euclidean_distances, outliers_pct=outliers_pct)

        unsigned_in = unsigned_euclidean_distances[mask_inliers]
        signed_in = signed_euclidean_distances[mask_inliers]

        n_in = int(len(unsigned_in))
        unsigned_euclidean_sum = np.sum(unsigned_in)
        signed_euclidean_sum = np.sum(signed_in)

        return dict(
            unsigned_euclidean_sum=unsigned_euclidean_sum,
            signed_euclidean_sum=signed_euclidean_sum,
            n=n_in,
            avg_unsigned=unsigned_euclidean_sum / n_in,
            avg_signed=signed_euclidean_sum / n_in
        )


def beak_plot_for_glyph(function, A, B, C, title, axis, mask=None, plot_outliers=False):
    x = np.linspace(0, 1, 20000)
    y = function(x)

    axis.plot(x, y, lw=2, label="model")

    beak_x, beak_y, fA, fC = beak_points(A, B, C, function, return_all=True)

    if mask is None:
        mask = np.ones(len(beak_x), dtype=bool)

    A_in = A[mask]
    C_in = C[mask]
    beak_x_in = beak_x[mask]
    beak_y_in = beak_y[mask]
    fA_in = fA[mask]
    fC_in = fC[mask]

    A_out = A[~mask]
    C_out = C[~mask]
    beak_x_out = beak_x[~mask]
    beak_y_out = beak_y[~mask]
    fA_out = fA[~mask]
    fC_out = fC[~mask]

    if plot_outliers:
        for Ai, Bi, Ci, yAi, yBi, yCi in zip(A_out, beak_x_out, C_out, fA_out, beak_y_out, fC_out):
            axis.plot([Ai, Bi], [yAi, yBi], color='red', linewidth=0.3, alpha=0.4)
            axis.plot([Ci, Bi], [yCi, yBi], color='red', linewidth=0.3, alpha=0.4)
            axis.scatter([Ai, Ci], [yAi, yCi], color='darkorange', s=8, alpha=0.4)
            axis.scatter([Bi], [yBi], color='red', s=32, alpha=0.8)

    for Ai, Bi, Ci, yAi, yBi, yCi in zip(A_in, beak_x_in, C_in, fA_in, beak_y_in, fC_in):
        axis.plot([Ai, Bi], [yAi, yBi], color='gray', linewidth=0.3)
        axis.plot([Ci, Bi], [yCi, yBi], color='gray', linewidth=0.3)
        axis.scatter([Ai, Ci], [yAi, yCi], color='black', s=8)
        axis.scatter([Bi], [yBi], color='blue', s=14)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("PHYSICAL (normalized to [0,1])")
    axis.set_ylabel("PERCEIVED (normalized to [0,1])")
    axis.set_title(title)
    axis.grid(linestyle='--', alpha=0.3)


####################################################
#                    INLIERS
####################################################
def mask_for_model(A, B, C, function, outliers_pct=OUTLIERS_PCT):
    beak_x, beak_y = beak_points(A, B, C, function, return_all=False)

    x_curve = np.linspace(0, 1, 20000)
    y_curve = function(x_curve)

    distances = euclidean_distance_from_curve(beak_x, beak_y, x_curve, y_curve, select_signed=False)

    mask_inliers, _ = inliers_mask(distances, outliers_pct=outliers_pct)
    return mask_inliers


def fit_gamma_model(A, B, C, outliers_pct=0.0):
    gamma = fit_gamma(A, B, C, robust=False)
    f = lambda x: gamma_function(x, gamma)

    if outliers_pct <= 0.0:
        mask = np.ones(len(A), dtype=bool)
        return gamma, mask

    mask = mask_for_model(A, B, C, f, outliers_pct)

    gamma = fit_gamma(A[mask], B[mask], C[mask], robust=False)
    f = lambda x: gamma_function(x, gamma)

    mask = mask_for_model(A, B, C, f, outliers_pct)
    return gamma, mask


def fit_poly3c_model(A, B, C, outliers_pct=0.0):
    b, c = fit_cubic_constrained(A, B, C, robust=False)
    f = lambda x: cubic_constrained_function(x, b, c)

    if outliers_pct <= 0.0:
        mask = np.ones(len(A), dtype=bool)
        return (b, c), mask

    mask = mask_for_model(A, B, C, f, outliers_pct)

    b, c = fit_cubic_constrained(A[mask], B[mask], C[mask], robust=False)
    f = lambda x: cubic_constrained_function(x, b, c)

    mask = mask_for_model(A, B, C, f, outliers_pct)
    return (b, c), mask

# -------------------- PCHIP SPLINE MODEL (flexibilní K) --------------------
PCHIP_INV_GRID = 4000
PCHIP_DEFAULT_K = 4

def pchip_xk(K):
    return np.linspace(0.0, 1.0, int(K), dtype=float)


def params_to_yk_general(z, K):
    z = np.asarray(z, float)
    if z.shape[0] != (K - 1):
        raise ValueError(f"z musí mít délku {K-1} pro K={K}")

    d = z * z
    S = np.sum(d) + 1e-12
    cum = np.cumsum(d) / S

    yk = np.empty(K, float)
    yk[0] = 0.0
    yk[1:] = cum
    yk[-1] = 1.0
    return yk


def make_f_pchip_K(z, K):
    XK = pchip_xk(K)
    yk = params_to_yk_general(z, K)
    interp = PchipInterpolator(XK, yk, extrapolate=True)

    def f(x):
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        return np.clip(interp(x), 0.0, 1.0)

    return f, XK, yk


def invert_monotone_f(f, y, grid_n=PCHIP_INV_GRID):
    y = np.asarray(y, float)
    xg = np.linspace(0.0, 1.0, grid_n)
    yg = f(xg)

    yg = np.maximum.accumulate(yg)

    idx = np.searchsorted(yg, y, side="left")
    idx = np.clip(idx, 1, grid_n - 1)

    x0, x1 = xg[idx - 1], xg[idx]
    y0, y1 = yg[idx - 1], yg[idx]

    t = np.where(np.abs(y1 - y0) > 1e-12, (y - y0) / (y1 - y0), 0.0)
    return x0 + t * (x1 - x0)


def predict_b_pchip_K(z, A, C, K):
    f, _, _ = make_f_pchip_K(z, K)
    target = 0.5 * (f(A) + f(C))
    return invert_monotone_f(f, target)


def residuals_pchip_K(z, A, B, C, K):
    return predict_b_pchip_K(z, A, C, K) - B


def fit_pchip_K(A, B, C, K, robust=False):
    z0 = np.ones(K - 1, float)
    if robust:
        res = least_squares(
            lambda zz: residuals_pchip_K(zz, A, B, C, K),
            x0=z0,
            loss=ROBUST_LOSS,
            f_scale=ROBUST_F_SCALE,
            max_nfev=6000
        )
    else:
        res = least_squares(
            lambda zz: residuals_pchip_K(zz, A, B, C, K),
            x0=z0,
            max_nfev=6000
        )
    return res.x


def fit_pchip_model_K(A, B, C, K, outliers_pct=0.0):
    z = fit_pchip_K(A, B, C, K, robust=False)
    f, _, _ = make_f_pchip_K(z, K)

    if outliers_pct <= 0.0:
        mask = np.ones(len(A), dtype=bool)
        return z, mask

    mask = mask_for_model(A, B, C, f, outliers_pct)

    z = fit_pchip_K(A[mask], B[mask], C[mask], K, robust=False)
    f, _, _ = make_f_pchip_K(z, K)

    mask = mask_for_model(A, B, C, f, outliers_pct)
    return z, mask


def robust_metrics_pchip_K(A, B, C, K, outliers_pct=OUTLIERS_PCT):
    z_final, mask_in = fit_pchip_model_K(A, B, C, K, outliers_pct=outliers_pct)
    f, XK, yk = make_f_pchip_K(z_final, K)

    metrics = compute_beak_error(
        A[mask_in], B[mask_in], C[mask_in],
        f, outliers=False, outliers_pct=outliers_pct
    )

    metrics["n_total"] = len(A)
    metrics["K"] = int(K)
    for i, val in enumerate(yk):
        metrics[f"yk_{i}"] = float(val)
    return metrics


####################################################
# --------- SAME-DATA / FAIR-COMPARISON FITS -------
####################################################
def fit_gamma_same_data(A, B, C):
    gamma = fit_gamma(A, B, C, robust=True)
    f = lambda x: gamma_function(x, gamma)
    return gamma, f


def fit_poly3c_same_data(A, B, C):
    b, c = fit_cubic_constrained(A, B, C, robust=True)
    f = lambda x: cubic_constrained_function(x, b, c)
    return (b, c), f

def fit_pchip_same_data(A, B, C, K=PCHIP_DEFAULT_K):
    z = fit_pchip_K(A, B, C, K, robust=True)
    f, XK, yk = make_f_pchip_K(z, K)
    return z, f, XK, yk


def same_data_metrics(A, B, C, function):
    return compute_beak_error(
        A, B, C,
        function,
        outliers=True
    )


####################################################
# --------------------BEAK PLOTS--------------------
####################################################
def beak_plot_models_for_glyph(
    glyph,
    glyph_types,
    A, B, C,
    outliers_pct=0.0,
    plot_outliers=False,
    outdir=IMAGES_OUTPUT_DIR,
    plot_mask_pct=OUTLIERS_PCT
):
    m = glyph_types == glyph
    A_g, B_g, C_g = A[m], B[m], C[m]

    mask_pct = plot_mask_pct if plot_outliers else outliers_pct

    models = []

    # Linear
    f_lin = lambda x: x
    mask_lin = np.ones(len(A_g), bool) if mask_pct <= 0 else \
        mask_for_model(A_g, B_g, C_g, f_lin, mask_pct)

    models.append((
        "linear",
        "Linear, y=x",
        f_lin,
        mask_lin
    ))

    # Gamma
    gamma, _ = fit_gamma_model(A_g, B_g, C_g, outliers_pct)
    f_gam = lambda x: gamma_function(x, gamma)
    mask_gam = mask_for_model(A_g, B_g, C_g, f_gam, mask_pct)

    models.append((
        "gamma",
        f"Gamma, y=x^γ, γ={gamma:.3f}",
        f_gam,
        mask_gam
    ))

    # Poly3 constrained
    (b, c), _ = fit_poly3c_model(A_g, B_g, C_g, outliers_pct)
    f_cc = lambda x: cubic_constrained_function(x, b, c)
    mask_cc = mask_for_model(A_g, B_g, C_g, f_cc, mask_pct)

    models.append((
        "poly3c",
        f"Poly3 constrained, b={b:.3f}, c={c:.3f}",
        f_cc,
        mask_cc
    ))

    # Spline K=4
    K0 = PCHIP_DEFAULT_K
    z0, _ = fit_pchip_model_K(A_g, B_g, C_g, K0, outliers_pct=outliers_pct)
    f_p0, XK0, yk0 = make_f_pchip_K(z0, K0)
    mask_p0 = mask_for_model(A_g, B_g, C_g, f_p0, mask_pct)

    models.append((
        f"spline_K{K0}",
        f"Spline (K={K0}), params={K0-1}",
        f_p0,
        mask_p0
    ))

    # Spline demo K=9, K=15
    for K in [9, 15]:
        zK, _ = fit_pchip_model_K(A_g, B_g, C_g, K, outliers_pct=outliers_pct)
        fK, XK, yk = make_f_pchip_K(zK, K)
        maskK = mask_for_model(A_g, B_g, C_g, fK, mask_pct)

        models.append((
            f"spline_K{K}",
            f"Spline (K={K}), params={K-1}",
            fK,
            maskK
        ))

    ensure_dir(outdir)

    fname = "with_outliers" if plot_outliers else "no_outliers"

    for model_name, title, function, mask in models:
        fig, ax = plt.subplots(figsize=(8, 6))

        beak_plot_for_glyph(
            function,
            A_g,
            B_g,
            C_g,
            title=title,
            axis=ax,
            mask=mask,
            plot_outliers=plot_outliers
        )

        fig.suptitle(
            f"{glyph.capitalize().replace('_', ' ')} — {title}",
            fontsize=14
        )

        plt.tight_layout()

        output_path = os.path.join(
            outdir,
            f"beak_{fname}_{glyph}_{model_name}.png"
        )

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[ok] saved {output_path}")


def beak_plot_same_data_for_glyph(
    glyph,
    glyph_types,
    A, B, C,
    outdir=SAME_DATA_OUTPUT_DIR
):
    m = glyph_types == glyph
    A_g, B_g, C_g = A[m], B[m], C[m]

    f_lin = lambda x: x
    gamma, f_gam = fit_gamma_same_data(A_g, B_g, C_g)
    (b, c), f_cc = fit_poly3c_same_data(A_g, B_g, C_g)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle(
        f"Same-data comparison for glyph: {glyph.capitalize().replace('_', ' ')}",
        fontsize=16
    )

    full_mask = np.ones(len(A_g), dtype=bool)

    beak_plot_for_glyph(f_lin, A_g, B_g, C_g, "Linear", axes[0, 0], mask=full_mask, plot_outliers=False)
    beak_plot_for_glyph(f_gam, A_g, B_g, C_g, f"Gamma, γ={gamma:.3f}", axes[0, 1], mask=full_mask, plot_outliers=False)
    beak_plot_for_glyph(f_cc, A_g, B_g, C_g, f"Poly3c, b={b:.3f}, c={c:.3f}", axes[1, 0], mask=full_mask, plot_outliers=False)

    plt.tight_layout()
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, f"same_data_beak_{glyph}.png"), dpi=120)
    plt.close()


####################################################
# --------------------METRICS-----------------------
####################################################
def robust_metrics_gamma(A, B, C, outliers_pct=OUTLIERS_PCT):
    gamma_final, mask_in = fit_gamma_model(A, B, C, outliers_pct=outliers_pct)
    f = lambda x: gamma_function(x, gamma_final)

    metrics = compute_beak_error(
        A[mask_in], B[mask_in], C[mask_in],
        f,
        outliers=False,
        outliers_pct=OUTLIERS_PCT
    )

    metrics["gamma"] = gamma_final
    metrics["n_total"] = len(A)
    return metrics


def robust_metrics_poly3c(A, B, C, outliers_pct=OUTLIERS_PCT):
    (b_final, c_final), mask_in = fit_poly3c_model(
        A, B, C, outliers_pct=outliers_pct
    )

    f = lambda x: cubic_constrained_function(x, b_final, c_final)

    metrics = compute_beak_error(
        A[mask_in],
        B[mask_in],
        C[mask_in],
        f,
        outliers=False,
        outliers_pct=OUTLIERS_PCT
    )

    metrics["b"] = b_final
    metrics["c"] = c_final
    metrics["n_total"] = len(A)
    return metrics


####################################################
# --------------------CSV EXPORTS-------------------
####################################################
def save_same_data_comparison_csv(glyph_types, sizeA, sizeB, sizeC):
    ensure_dir(SAME_DATA_OUTPUT_DIR)

    with open(SAME_DATA_OUTPUT_FILE_CSV, "w", encoding="utf-8") as f:
        f.write("glyph_type,model,unsigned_euclidean_sum,signed_euclidean_sum,n_points,avg_unsigned,avg_signed,param_1,param_2,param_3\n")

        for g in np.unique(glyph_types):
            mask_g = (glyph_types == g)
            A_g = sizeA[mask_g]
            B_g = sizeB[mask_g]
            C_g = sizeC[mask_g]

            f_lin = lambda x: x
            metrics_lin = same_data_metrics(A_g, B_g, C_g, f_lin)
            f.write(f"{g},linear,{metrics_lin['unsigned_euclidean_sum']:.3f},{metrics_lin['signed_euclidean_sum']:.3f},{metrics_lin['n']},{metrics_lin['avg_unsigned']:.5f},{metrics_lin['avg_signed']:.5f},,,\n")

            gamma, f_gam = fit_gamma_same_data(A_g, B_g, C_g)
            metrics_gam = same_data_metrics(A_g, B_g, C_g, f_gam)
            f.write(f"{g},gamma,{metrics_gam['unsigned_euclidean_sum']:.3f},{metrics_gam['signed_euclidean_sum']:.3f},{metrics_gam['n']},{metrics_gam['avg_unsigned']:.5f},{metrics_gam['avg_signed']:.5f},{gamma:.6f},,\n")

            (b, c), f_cc = fit_poly3c_same_data(A_g, B_g, C_g)
            metrics_cc = same_data_metrics(A_g, B_g, C_g, f_cc)
            f.write(f"{g},poly3c,{metrics_cc['unsigned_euclidean_sum']:.3f},{metrics_cc['signed_euclidean_sum']:.3f},{metrics_cc['n']},{metrics_cc['avg_unsigned']:.5f},{metrics_cc['avg_signed']:.5f},{b:.6f},{c:.6f},\n")

            z, f_p, XK, yk = fit_pchip_same_data(A_g, B_g, C_g, K=PCHIP_DEFAULT_K)
            metrics_p = same_data_metrics(A_g, B_g, C_g, f_p)
            y1 = yk[1] if len(yk) > 1 else np.nan
            y2 = yk[2] if len(yk) > 2 else np.nan
            f.write(f"{g},pchip,{metrics_p['unsigned_euclidean_sum']:.3f},{metrics_p['signed_euclidean_sum']:.3f},{metrics_p['n']},{metrics_p['avg_unsigned']:.5f},{metrics_p['avg_signed']:.5f},{y1:.6f},{y2:.6f},{len(z)}\n")

    print(f"[ok] Results saved to {SAME_DATA_OUTPUT_FILE_CSV}")


####################################################
# ----------------GLYPH DEVIATIONS------------------
####################################################
def glyph_deviations(glyph_types, sizeA, sizeB, sizeC, outliers_pct=OUTLIERS_PCT, filter_outliers=True):
    df = pd.DataFrame({
        'glyph_type': glyph_types,
        'sizeA': sizeA,
        'sizeB': sizeB,
        'sizeC': sizeC,
    })
    df['arith_mean'] = (df['sizeA'] + df['sizeC']) / 2
    df['geom_mean']  = np.sqrt(df['sizeA'] * df['sizeC'])
    df['diff_arith'] = df['sizeB'] - df['arith_mean']
    df['diff_geom']  = df['sizeB'] - df['geom_mean']

    def remove_outliers_by_deviation(group, pct):
        threshold = np.percentile(np.abs(group['diff_arith']), 100 - pct)
        return group[np.abs(group['diff_arith']) <= threshold]

    if filter_outliers:
        parts = []
        for glyph, group in df.groupby('glyph_type'):
            parts.append(remove_outliers_by_deviation(group, outliers_pct))
        df_filtered = pd.concat(parts).reset_index(drop=True)
    else:
        df_filtered = df.copy()

    n_before = len(df)
    n_after = len(df_filtered)
    if filter_outliers:
        print(f"\n[info] Odstraneno {n_before - n_after} outliers "
              f"({outliers_pct} % per glyph), "
              f"zbývá {n_after}/{n_before} odpovědí")

    summary = df_filtered.groupby('glyph_type').agg(
        n=('sizeB', 'count'),
        prumer_odchylky_arith=('diff_arith', 'mean'),
        std_arith=('diff_arith', 'std'),
        prumer_odchylky_geom=('diff_geom', 'mean'),
        std_geom=('diff_geom', 'std'),
    ).round(4)

    print("\n=== Odchylky od aritmetickeho a geometrickeho stredu ===")
    print(f"{'Glyph':<25} {'N':<6} {'avg_arith':<12} {'std_arith':<12} "
          f"{'avg_geom':<12} {'std_geom':<12}")
    print("-" * 80)
    for name, row in summary.iterrows():
        print(f"{name:<25} {int(row['n']):<6} {row['prumer_odchylky_arith']:<12.4f} "
              f"{row['std_arith']:<12.4f} {row['prumer_odchylky_geom']:<12.4f} "
              f"{row['std_geom']:<12.4f}")

    glyph_list = df_filtered['glyph_type'].value_counts().index.tolist()
    ncols = 3
    nrows = int(np.ceil(len(glyph_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, glyph in enumerate(glyph_list):
        ax = axes[i]
        subset_all = df[df['glyph_type'] == glyph]
        subset_filtered = df_filtered[df_filtered['glyph_type'] == glyph]
        diff = subset_filtered['diff_arith']
        mean_diff = diff.mean()
        std_diff = diff.std()
        n_all = len(subset_all)
        n_in = len(subset_filtered)

        ax.hist(diff, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Aritmetický střed')
        ax.axvline(mean_diff, color='tomato', linestyle='-', linewidth=1.8, label=f'Průměr: {mean_diff:.3f}')
        ax.set_title(f'{glyph}\n(n={n_in}/{n_all}, σ={std_diff:.3f})', fontsize=11)
        ax.set_xlabel('Odchylka od aritmetického středu', fontsize=9)
        ax.set_ylabel('Počet odpovědí', fontsize=9)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f'Odchylky odpovědí od aritmetického středu podle typu glyphu '
        f'(odstraněno {outliers_pct} % outlierů)',
        fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    ensure_dir(IMAGES_OUTPUT_DIR)
    plt.savefig(os.path.join(IMAGES_OUTPUT_DIR, 'glyph_deviations.png'),
                bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(IMAGES_OUTPUT_DIR, 'glyph_deviations.pdf'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("[ok] glyph_deviations.png/.pdf")


###############################################
# --------------------MAIN---------------------
###############################################
def main():
    glyph_types, a, b, c = open_data_file()
    print(f"[info] Loaded from {INPUT_FILE}: {len(a)} records")

    sizeA = normalize_data(a)
    sizeB = normalize_data(b)
    sizeC = normalize_data(c)

    if PLOT_GLYPH_DEVIATIONS:
        print("\n[run] Glyph deviations from arithmetic mean")
        glyph_deviations(
            glyph_types, a, b, c,
            outliers_pct=OUTLIERS_PCT,
            filter_outliers=GLYPH_DEVIATION_FILTER_OUTLIERS
        )

    if SAVE_OUTLIER_BEAK_PLOTS:
        print("\n[run] Beak plots with outliers per glyph (model-specific)")
        for g in np.unique(glyph_types):
            beak_plot_models_for_glyph(
                g, glyph_types, sizeA, sizeB, sizeC,
                outliers_pct=0.0,
                plot_outliers=True,
                outdir=IMAGES_OUTPUT_DIR,
                plot_mask_pct=OUTLIERS_PCT
            )

    if SAVE_INLIER_BEAK_PLOTS:
        print("\n[run] Beak plots without outliers per glyph (model-specific)")
        for g in np.unique(glyph_types):
            beak_plot_models_for_glyph(
                g, glyph_types, sizeA, sizeB, sizeC,
                outliers_pct=OUTLIERS_PCT,
                plot_outliers=False,
                outdir=IMAGES_OUTPUT_DIR
            )

    if SAVE_RESULTS_CSV_OUT:
        ensure_dir(OUTLIER_OUTPUT_DIR)

        with open(OUTLIER_OUTPUT_FILE_CSV, "w", encoding="utf-8") as f:
            f.write("glyph_type,model,unsigned_euclidean_sum,signed_euclidean_sum,n_points_all,avg_unsigned,avg_signed\n")

            for g in np.unique(glyph_types):
                mask_g = (glyph_types == g)
                A_g = sizeA[mask_g]
                B_g = sizeB[mask_g]
                C_g = sizeC[mask_g]

                metrics_lin = compute_beak_error(A_g, B_g, C_g, lambda x: x, outliers=True)

                gamma_fit = fit_gamma(A_g, B_g, C_g, robust=False)
                f_gam = lambda x: gamma_function(x, gamma_fit)
                metrics_gam = compute_beak_error(A_g, B_g, C_g, f_gam, outliers=True)

                b_fit, c_fit = fit_cubic_constrained(A_g, B_g, C_g, robust=False)
                f_cc = lambda x: cubic_constrained_function(x, b_fit, c_fit)
                metrics_cc = compute_beak_error(A_g, B_g, C_g, f_cc, outliers=True)

                z_fit = fit_pchip_K(A_g, B_g, C_g, K=PCHIP_DEFAULT_K, robust=False)
                f_p, _, _ = make_f_pchip_K(z_fit, PCHIP_DEFAULT_K)
                metrics_pchip = compute_beak_error(A_g, B_g, C_g, f_p, outliers=True)

                f.write(f"{g},linear,{metrics_lin['unsigned_euclidean_sum']:.3f},{metrics_lin['signed_euclidean_sum']:.3f},{metrics_lin['n']},{metrics_lin['avg_unsigned']:.5f},{metrics_lin['avg_signed']:.5f}\n")
                f.write(f"{g},gamma,{metrics_gam['unsigned_euclidean_sum']:.3f},{metrics_gam['signed_euclidean_sum']:.3f},{metrics_gam['n']},{metrics_gam['avg_unsigned']:.5f},{metrics_gam['avg_signed']:.5f}\n")
                f.write(f"{g},poly3c,{metrics_cc['unsigned_euclidean_sum']:.3f},{metrics_cc['signed_euclidean_sum']:.3f},{metrics_cc['n']},{metrics_cc['avg_unsigned']:.5f},{metrics_cc['avg_signed']:.5f}\n")
                f.write(f"{g},pchip,{metrics_pchip['unsigned_euclidean_sum']:.3f},{metrics_pchip['signed_euclidean_sum']:.3f},{metrics_pchip['n']},{metrics_pchip['avg_unsigned']:.5f},{metrics_pchip['avg_signed']:.5f}\n")

        print(f"[ok] Results saved to {OUTLIER_OUTPUT_FILE_CSV}")

    if SAVE_RESULTS_CSV_IN:
        ensure_dir(INLIER_OUTPUT_DIR)

        with open(INLIER_OUTPUT_FILE_CSV, "w", encoding="utf-8") as f:
            f.write("glyph_type,model,unsigned_euclidean_sum,signed_euclidean_sum,n_points_in,avg_unsigned,avg_signed\n")

            for g in np.unique(glyph_types):
                mask_g = (glyph_types == g)
                A_g = sizeA[mask_g]
                B_g = sizeB[mask_g]
                C_g = sizeC[mask_g]

                metrics_lin = compute_beak_error(
                    A_g, B_g, C_g,
                    lambda x: x,
                    outliers=False,
                    outliers_pct=OUTLIERS_PCT
                )

                gamma_final, mask_in_gam = fit_gamma_model(A_g, B_g, C_g, outliers_pct=OUTLIERS_PCT)
                f_gam = lambda x: gamma_function(x, gamma_final)
                metrics_gam = compute_beak_error(
                    A_g[mask_in_gam], B_g[mask_in_gam], C_g[mask_in_gam],
                    f_gam,
                    outliers=False,
                    outliers_pct=0.0
                )

                (b_final, c_final), mask_in_cc = fit_poly3c_model(A_g, B_g, C_g, outliers_pct=OUTLIERS_PCT)
                f_cc = lambda x: cubic_constrained_function(x, b_final, c_final)
                metrics_cc = compute_beak_error(
                    A_g[mask_in_cc], B_g[mask_in_cc], C_g[mask_in_cc],
                    f_cc,
                    outliers=False,
                    outliers_pct=0.0
                )

                z_final, mask_in_p = fit_pchip_model_K(A_g, B_g, C_g, K=PCHIP_DEFAULT_K, outliers_pct=OUTLIERS_PCT)
                f_p, _, _ = make_f_pchip_K(z_final, PCHIP_DEFAULT_K)
                metrics_pchip = compute_beak_error(
                    A_g[mask_in_p], B_g[mask_in_p], C_g[mask_in_p],
                    f_p,
                    outliers=False,
                    outliers_pct=0.0
                )

                f.write(f"{g},linear,{metrics_lin['unsigned_euclidean_sum']:.3f},{metrics_lin['signed_euclidean_sum']:.3f},{metrics_lin['n']},{metrics_lin['avg_unsigned']:.5f},{metrics_lin['avg_signed']:.5f}\n")
                f.write(f"{g},gamma,{metrics_gam['unsigned_euclidean_sum']:.3f},{metrics_gam['signed_euclidean_sum']:.3f},{metrics_gam['n']},{metrics_gam['avg_unsigned']:.5f},{metrics_gam['avg_signed']:.5f}\n")
                f.write(f"{g},poly3c,{metrics_cc['unsigned_euclidean_sum']:.3f},{metrics_cc['signed_euclidean_sum']:.3f},{metrics_cc['n']},{metrics_cc['avg_unsigned']:.5f},{metrics_cc['avg_signed']:.5f}\n")
                f.write(f"{g},pchip,{metrics_pchip['unsigned_euclidean_sum']:.3f},{metrics_pchip['signed_euclidean_sum']:.3f},{metrics_pchip['n']},{metrics_pchip['avg_unsigned']:.5f},{metrics_pchip['avg_signed']:.5f}\n")

        print(f"[ok] Results saved to {INLIER_OUTPUT_FILE_CSV}")

    if SAVE_RESULTS_CSV_SAME_DATA:
        print("\n[run] Fair model comparison on the same dataset using robust loss")
        save_same_data_comparison_csv(glyph_types, sizeA, sizeB, sizeC)

    if SAVE_SAME_DATA_BEAK_PLOTS:
        print("\n[run] Same-data beak plots per glyph")
        for g in np.unique(glyph_types):
            beak_plot_same_data_for_glyph(g, glyph_types, sizeA, sizeB, sizeC)


if __name__ == "__main__":
    main()
