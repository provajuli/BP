import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy as np
import csv
import os


####################################################
# --------------------PREPINACE---------------------
####################################################
PATH = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(PATH, "input/filtered_results.csv")
IMAGES_OUTPUT_DIR = os.path.join(PATH, "images/unfiltered_outliers")
OUTLIER_OUTPUT_DIR = os.path.join(PATH, "output/unfiltered_outliers")
INLIER_OUTPUT_DIR = os.path.join(PATH, "output/filtered/inliers")

OUTLIER_OUTPUT_FILE_CSV = os.path.join(OUTLIER_OUTPUT_DIR, "model_comparison.csv")

PLOT_GAMMA = True
PLOT_CC = True
SAVE_OUTLIER_BEAK_PLOTS = False
SAVE_INLIER_BEAK_PLOTS = False
OUTLIERS_PCT = 5.0
SAVE_RESULTS_CSV = True
SAVE_RESULTS_TXT = True


GLYPH_TYPES = []
EPS = 1e-9


#-------------------------------------------------------------
def open_data_file(filename = INPUT_FILE):
    A, B, C = [], [], []
    glyph_types = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            glyph_types.append(row["glyph_type"])
            A.append(float(row["sizeA"]))
            B.append(float(row["sizeB"]))
            C.append(float(row["sizeC"]))
    return (np.array(glyph_types), np.array(A), np.array(B), np.array(C))


def normalize_data(x):
    return np.clip(x / 100.0, 0.0, 1.0)
# -------------------------------------------------------------


#############################################################
# --------------------GAMMA MODEL FUNKCE---------------------
#############################################################
GAMMA_MIN = 0.05
GAMMA_MAX = 3.0

def gamma_function(x, gamma):
    x = np.asarray(x, float)
    g = max(float(gamma), 1e-6)
    return np.power(np.clip(x, 0, 1), g)


# (a^gamma + c^gamma / 2)^(1/gamma)
# predikuje percepcni stred 
def predict_b_gamma(gamma, a, c):
    middle = (a**gamma + c**gamma) / 2
    return middle ** (1.0/gamma)


# chyba mezi predikovanym B a skutecnym B
def residuals_gamma(gamma, a, b, c):
    return predict_b_gamma(gamma, a, c) - b 


# metodou nejmensich ctvercu najdeme optimalni gamma
def fit_gamma(a, b, c):
    res = least_squares(residuals_gamma,
                        x0=1.0,
                        bounds=(GAMMA_MIN, GAMMA_MAX),
                        args=(a, b, c))
    return float(res.x[0])


# Vypis pro gamma model pro kazdy typ glyphu
def gamma_stats_per_glyph_type(glyph_types, a, b, c):
    rows = []
    g_all= fit_gamma(a, b, c) # optimalni gamma pro vsechny glyphy
    rows.append(("All", len(b), g_all))

    for t in np.unique(glyph_types):
        idx = (glyph_types == t) # na jakych indexech je dany typ glyphu
        a_t= a[idx]
        b_t= b[idx]
        c_t= c[idx]
        g_t = fit_gamma(a_t, b_t, c_t) # optimalni gamma pro dany typ glyphu
        rows.append((t.capitalize(), len(b_t), g_t)) 
    return rows


# Plot pro gamma model
def plot_gamma_model(rows, outpath):
    x = np.linspace(0,1,200)
    plt.figure(figsize=(8,6))
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
CC_LOW    = np.array([0.2, -1.5], dtype=float)
CC_HIGH   = np.array([1.8,  1.5], dtype=float)


def cubic_constrained_function(x, b_par, c_par):
    x = np.asarray(x, float)
    return b_par*x + c_par*(x**2) + (1.0 - b_par - c_par)*(x**3)


def inv_cubic_constrained(y, b_par, c_par):
    y = np.atleast_1d(np.asarray(y, float)).ravel()
    out = np.empty_like(y)
    A = (1.0 - b_par - c_par)   # koef. u x^3
    B = c_par                   # u x^2
    C = b_par                   # u x
    def fval(xx): return b_par*xx + c_par*xx**2 + (1.0-b_par-c_par)*xx**3

    for i, yi in enumerate(y):
        roots = np.roots([A, B, C, -yi])
        real = roots[np.isreal(roots)].real
        real = np.asarray(real, float)

        if real.size == 0:
            xi = 0.0
        else:
            in01 = real[(real >= -EPS) & (real <= 1.0 + EPS)]
            if in01.size == 0:
                xi = in01[np.argmin(np.abs(fval(in01) - yi))]
            else:
                clipped = np.clip(real, 0.0, 1.0)
                xi = real[np.argmin(np.abs(real - clipped))]

        out[i] = np.clip(xi, 0.0, 1.0)

    return out.reshape(y.shape)


def predict_b_cubic_constrained(b_par, c_par, A, C):
    mid = 0.5 * (cubic_constrained_function(A, b_par, c_par) 
                + cubic_constrained_function(C, b_par, c_par))
    return inv_cubic_constrained(mid, b_par, c_par)


def residuals_cubic_constrained(theta, A, B, C):
    b_par= float(theta[0]) 
    c_par = float(theta[1])
    return predict_b_cubic_constrained(b_par, c_par, A, C) - B


def fit_cubic_constrained(a, b, c):
    res = least_squares(
        residuals_cubic_constrained,
        x0=CC_THETA0, bounds=(CC_LOW, CC_HIGH),
        args=(np.asarray(a,float),
              np.asarray(b,float),
              np.asarray(c,float)),
        max_nfev=5000
    )
    return res.x  # [b, c]


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
    x = np.linspace(0,1,200)
    plt.figure(figsize=(8,6))
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


################################################################
# --------------------POLY3 MODEL NEOMEZENY---------------------
################################################################
def cubic_function(x, a_par, b_par, c_par, d_par):
    x = np.asarray(x, float)
    return a_par + b_par*x + c_par*(x**2) + d_par*(x**3)


#TODO: fitting neomezeneho poly3


#####################################################
# --------------------BEAK PLOTS---------------------
#####################################################
def linear_function(x):
    return np.asarray(x, float)


# bude vracet seznam bodu
def beak_points(A, B, C, function, return_all = False):
    fA = function(A)
    fC = function(C)
    beak_y = (fA + fC) / 2
    beak_x = B
    if return_all:
        return beak_x, beak_y, fA, fC
    return beak_x, beak_y


# vykresli jakoukoliv funkci podle zadaneho modelu
def plot_curve(function):
    x = np.linspace(0, 1, 1000)
    y = function(x)
    return x, y


# pro kazdy bod B najde nejblizsi bod na krivce
# suma nad krivkou a pod krivkou by se mela blizit 0
def euclidean_distance_from_curve(beak_x, beak_y, curve_x, curve_y, select_signed = False):
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
        if j-1 >= 0:  
            cand.append(j-1)

        best = None
        best_sign = 1
        for k in cand:
            dx = xb - curve_x[k]
            dy = yb - curve_y[k]
            d = np.sqrt(dx*dx + dy*dy)

            if select_signed:
                if dy >= 0:
                    sign = 1
                else:
                    sign = -1

            if (best is None) or (d < best):
                best = d
                if select_signed:
                    best_sign = sign

        distances[i] = 0.0 if best is None else best * (best_sign if select_signed else 1)
    return distances


# nejvyssi hodnoty budu mazat - nejvetsi euklid vzdalenost
# potrebuji predat np.array euklidovskych unsigned vzdalenosti a procento outlieru k odstraneni
# vzdalenosti seradim od nejmensi po nejvetsi 
# KEEP = 1 - (OUTLIERS_PCT / 100)
# posledni index ponechaneho bodu by mel byt int(KEEP * len(distances))
# udelam si masku pro vybrane body a vratim ji
# tuhle masku muzu pouzit i pri vykresleni zobackuuu
# TODO: asi bude lepsi mit jako parametry funkce body, ze kterych se pocitaji vzdalenosti??? dunno
# vstup bude function, x np.array 1000 dilku, beak_x (B) a beak_y (f(A) + f(C)) / 2
def inliers_mask(distances, outliers_pct = OUTLIERS_PCT):
    # kdyz budu mit OUTLIERS_PCT 0.0, chci vratit vsechny body
    # mask bude mit delku len(distances) a vsechny hodnoty True
    n_points = len(distances)
    
    if outliers_pct <= 0.0:
        mask = np.ones(n_points, dtype=bool) # vsechny body jsou inliers, vraci pole True o delce n_points
        sorted_idx = np.arange(n_points)  # vracim serazene indexy - nic s nima nedelam (0, 1, 2, ..., n_points-1)
        return mask, sorted_idx

    keep = 1.0 - (outliers_pct / 100.0)

    n_keep = int(keep * n_points) # "pretypovani" na int automaticky orizne

    sorted_idx = np.argsort(distances)
    inlier_slice = sorted_idx[:n_keep] # do indexu, ktery odpovida procento * delka si to necham

    mask = np.zeros(n_points, dtype=bool) # celou masku nastavim na False
    mask[inlier_slice] = True # na indexech, ktere si chci nechat, dam True

    return mask, sorted_idx # vraci masku a poradi indexu podle velikosti vzdalenosti


def filter_outliers(A, B, C, function, outliers_pct=OUTLIERS_PCT):
    beak_x, beak_y = beak_points(A, B, C, function, return_all=False)

    x = np.linspace(0, 1, 1000)
    y = function(x)

    distances = euclidean_distance_from_curve(beak_x, beak_y, x, y, select_signed=False)
    mask_inliers, _ = inliers_mask(distances, outliers_pct=outliers_pct)

    # dostanu pouze body z jednoho trial, ktere nejsou outliery
    # mask_inliers je maska - na indexech True jsou data vyhovujici podmince
    return A[mask_inliers], B[mask_inliers], C[mask_inliers], mask_inliers


def compute_beak_error_outliers(A, B, C, function, outliers_pct=OUTLIERS_PCT):
    # beak_x je hodnota B od uzivatele, beak_y je ( f(A) + f(C) ) / 2
    beak_x, beak_y = beak_points(A, B, C, function, return_all=False)

    x = np.linspace(0, 1, 1000)
    y = function(x)

    unsigned_euclidean_distances = euclidean_distance_from_curve(beak_x, beak_y, x, y, select_signed=False)
    signed_euclidean_distances = euclidean_distance_from_curve(beak_x, beak_y, x, y, select_signed=True)

    n_all = int(len(beak_x))

    unsigned_euclidean_sum=np.sum(unsigned_euclidean_distances)
    signed_euclidean_sum=np.sum(signed_euclidean_distances)

    return dict(
        unsigned_euclidean_sum=unsigned_euclidean_sum,
        signed_euclidean_sum=signed_euclidean_sum,
        n_points_all = n_all,
        avg_unsigned=unsigned_euclidean_sum / n_all,
        avg_signed=signed_euclidean_sum / n_all
    )


def beak_plot_for_glyph(function, A, B, C, title, axis, mask=None):
    # krivka modelu
    x = np.linspace(0, 1, 1000)
    y = function(x)

    axis.plot(x, y, lw=2, label="model")

    # zobacky 🐔 hihi
    beak_x, beak_y, fA, fC = beak_points(A, B, C, function, return_all=True)

    if mask is None:
        mask = np.ones(len(beak_x), dtype=bool)

    # chci vykreslit jen body, ktere jsou inliers
    A_in = A[mask]
    C_in = C[mask]
    beak_x_in = beak_x[mask]
    beak_y_in = beak_y[mask]
    fA_in = fA[mask]
    fC_in = fC[mask]

    # outliers vykreslim cervene
    A_out = A[~mask]
    C_out = C[~mask]
    beak_x_out = beak_x[~mask]
    beak_y_out = beak_y[~mask]
    fA_out = fA[~mask]
    fC_out = fC[~mask]

    # vykreslit outliers
    for Ai, Bi, Ci, yAi, yBi, yCi in zip(A_out, beak_x_out, C_out, fA_out, beak_y_out, fC_out):
        axis.plot([Ai, Bi], [yAi, yBi], color='red', linewidth=0.3, alpha=0.4)
        axis.plot([Ci, Bi], [yCi, yBi], color='red', linewidth=0.3, alpha=0.4)
        axis.scatter([Ai, Ci], [yAi, yCi], color='darkorange', s=8, alpha=0.4)
        axis.scatter([Bi], [yBi], color='red', s=14, alpha=0.8)

    # vykreslit inliers
    for Ai, Bi, Ci, yAi, yBi, yCi in zip(A_in, beak_x_in, C_in, fA_in, beak_y_in, fC_in):
        axis.plot([Ai, Bi], [yAi, yBi], color='gray', linewidth=0.3)
        axis.plot([Ci, Bi], [yCi, yBi], color='gray', linewidth=0.3)
        axis.scatter([Ai, Ci], [yAi, yCi], color='black', s=8)
        axis.scatter([Bi], [yBi], color='blue', s=14)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("x (normalized)")
    axis.set_ylabel("y (perceived)")
    axis.set_title(title)
    axis.grid(linestyle='--', alpha=0.3)


def beak_plots_all_models_for_glyph_outliers(glyph, glyph_types, A, B, C, outdir=IMAGES_OUTPUT_DIR):
    glyph_mask = (glyph_types == glyph)
    A_glyph, B_glyph, C_glyph = A[glyph_mask], B[glyph_mask], C[glyph_mask]

    # odstranim outliers na zaklade linearniho modelu 
    A_in, B_in, C_in, mask_inliers = filter_outliers(
        A_glyph, B_glyph, C_glyph,
        lambda x: x,             
        OUTLIERS_PCT
    )

    # do modelu posilam jen inliers
    g_fit = fit_gamma(A_in, B_in, C_in)
    b_fit, c_fit = fit_cubic_constrained(A_in, B_in, C_in)

    fig, axes = plt.subplots(1, 3, figsize=(18,6), sharex=True, sharey=True)

    # plotu musim hodit normalni A, B, C a masku inliers -> vykresli mi jen inliers
    beak_plot_for_glyph(
        lambda x: x,
        A_glyph, B_glyph, C_glyph,
        title=f"Beak — Linear (y=x) — {glyph}",
        axis=axes[0],
        mask=mask_inliers
    )

    beak_plot_for_glyph(
        lambda x: gamma_function(x, g_fit),
        A_glyph, B_glyph, C_glyph,
        title=f"Beak — Gamma (gamma={g_fit:.3f}) — {glyph}",
        axis=axes[1],
        mask=mask_inliers
    )

    beak_plot_for_glyph(
        lambda x: cubic_constrained_function(x, b_fit, c_fit),
        A_glyph, B_glyph, C_glyph,
        title=f"Beak — Poly3C (b={b_fit:.3f}, c={c_fit:.3f}) — {glyph}",
        axis=axes[2],
        mask=mask_inliers
    )

    fig.suptitle(f"Beak plots for glyph '{glyph}'", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(outdir, f"outlier_beak_plots_{glyph}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {glyph}: saved outlier_beak_plots_{glyph}.png")


####################################################
#                    INLIERS
####################################################
# 1. vytvorim si masku pro dany model
# 2. nafituju parametry na body bez outliers
#   a. fitnu na vsechny body
#   b. udelam euclidean metrics
#   c. podle masky vyhodim outliers
#   d. vratim params nove nafitovane a masku?
# 3. Nafituju nanovo jen s inliers
# 4. chci vypsat i errors pro jen inlier data
# 5. plot jen pro inliers
# bude pocitat masku pro dany model 
def mask_for_model(A, B, C, function, outliers=OUTLIERS_PCT):
    pass


def fit_gamma_no_outliers(A, B, C, outliers=OUTLIERS_PCT):
    pass


def fit_poly3c_no_outliers(A, B, C, outliers=OUTLIERS_PCT):
    pass

###############################################
# --------------------MAIN---------------------
###############################################
def main():
    glyph_types, a, b, c = open_data_file()
    print(f"[info] Loaded from {INPUT_FILE}: {len(a)} records")

    sizeA = normalize_data(a)
    sizeB = normalize_data(b)
    sizeC = normalize_data(c)

    if PLOT_GAMMA:
        print("\n[run] Gamma model… (ALL + per glyph)")
        rows_g = gamma_stats_per_glyph_type(glyph_types, sizeA, sizeB, sizeC)
        print("\nExponential (x^gamma) model")
        print(f"{'Glyph':<25} {'N':<5} {'Gamma':<10}")
        print("-"*60)
        for name, n, g in rows_g:
            print(f"{name:<25} {n:<5} {g:<10.5f}")
        out = os.path.join(IMAGES_OUTPUT_DIR, "gamma_curves.png")
        if not os.path.exists(IMAGES_OUTPUT_DIR):
            os.makedirs(IMAGES_OUTPUT_DIR)
        plot_gamma_model(rows_g, out)
        print("[ok] gamma_curves.png")


    if PLOT_CC:
        print("\n[run] Cubic constrained model… (ALL + per glyph)")
        rows_cc = cubic_constrained_stats_by_glyph(glyph_types, sizeA, sizeB, sizeC)
        print("\n3rd grade polynomial model")
        print(f"{'Glyph':<25} {'N':<5} {'a':<10} {'b':<10} {'c':<10} {'d':<10}")
        print("-"*90)
        for name, n, a_, b_, c_, d_ in rows_cc:
            print(f"{name:<25} {n:<5} {a_:<10.5f} {b_:<10.5f} {c_:<10.5f} {d_:<10.5f}")
        out = os.path.join(IMAGES_OUTPUT_DIR, "cubic_constrained_curves.png")
        if not os.path.exists(IMAGES_OUTPUT_DIR):
            os.makedirs(IMAGES_OUTPUT_DIR)
        plot_cubic_constrained_model(rows_cc, out)
        print("[ok] cubic_constrained_curves.png")


    if SAVE_OUTLIER_BEAK_PLOTS:
        print("\n[run] Beak plots with outliers per glyph (linear/gamma/poly3c)")
        for g in np.unique(glyph_types):
            beak_plots_all_models_for_glyph_outliers(g, glyph_types, sizeA, sizeB, sizeC, outdir=IMAGES_OUTPUT_DIR)


    if SAVE_INLIER_BEAK_PLOTS:
        print("\n[run] Beak plots with outliers per glyph (linear/gamma/poly3c)")
        for g in np.unique(glyph_types):
            beak_plots_all_models_for_glyph_outliers(g, glyph_types, sizeA, sizeB, sizeC, outdir=IMAGES_OUTPUT_DIR)


    if SAVE_RESULTS_CSV:
        if not os.path.exists(OUTLIER_OUTPUT_DIR):
            os.makedirs(OUTLIER_OUTPUT_DIR)
        with open(OUTLIER_OUTPUT_FILE_CSV, "w", encoding="utf-8") as f:
            f.write("glyph_type,model,unsigned_euclidean_sum,signed_euclidean_sum,n_points_all,avg_unsigned,avg_signed\n")
            for g in np.unique(glyph_types):
                metrics_lin = compute_beak_error_outliers(sizeA[glyph_types==g], sizeB[glyph_types==g], sizeC[glyph_types==g], lambda x: x)
                g_fit = fit_gamma(sizeA[glyph_types==g], sizeB[glyph_types==g], sizeC[glyph_types==g])
                metrics_gam = compute_beak_error_outliers(sizeA[glyph_types==g], sizeB[glyph_types==g], sizeC[glyph_types==g], lambda x: gamma_function(x, g_fit))
                b_fit, c_fit = fit_cubic_constrained(sizeA[glyph_types==g], sizeB[glyph_types==g], sizeC[glyph_types==g])
                metrics_cc = compute_beak_error_outliers(sizeA[glyph_types==g], sizeB[glyph_types==g], sizeC[glyph_types==g], lambda x: cubic_constrained_function(x, b_fit, c_fit))

                f.write(f"{g},linear,{metrics_lin['unsigned_euclidean_sum']:.3f},{metrics_lin['signed_euclidean_sum']:.3f},{metrics_lin['n_points_all']},{metrics_lin['avg_unsigned']:.5f},{metrics_lin['avg_signed']:.5f}\n")
                f.write(f"{g},gamma,{metrics_gam['unsigned_euclidean_sum']:.3f},{metrics_gam['signed_euclidean_sum']:.3f},{metrics_lin['n_points_all']},{metrics_gam['avg_unsigned']:.5f},{metrics_gam['avg_signed']:.5f}\n")
                f.write(f"{g},poly3c,{metrics_cc['unsigned_euclidean_sum']:.3f},{metrics_cc['signed_euclidean_sum']:.3f},{metrics_lin['n_points_all']},{metrics_cc['avg_unsigned']:.5f},{metrics_cc['avg_signed']:.5f}\n")
        print(f"[ok] Results saved to {OUTLIER_OUTPUT_FILE_CSV}")


if __name__ == "__main__":
    main()