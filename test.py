import numpy as np

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
                sign = 1 if dy >= 0 else -1

            if (best is None) or (d < best):
                best = d
                if select_signed:
                    best_sign = sign

        distances[i] = 0.0 if best is None else best * (best_sign if select_signed else 1)
    return distances


# ------------------------------
# SLOŽITĚJŠÍ TEST: y = sin(x)
# ------------------------------

# Křivka: y = sin(x) na intervalu 0..2π
curve_x = np.linspace(0, 2*np.pi, 50)
curve_y = np.sin(curve_x)

# Body kolem křivky:
#  - jeden trochu nad sinem,
#  - jeden pod sinem,
#  - jeden mezi vzorky, kde je křivka zakřivená,
#  - jeden přesně na křivce
beak_x = [
    np.pi/6,          # ~0.52
    np.pi/2,          # ~1.57
    3*np.pi/4,        # ~2.36
    np.pi             # ~3.14
]
beak_y = [
    np.sin(np.pi/6) + 0.2,   # trochu nad křivkou
    np.sin(np.pi/2) - 0.3,   # trochu pod křivkou (sin=1)
    0.0,                     # někde kolem průsečíku se 0
    np.sin(np.pi)            # přesně na křivce (0)
]

d_points = euclidean_distance_from_curve(beak_x, beak_y, curve_x, curve_y, select_signed=False)

print("Beak X:", np.round(beak_x, 3))
print("Beak Y:", np.round(beak_y, 3))
print("Vzdálenost k nejbližšímu VZORKU:   ", np.round(d_points, 4))
