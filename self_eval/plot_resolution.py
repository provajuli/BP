import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


FILES = [
    "experiment_gamma_set_answers.json",
    "experiment_set_answers.json",
]

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
print("Data path:", PATH)

def load_glyph_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: root JSON není dict")

    if "answers" not in data or not isinstance(data["answers"], list):
        raise ValueError(f"{path}: chybí pole 'answers'")

    glyph_id = data.get("id", os.path.splitext(os.path.basename(path))[0])
    glyph_name = data.get("name", glyph_id)

    parsed = []
    for ans in data["answers"]:
        if not isinstance(ans, dict):
            continue

        if "val1" not in ans or "val2" not in ans or "correct" not in ans:
            continue

        val1 = float(ans["val1"])
        val2 = float(ans["val2"])
        correct = bool(ans["correct"])

        parsed.append({
            "val1": val1,
            "val2": val2,
            "correct": correct,
            "delta": abs(val1 - val2),
            "time": ans.get("time"),
        })

    if not parsed:
        raise ValueError(f"{path}: v 'answers' nejsou použitelná data")

    return {
        "path": path,
        "id": glyph_id,
        "name": glyph_name,
        "meta": {
            "distance": data.get("distance"),
            "gamma": data.get("gamma"),
            "glyphStepsCount": data.get("glyphStepsCount"),
            "rotation": data.get("rotation"),
        },
        "answers": parsed,
    }


def bin_accuracy(deltas, corrects, bins=12):
    deltas = np.asarray(deltas, dtype=float)
    corrects = np.asarray(corrects, dtype=float)

    if len(deltas) < 2:
        return np.array([]), np.array([]), np.array([])

    edges = np.linspace(deltas.min(), deltas.max(), bins + 1)

    centers = []
    accs = []
    counts = []

    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]

        if i == bins - 1:
            mask = (deltas >= left) & (deltas <= right)
        else:
            mask = (deltas >= left) & (deltas < right)

        n = np.sum(mask)
        if n == 0:
            continue

        centers.append((left + right) / 2)
        accs.append(corrects[mask].mean())
        counts.append(n)

    return np.array(centers), np.array(accs), np.array(counts)


def moving_average_curve(x, y, window=3):
    if len(y) < window:
        return x, y

    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="same")
    return x, y_smooth


def estimate_threshold(x, y, target=0.75):
    if len(x) < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]

        if (y1 <= target <= y2) or (y2 <= target <= y1):
            if y1 == y2:
                return float(x1)
            t = (target - y1) / (y2 - y1)
            return float(x1 + t * (x2 - x1))

    return None


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan

    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom

    return center - margin, center + margin


def analyze_file(path):
    glyph = load_glyph_file(path)
    answers = glyph["answers"]

    deltas = np.array([a["delta"] for a in answers], dtype=float)
    corrects = np.array([a["correct"] for a in answers], dtype=float)

    centers, accs, counts = bin_accuracy(deltas, corrects, bins=12)
    smooth_x, smooth_y = moving_average_curve(centers, accs, window=3)
    threshold = estimate_threshold(smooth_x, smooth_y, target=0.75)

    auc = float(np.trapz(smooth_y, smooth_x)) if len(smooth_x) > 1 else float("nan")

    glyph["analysis"] = {
        "n": len(answers),
        "mean_accuracy": float(corrects.mean()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "threshold_75": threshold,
        "bin_centers": centers,
        "bin_accs": accs,
        "counts": counts,
        "smooth_x": smooth_x,
        "smooth_y": smooth_y,
        "auc": auc,
    }

    return glyph


def plot_results(results):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {
        "sun_set": "#4C72B0",
        "sun_cc_set": "#DD8452",
        "sun": "#4C72B0",
        "sun_cc": "#DD8452",
    }

    fallback_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    all_xticks = []

    for i, glyph in enumerate(results):
        color = palette.get(glyph["id"], fallback_colors[i % len(fallback_colors)])
        a = glyph["analysis"]
        label = glyph["id"]

        x = np.asarray(a["bin_centers"], dtype=float)
        y = np.asarray(a["bin_accs"], dtype=float)
        counts = np.asarray(a["counts"], dtype=int)

        if len(x) == 0:
            continue

        all_xticks.extend(list(x))

        correct_counts = np.rint(y * counts).astype(int)

        lower = []
        upper = []

        for k, n in zip(correct_counts, counts):
            lo, hi = wilson_ci(k, n)
            lower.append(lo)
            upper.append(hi)

        lower = np.asarray(lower)
        upper = np.asarray(upper)

        ax.fill_between(
            x, lower, upper,
            color=color,
            alpha=0.18,
            linewidth=0
        )

        ax.plot(
            x, y,
            "-o",
            color=color,
            linewidth=2.2,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label
        )

        thr = a.get("threshold_75")
        if thr is not None:
            ax.axvline(
                thr,
                color=color,
                linestyle=":",
                linewidth=2.2
            )

    ax.axhline(
        0.75,
        color="gray",
        linestyle="--",
        linewidth=1.4
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)

    # vlastní tick positions, aby nebylo 10^1
    if all_xticks:
        xticks = sorted(set(round(t, 3) for t in all_xticks))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:.2f}" for t in xticks], rotation=0)

    ax.set_xlabel("|val1 - val2|", fontsize=13)
    ax.set_ylabel("accuracy", fontsize=13)
    ax.set_title("Glyph discrimination threshold", fontsize=15)

    ax.set_ylim(-0.05, 1.05)

    ax.grid(True, which="major", color="#b0b0b0", alpha=0.6, linewidth=0.7)
    ax.grid(False, which="minor")

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()


def main():
    results = []

    for filename in FILES:
        print(f"Found file: {filename}")
        full_path = os.path.join(PATH, filename)
        print(f"Processing {full_path}...")

        if not os.path.exists(full_path):
            print(f"Soubor neexistuje: {full_path}")
            continue

        try:
            glyph = analyze_file(full_path)
            results.append(glyph)

            a = glyph["analysis"]
            print(f"\n=== {glyph['id']} ===")
            print(f"Název: {glyph['name']}")
            print(f"Počet odpovědí: {a['n']}")
            print(f"Průměrná úspěšnost: {a['mean_accuracy']:.3f}")
            print(f"Průměrné |val1-val2|: {a['mean_delta']:.3f}")
            print(f"Medián |val1-val2|: {a['median_delta']:.3f}")

            if a["threshold_75"] is not None:
                print(f"Odhad rozlišení (75 % správnosti): {a['threshold_75']:.3f}")
            else:
                print("Odhad rozlišení (75 % správnosti): nepodařilo se určit")

        except Exception as e:
            print(f"Chyba při zpracování {filename}: {e}")

    if not results:
        print("Nepodařilo se načíst žádná data.")
        return

    plot_results(results)


if __name__ == "__main__":
    main()