import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Nastavení
# =========================
BOOTSTRAP_SAMPLES = 2000

# Náhodné tipování při 3 možnostech: <, =, >
CHANCE_LEVEL = 1 / 3

# Threshold accuracy
TARGET_ACCURACY = (1 + CHANCE_LEVEL) / 2   # 0.666666...

# Přepínač osy x
USE_LOG_X = True   # True = podobnější paperu, False = lineární osa


# =========================
# Načtení dat
# =========================
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Výpočet accuracy po distance
# =========================
def group_answers_by_distance(data):
    grouped = defaultdict(list)

    for answer in data["answers"]:
        # rozdíl hodnot na ose x, analogicky k paperu
        d = round(abs(float(answer["val1"]) - float(answer["val2"])), 2)
        correct = bool(answer["correct"])
        grouped[d].append(correct)

    return grouped


def compute_accuracy_with_ci(data, n_bootstrap=2000, ci=(2.5, 97.5)):
    grouped = group_answers_by_distance(data)

    distances = []
    accuracies = []
    lowers = []
    uppers = []
    counts = []

    for d in sorted(grouped.keys()):
        vals = np.array(grouped[d], dtype=float)
        acc = np.mean(vals)

        boot = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(vals, size=len(vals), replace=True)
            boot.append(np.mean(sample))

        low, high = np.percentile(boot, ci)

        distances.append(d)
        accuracies.append(acc)
        lowers.append(low)
        uppers.append(high)
        counts.append(len(vals))

    return {
        "distances": np.array(distances, dtype=float),
        "accuracies": np.array(accuracies, dtype=float),
        "lower": np.array(lowers, dtype=float),
        "upper": np.array(uppers, dtype=float),
        "counts": np.array(counts, dtype=int),
    }


# =========================
# AUC
# =========================
def auc_linear(x, y):
    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]
    return np.trapezoid(y, x)


def auc_logx(x, y):
    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]

    if np.any(x <= 0):
        raise ValueError("Pro log-AUC musí být všechny distance > 0.")

    return np.trapezoid(y, np.log(x))


# =========================
# Threshold
# =========================
def find_resolution_threshold(distances, accuracies, target=2/3):
    order = np.argsort(distances)
    x = np.array(distances)[order]
    y = np.array(accuracies)[order]

    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]

        if y1 == target:
            return x1

        crosses = (y1 - target) * (y2 - target) < 0
        if crosses:
            t = (target - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)

    return None


# =========================
# Diagnostika
# =========================
def print_summary(name, stats, threshold):
    x = stats["distances"]
    y = stats["accuracies"]
    n = stats["counts"]

    print(f"\n=== {name} ===")
    print("Distances:", x)
    print("Accuracy :", np.round(y, 3))
    print("Counts    :", n)
    print(f"AUC (linear): {auc_linear(x, y):.4f}")

    x_pos = x[x > 0]
    y_pos = y[x > 0]
    if len(x_pos) >= 2:
        print(f"AUC (log-x) : {auc_logx(x_pos, y_pos):.4f}")
    else:
        print("AUC (log-x) : nelze spočítat")

    print(f"Resolution threshold @ accuracy={TARGET_ACCURACY:.3f}: {threshold}")


# =========================
# Vykreslení
# =========================
def plot_results(results, target_accuracy=2/3, use_log_x=USE_LOG_X):
    plt.figure(figsize=(11, 6.5))

    all_x_for_ticks = []
    all_x_positive = []

    for name, stats in results.items():
        x = stats["distances"]
        y = stats["accuracies"]
        low = stats["lower"]
        high = stats["upper"]
        thr = stats["threshold"]


        if use_log_x:
            mask = x > 0
            x_plot = x[mask]
            y_plot = y[mask]
            low_plot = low[mask]
            high_plot = high[mask]
        else:
            x_plot = x
            y_plot = y
            low_plot = low
            high_plot = high

        all_x_for_ticks.extend(list(x_plot))
        all_x_positive.extend([v for v in x_plot if v > 0])

        label = f"{name} (AUC={auc_linear(x, y):.3f})"

        plt.plot(x_plot, y_plot, marker="o", linewidth=2, label=label, alpha=0.5)
        plt.fill_between(x_plot, low_plot, high_plot, alpha=0.20)

        if thr is not None:
            if (not use_log_x) or (thr > 0):
                plt.axvline(thr, linestyle=":", linewidth=1.8)

    #plt.axhline(CHANCE_LEVEL, linestyle="--", linewidth=1.2, label=f"chance = {CHANCE_LEVEL:.3f}")
    #plt.axhline(target_accuracy, linestyle="--", linewidth=1.2, label=f"threshold = {target_accuracy:.3f}")

    if use_log_x:
        plt.xscale("log")

        # unikátní tick hodnoty z dat
        tick_values = sorted(set(round(v, 2) for v in all_x_for_ticks if v > 0))
        plt.xticks(tick_values, [str(v) for v in tick_values], rotation=45)

    plt.xlabel("difference |val1 - val2|")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.minorticks_off()
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if all_x_positive:
        plt.xlim(min(all_x_positive), max(all_x_positive))
    plt.savefig("self_eval_results.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# Hlavní část
# =========================
sun = load_data("sun_answers.json")
sun_cc = load_data("sun_cc_answers.json")

sun_stats = compute_accuracy_with_ci(sun, n_bootstrap=BOOTSTRAP_SAMPLES)
sun_cc_stats = compute_accuracy_with_ci(sun_cc, n_bootstrap=BOOTSTRAP_SAMPLES)

sun_thr = find_resolution_threshold(
    sun_stats["distances"],
    sun_stats["accuracies"],
    target=TARGET_ACCURACY
)

sun_cc_thr = find_resolution_threshold(
    sun_cc_stats["distances"],
    sun_cc_stats["accuracies"],
    target=TARGET_ACCURACY
)

results = {
    "sun": {
        **sun_stats,
        "threshold": sun_thr,
    },
    "sun_cc": {
        **sun_cc_stats,
        "threshold": sun_cc_thr,
    },
}

print_summary("sun", sun_stats, sun_thr)
print_summary("sun_cc", sun_cc_stats, sun_cc_thr)

plot_results(results, target_accuracy=TARGET_ACCURACY, use_log_x=USE_LOG_X)