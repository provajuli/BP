import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import PercentFormatter, MultipleLocator


# -------------------- CONFIG --------------------
MODE = "matus"  # "matus" nebo "julie"

M_FILE_ORIGINAL = "sun_matus_answers.json"
M_FILE_TRANSFORMED = "sun_cc_matus_answers.json"

J_FILE_ORIGINAL = "sun_julie_answers.json"
J_FILE_TRANSFORMED = "sun_cc_julie_answers.json"

M_OUTPUT_FILE = "sun_resolution_comparison_matus.png"
J_OUTPUT_FILE = "sun_resolution_comparison_julie.png"

LABEL_ORIGINAL = "Bez percepční transformace"
LABEL_TRANSFORMED = "S percepční transformací"


# -------------------- LOAD DATA --------------------
def load_answers(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["answers"]


# -------------------- PROCESS DATA --------------------
def compute_accuracy_by_distance(answers):
    grouped = defaultdict(list)

    for answer in answers:
        distance = round(float(answer["distance"]), 3)
        correct = bool(answer["correct"])
        grouped[distance].append(correct)

    distances = []
    accuracies = []
    counts = []

    for distance in sorted(grouped.keys()):
        values = grouped[distance]
        accuracy = np.mean(values)

        distances.append(distance)
        accuracies.append(accuracy)
        counts.append(len(values))

    return np.array(distances), np.array(accuracies), np.array(counts)


def estimate_jnd(distances, accuracies, threshold=0.75):
    """
    Vrací první vzdálenost, kde accuracy dosáhne zvoleného threshold.
    Pokud threshold není dosažen, vrací None.
    """
    for d, a in zip(distances, accuracies):
        if a >= threshold:
            return d
    return None


# -------------------- PLOT --------------------
def smooth(y, window=3):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='same')


def plot_accuracy_comparison():
    if MODE == "matus":
        original_answers = load_answers(M_FILE_ORIGINAL)
        transformed_answers = load_answers(M_FILE_TRANSFORMED)
    else:
        original_answers = load_answers(J_FILE_ORIGINAL)
        transformed_answers = load_answers(J_FILE_TRANSFORMED)

    d_orig, acc_orig, _ = compute_accuracy_by_distance(original_answers)
    d_trans, acc_trans, _ = compute_accuracy_by_distance(transformed_answers)

    # smoothing
    acc_orig_s = smooth(acc_orig)
    acc_trans_s = smooth(acc_trans)

    jnd_orig = estimate_jnd(d_orig, acc_orig)
    jnd_trans = estimate_jnd(d_trans, acc_trans)

    plt.figure(figsize=(8, 5))

    # křivky
    plt.plot(d_orig, acc_orig_s, marker="o", linewidth=2.5,
             label=f"Bez transformace (JND = {jnd_orig:.2f})", alpha=0.8)

    plt.plot(d_trans, acc_trans_s, marker="o", linewidth=2.5,
             label=f"S transformací (JND = {jnd_trans:.2f})", alpha=0.8)

    # 75 % threshold
    plt.axhline(0.75, linestyle="--", linewidth=1.5, alpha=0.7,
                label="75 % práh rozlišení")

    # vertikální čáry (bez textu!)
    plt.axvline(jnd_orig, linestyle=":", linewidth=1.5, alpha=0.7)
    plt.axvline(jnd_trans, linestyle=":", linewidth=1.5, alpha=0.7)

    # osy
    plt.xlim(0, 15)
    plt.ylim(0, 1.05)

    plt.xlabel("Rozdíl mezi porovnávanými hodnotami")
    plt.ylabel("Úspěšnost odpovědí")
    plt.title("Porovnání rozlišení glyphu slunce")

    # jemný grid
    plt.grid(True, linestyle="--", alpha=0.2)

    # legenda
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.show()

    print("=== Výsledky ===")
    print(f"JND bez transformace: {jnd_orig}")
    print(f"JND s transformací: {jnd_trans}")
    if MODE == "matus":
        print(f"Graf uložen jako: {M_OUTPUT_FILE}")
    else:
        print(f"Graf uložen jako: {J_OUTPUT_FILE}")


def plot_all_respondents_comparison(
    file_r1_original,
    file_r1_transformed,
    file_r2_original,
    file_r2_transformed,
    output_file="sun_resolution_comparison_all.png"
):
    datasets = [
        (file_r1_original, "Respondent 1 bez transformace", "o", "-"),
        (file_r1_transformed, "Respondent 1 s transformací", "o", "--"),
        (file_r2_original, "Respondent 2 bez transformace", "s", "-"),
        (file_r2_transformed, "Respondent 2 s transformací", "s", "--"),
    ]

    plt.figure(figsize=(9, 5.5))

    for filename, label, marker, linestyle in datasets:
        answers = load_answers(filename)
        d, acc, _ = compute_accuracy_by_distance(answers)
        jnd = estimate_jnd(d, acc)

        plt.plot(
            d,
            acc,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            label=f"{label} (JND = {jnd:.2f})",
            alpha=0.8
        )

        if jnd is not None:
            plt.axvline(jnd, linestyle=":", linewidth=1, alpha=0.45)

    plt.axhline(
        0.75,
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="75 % práh rozlišení"
    )

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))   # hlavní tick každých 1
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5)) # menší tick každých 0.5    

    plt.xlim(0, 20)
    plt.ylim(0, 1.05)

    plt.xlabel("Rozdíl mezi porovnávanými hodnotami", fontsize=12)
    plt.ylabel("Úspěšnost odpovědí", fontsize=12)
    #plt.title("Porovnání rozlišení glyphu slunce")

    plt.grid(True, which="major", linestyle="--", alpha=0.3)
    plt.grid(True, which="minor", linestyle=":", alpha=0.15)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Graf uložen jako: {output_file}")


if __name__ == "__main__":
    #plot_accuracy_comparison()
    plot_all_respondents_comparison(
    "sun_julie_answers.json",
    "sun_cc_julie_answers.json",
    "sun_matus_answers.json",
    "sun_cc_matus_answers.json"
)