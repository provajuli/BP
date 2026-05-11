import matplotlib.pyplot as plt
import numpy as np

# Přibližná data podle původního obrázku
trial = np.arange(1, 17)
intensity = np.array([12, 10, 8, 6, 8, 6, 7, 6, 5, 6, 7, 6, 5, 6, 5, 6])

# 1 = perceived, 0 = not perceived
perceived = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=bool)

plt.figure(figsize=(7.2, 4.2))

# Spojnice
plt.plot(
    trial,
    intensity,
    color="black",
    linewidth=1.4,
    zorder=1
)

# Perceived = prázdné kolečko
plt.scatter(
    trial[perceived],
    intensity[perceived],
    facecolors="white",
    edgecolors="black",
    linewidths=1.8,
    s=70,
    label="Zaznamenáno",
    zorder=3
)

# Not perceived = plné kolečko
plt.scatter(
    trial[~perceived],
    intensity[~perceived],
    facecolors="black",
    edgecolors="black",
    linewidths=1.4,
    s=70,
    label="Nezaznamenáno",
    zorder=3
)

plt.xlabel("Číslo pokusu", fontsize=13)
plt.ylabel("Intenzita stimulu", fontsize=13)

plt.xlim(0.5, 16.5)
plt.ylim(0, 13)

plt.xticks(np.arange(0, 17, 2))
plt.yticks(np.arange(0, 14, 2))

plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

plt.legend(
    frameon=False,
    loc="upper right",
    fontsize=11
)

plt.tight_layout()

plt.savefig("staircase_method.pdf", bbox_inches="tight")
plt.savefig("staircase_method.png", dpi=300, bbox_inches="tight")

plt.show()