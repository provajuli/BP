import numpy as np
import matplotlib.pyplot as plt

# aby byl výstup vždy stejný
np.random.seed(42)

def sigmoid(x, x0, k):
    return 100 / (1 + np.exp(-k * (x - x0)))

x = np.linspace(0, 10, 400)

x0_A = 3.2
x0_B = 6.2
k = 1.2

y_A = sigmoid(x, x0_A, k)
y_B = sigmoid(x, x0_B, k)

# Body obou respondentů na velmi podobných relativních pozicích vůči jejich prahu
relative_positions = np.array([-2.5, -1.1, 0.0, 0.7, 1.8])

x_A_points = x0_A + relative_positions
x_B_points = x0_B + relative_positions

# Jemný šum
noise_A = np.random.normal(0, 4, size=len(x_A_points))
noise_B = np.random.normal(0, 4, size=len(x_B_points))

y_A_points = sigmoid(x_A_points, x0_A, k) + noise_A
y_B_points = sigmoid(x_B_points, x0_B, k) + noise_B

# Ořez na 0–100 %
y_A_points = np.clip(y_A_points, 0, 100)
y_B_points = np.clip(y_B_points, 0, 100)

fig, ax = plt.subplots(figsize=(7.2, 5.2))

# Křivky
ax.plot(x, y_A, color="black", linewidth=2)
ax.plot(x, y_B, color="black", linewidth=2)

# Body
ax.scatter(x_A_points, y_A_points, s=60, color="green", zorder=3)
ax.scatter(x_B_points, y_B_points, s=60, color="blue", zorder=3)

# Referenční čáry
ax.hlines(50, 0, x0_B, linestyles="dashed", color="grey")
ax.vlines(x0_A, 0, 50, linestyles="dashed", color="grey")
ax.vlines(x0_B, 0, 50, linestyles="dashed", color="grey")

# Popisky
ax.text(4.7, 78, "Respondent A", fontsize=10, color="green")
ax.text(7.6, 78, "Respondent B", fontsize=10, color="blue")

# Osy
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)

ax.set_xlabel("Intenzita jasu", fontsize=13, labelpad=12)
ax.set_ylabel("Procento odpovědí 'ano'", fontsize=13)

# Tick značky
ax.set_xticks([0, 10])
ax.set_xticklabels(["Nízká", "Vysoká"])

ax.set_yticks([0, 50, 100])
ax.set_yticklabels(["0", "50", "100"])


plt.tight_layout()
plt.savefig("response_criterion.pdf", dpi=300, bbox_inches="tight")
plt.show()