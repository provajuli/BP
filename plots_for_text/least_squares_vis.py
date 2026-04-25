import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# syntetická data
np.random.seed(0)
B = np.linspace(0.1, 0.9, 10)
B_hat = B + np.random.normal(0, 0.05, size=len(B))

fig, ax = plt.subplots(figsize=(6, 6))

# body
ax.scatter(B, B_hat, s=60)

# diagonála (ideální model)
x = np.linspace(0, 1, 100)
ax.plot(x, x, linestyle="--", linewidth=1, alpha=0.6)

# všechny chyby (tenké čáry)
for b, bh in zip(B, B_hat):
    ax.plot([b, b], [b, bh], linewidth=1, alpha=0.6)

# --- zvýrazněný bod ---
i = 4
b = B[i]
bh = B_hat[i]
error = bh - b

# silnější čára chyby
ax.plot([b, b], [b, bh], linewidth=2)

# čtverec chyby (area = error^2)
square = Rectangle(
    (b, b),              # levý dolní roh
    abs(error),          # šířka
    abs(error),          # výška
    fill=False,
    linewidth=2
)
ax.add_patch(square)

# popisky
ax.set_xlabel("Skutečná hodnota $B$")
ax.set_ylabel("Predikovaná hodnota $\\hat{B}$")

# styl
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("least_squares_square.pdf", dpi=300, bbox_inches="tight")
plt.close()