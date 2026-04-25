import matplotlib.pyplot as plt
import numpy as np

# Styl (čistý, publikovatelný)
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

fig, ax = plt.subplots(figsize=(6, 6))

# Data
x = np.linspace(0, 1, 200)
ax.plot(x, x, color='black', linewidth=1.5, label=r'$y = x$')

# Body A, B, C
A = (0.25, 0.25)
C = (0.5, 0.5)
B = (0.3, 0.375)

ax.scatter(*A, color='black', s=50)
ax.scatter(*C, color='black', s=50)
ax.scatter(*B, color='blue', s=80)

# Popisky bodů (lepší než legenda)
ax.text(A[0]-0.18, A[1]-0.01, r'$(A, f(A))$', fontsize=14)
ax.text(C[0]+0.04, C[1]+0.00, r'$(C, f(C))$', fontsize=14)
ax.text(B[0]-0.24, B[1]+0.01, r'$(B, \frac{f(A) + f(C)}{2})$', fontsize=14, color='blue')

# Spojnice (jemnější)
ax.plot([A[0], B[0]], [A[1], B[1]], color='gray', linewidth=1)
ax.plot([C[0], B[0]], [C[1], B[1]], color='gray', linewidth=1)

# Osy
ax.set_xlabel(r'Fyzická hodnota $x$')
ax.set_ylabel(r'Vnímaná hodnota $f(x)$')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Jemná mřížka (velký rozdíl vizuálně)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Odstranění horního a pravého rámu (moderní styl)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Uložení do PDF (důležité!)
plt.savefig("beak_plot.pdf")
plt.show()