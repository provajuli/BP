import matplotlib.pyplot as plt
import numpy as np


def linear_function(x):
    return x


def gamma_function(x, gamma):
    return np.clip(x, 0, 1) ** gamma


def cubic_constrained_function(x, b, c):
    return b * x + c * x**2 + (1 - b - c) * x**3


def style_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xlabel("Fyzická hodnota $x$", fontsize=15)
    ax.set_ylabel("Vnímaná hodnota $f(x)$", fontsize=15)


# -------------------- LINEAR --------------------
def plot_linear(x):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, linear_function(x), linewidth=2, label="y = x")

    style_axis(ax)
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig("linear_model.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------- GAMMA --------------------
def plot_gamma(x):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, gamma_function(x, 0.05), linewidth=2, label=r"$\gamma = 0.05$")
    ax.plot(x, gamma_function(x, 5), linewidth=2, label=r"$\gamma = 3$")
    ax.plot(x, x, linestyle="--", linewidth=1, alpha=0.5, label="y = x")

    style_axis(ax)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()
    #plt.savefig("gamma_model.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------- POLYNOMIAL --------------------
def plot_polynomial(x):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        x,
        cubic_constrained_function(x, b=1.5, c=-2.0),
        linewidth=2,
        label="b = 1.5, c = -2.0",
    )

    ax.plot(x, x, linestyle="--", linewidth=1, alpha=0.5, label="y = x")

    style_axis(ax)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("polynomial_model.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------- MAIN --------------------
x = np.linspace(0, 1, 400)

#plot_linear(x)
plot_gamma(x)
#plot_polynomial(x)