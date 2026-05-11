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
    ax.set_xlabel("Physical value $x$", fontsize=15)
    ax.set_ylabel("Perceived value $f(x)$", fontsize=15)


# -------------------- LINEAR --------------------
def plot_linear(x):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, linear_function(x), linewidth=2, label="y = x")

    style_axis(ax)
    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig("linear_model.png", dpi=300, bbox_inches="tight")
    plt.close()


# -------------------- GAMMA --------------------
def plot_gamma(x):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, gamma_function(x, 0.5), linewidth=2, label=r"$\gamma = 0.5$")
    ax.plot(x, gamma_function(x, 3), linewidth=2, label=r"$\gamma = 3$")
    ax.plot(x, x, linestyle="--", linewidth=1, alpha=0.5, label="y = x")

    style_axis(ax)
    ax.legend(frameon=False)

    plt.tight_layout()
    #plt.show()
    plt.savefig("gamma_model.png", dpi=300, bbox_inches="tight")
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

    ax.plot(
        x,
        cubic_constrained_function(x, b=0.2, c=2.0),
        linewidth=2,
        label="b = 0.2, c = 2.0",
    )

    ax.plot(
        x,
        x,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="y = x",
    )

    style_axis(ax)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("polynomial_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_models_one_graph(x):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Linear
    ax.plot(
        x,
        linear_function(x),
        linewidth=2,
        label="Linear: y = x",
        linestyle="--",
    )

    # Gamma
    ax.plot(
        x,
        gamma_function(x, 3),
        linewidth=2,
        label=r"Gamma: $\gamma = 3$"
    )

    # Polynomial
    ax.plot(
        x,
        cubic_constrained_function(x, b=1.5, c=-2.0),
        linewidth=2,
        label="Polynomial: b = 1.5, c = -2.0"
    )

    style_axis(ax)

    ax.set_title("Comparison of mapping models", fontsize=16)

    ax.legend(
        frameon=False,
        fontsize=11,
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        "all_models_one_graph.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# -------------------- MAIN --------------------
x = np.linspace(0, 1, 400)

plot_linear(x)
plot_all_models_one_graph(x)
plot_gamma(x)
plot_polynomial(x)