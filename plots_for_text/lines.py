import numpy as np
from pathlib import Path

# =====================
# CONFIG
# =====================

WIDTH = 1600
HEIGHT = 320

OUTPUT = "header_pattern.svg"

N_LINES = 7
POINTS_PER_LINE = 7

STROKE_COLOR = "#FFFFFF"

# výraznější
STROKE_OPACITY = 0.22
STROKE_WIDTH = 3.2

SEED = 42


# =====================
# HELPERS
# =====================

def bezier_path(points):
    path = f"M {points[0][0]:.1f},{points[0][1]:.1f}"

    for i in range(1, len(points) - 2, 3):
        p1, p2, p3 = points[i], points[i + 1], points[i + 2]

        path += (
            f" C {p1[0]:.1f},{p1[1]:.1f}"
            f" {p2[0]:.1f},{p2[1]:.1f}"
            f" {p3[0]:.1f},{p3[1]:.1f}"
        )

    return path


def generate_curve(y_base, amplitude, phase, rng):
    xs = np.linspace(-100, WIDTH + 100, POINTS_PER_LINE)

    points = []

    for x in xs:
        y = (
            y_base
            + amplitude * np.sin((x / WIDTH) * np.pi * 1.4 + phase)
            + amplitude * 0.4 * np.sin((x / WIDTH) * np.pi * 3 + phase)
            + rng.normal(0, 5)
        )

        points.append((x, y))

    while (len(points) - 1) % 3 != 0:
        points.append(points[-1])

    return points


# =====================
# SVG
# =====================

rng = np.random.default_rng(SEED)

svg = []

svg.append(f'''
<svg xmlns="http://www.w3.org/2000/svg"
     width="{WIDTH}"
     height="{HEIGHT}"
     viewBox="0 0 {WIDTH} {HEIGHT}">
''')

svg.append('<rect width="100%" height="100%" fill="none"/>')

# hlavní výrazné křivky
for i in range(N_LINES):

    y_base = np.interp(i, [0, N_LINES - 1], [40, HEIGHT - 40])

    amplitude = rng.uniform(20, 55)
    phase = rng.uniform(0, np.pi * 2)

    points = generate_curve(y_base, amplitude, phase, rng)

    d = bezier_path(points)

    opacity = STROKE_OPACITY * rng.uniform(0.8, 1.2)
    width = STROKE_WIDTH * rng.uniform(0.8, 1.3)

    svg.append(f'''
    <path
        d="{d}"
        fill="none"
        stroke="{STROKE_COLOR}"
        stroke-width="{width:.2f}"
        stroke-opacity="{opacity:.3f}"
        stroke-linecap="round"
    />
    ''')

# jedna dominantní contour-like linka
points = generate_curve(
    y_base=HEIGHT * 0.52,
    amplitude=75,
    phase=1.1,
    rng=rng
)

d = bezier_path(points)

svg.append(f'''
<path
    d="{d}"
    fill="none"
    stroke="{STROKE_COLOR}"
    stroke-width="5"
    stroke-opacity="0.16"
    stroke-linecap="round"
/>
''')

svg.append("</svg>")

Path(OUTPUT).write_text("\n".join(svg), encoding="utf-8")

print(f"Saved SVG: {OUTPUT}")