import mglyph as mg
import math
import colorsys
import numpy as np
import random


def inv_cubic_constrained(y, b_par, c_par, eps=1e-9):
    y = np.atleast_1d(np.asarray(y, float)).ravel()
    out = np.empty_like(y)

    A = (1.0 - b_par - c_par)
    B = c_par
    C = b_par

    def fval(xx):
        return b_par*xx + c_par*xx**2 + (1.0-b_par-c_par)*xx**3

    for i, yi in enumerate(y):
        yi = float(np.clip(yi, 0.0, 1.0))

        roots = np.roots([A, B, C, -yi])
        real = roots[np.isreal(roots)].real.astype(float)

        if real.size == 0:
            xi = 0.0
        else:
            in01 = real[(real >= -eps) & (real <= 1.0 + eps)]

            if in01.size > 0:
                # kořeny v [0,1] – vyber ten, co nejlíp sedí (většinou stačí první, ale tohle je bezpečné)
                cand = np.clip(in01, 0.0, 1.0)
                xi = cand[np.argmin(np.abs(fval(cand) - yi))]
            else:
                # žádný kořen v [0,1] – fallback: vezmi kořen nejblíž intervalu a potom ho ořízni
                xi = np.clip(real[np.argmin(np.minimum(np.abs(real - 0.0), np.abs(real - 1.0)))], 0.0, 1.0)

        out[i] = np.clip(xi, 0.0, 1.0)

    return out.reshape(y.shape)


def horizontal_line(x: float, canvas:mg.Canvas) -> None:
    canvas.line((mg.lerp(x, canvas.xcenter, canvas.xleft), canvas.ycenter),    
                (mg.lerp(x, canvas.xcenter, canvas.xright), canvas.ycenter),   
                linecap='round', width='30p', color='navy')

def simple_scaled_square(x: float, canvas: mg.Canvas) -> None:
    tl = (mg.lerp(x, 0.0, -1), mg.lerp(x, 0.0, -1.0))
    br = (mg.lerp(x, 0, 1), mg.lerp(x, 0, 1))

    canvas.rect(tl, br, color='blue', width='20p')

def simple_scaled_circle(x: float, canvas: mg.Canvas) -> None:
    canvas.circle(canvas.center, mg.lerp(x, 0, 1), color='red', style='stroke', width='20p')

def simple_scaled_star(x: float, canvas: mg.Canvas) -> None:
    radius = mg.lerp(x, 0.01, canvas.ysize/2)
    vertices = []
    for segment in range(5):
        vertices.append(mg.orbit(canvas.center, segment * 2*math.pi/5, radius))
        vertices.append(mg.orbit(canvas.center, (segment + 0.5) * 2*math.pi/5, math.cos(2*math.pi/5)/math.cos(math.pi/5) * radius))
    canvas.polygon(vertices, color="#4BCA4B", closed=True, width='30p')

def simple_polygon(x: float, canvas: mg.Canvas) -> None:
    l = (mg.lerp(x, canvas.xcenter, canvas.xleft), canvas.ycenter)
    ct = (canvas.xcenter, -0.2)
    r = (mg.lerp(x, canvas.xcenter, canvas.xright), canvas.ycenter)
    cb = (canvas.xcenter, 0.1)
    canvas.polygon([l, cb, r, ct], color='#A2a', closed=True)

# _______________________ADVANCED GLYPHS_______________________

# author: Peter Ďurica
def sun_graph(x: float, canvas: mg.Canvas) -> None:
    cm = mg.ColorMap({0: (0, 0.8, 1), 
                      100/3: (1, 0.423, 0), 
                      200/3: (1, 0.423, 0), 
                      100: (0, 0.8, 1)})
    
    canvas.circle((0, 0), 5, style='fill', color=cm.get_color(x))
    sun_x = mg.lerp(x, canvas.xleft, canvas.xright)
    sun_y = (sun_x * 1.3)**2 - 0.85
    canvas.circle((sun_x, sun_y), 0.2, style='fill', color='gold')
    for ray in range(16):
        canvas.line((sun_x, sun_y), 
                    mg.orbit((sun_x, sun_y), ray * 2*np.pi/16, 0.25), 
                    width='20p', color='gold')


# author: Dinara Garipova
def tree_growth(x: float, canvas: mg.Canvas) -> None:
    # 0–5: Growing the seed (circle + triangle)
    seed_growth = mg.clamped_linear(x, 0, 5)
    if seed_growth > 0:
        circle_radius = mg.lerp(seed_growth, 0.02, 0.07)
        triangle_width = mg.lerp(seed_growth, 0.02, 0.08)
        triangle_height = mg.lerp(seed_growth, 0.02, 0.08)

        canvas.circle(center=(canvas.xcenter, canvas.ybottom - 0.1),
                      radius=circle_radius, color='saddlebrown')
        canvas.polygon(vertices=[
            (canvas.xcenter - triangle_width / 2, canvas.ybottom - 0.08 - circle_radius),
            (canvas.xcenter + triangle_width / 2, canvas.ybottom - 0.08 - circle_radius),
            (canvas.xcenter, canvas.ybottom - 0.1 - circle_radius - triangle_height)
        ], color='saddlebrown')

    # 5–15: Sprout grows and straightens
    sprout_growth = mg.clamped_linear(x, 5, 15)
    if sprout_growth > 0:
        start_y = canvas.ybottom - 0.07 - circle_radius - triangle_height
        points = [(canvas.xcenter, start_y)]
        num_points = 10
        for i in range(1, num_points + 1):
            progress = i / num_points
            curve_factor = (1 - sprout_growth / 100)
            dx = 0.1 * np.sin(progress * np.pi) * curve_factor
            dy = progress * 0.3 * (sprout_growth / 100)
            points.append((canvas.xcenter + dx, start_y - dy))
        for i in range(len(points) - 1):
            canvas.line(points[i], points[i+1], color='green', width='40p')

    # 15–20: Sprout starts to turn brown
    stem_growth = mg.clamped_linear(x, 15, 20)
    if stem_growth > 0:
        start_y = canvas.ybottom - 0.08 - circle_radius - triangle_height - 0.3
        total_height = 0.5
        num_segments = 10
        for i in range(num_segments):
            segment_progress = i / num_segments
            next_progress = (i + 1) / num_segments
            brownness = segment_progress * (stem_growth / 100)
            color = (mg.interpolate_color("saddlebrown", "green", 1 - brownness)
                     if hasattr(mg, "interpolate_color") else "green")
            y1 = start_y - segment_progress * total_height * (stem_growth / 100)
            y2 = start_y - next_progress * total_height * (stem_growth / 100)
            canvas.line((canvas.xcenter, y1), (canvas.xcenter, y2),
                        color=color, width="40p")

    # 20–30: Sprout fully hardens into brown trunk
    harden_growth = mg.clamped_linear(x, 20, 30)
    if harden_growth > 0:
        start_y = canvas.ybottom - 0.2
        max_height = 0.85
        cover_height = max_height * (harden_growth / 100)
        y1 = start_y
        y2 = start_y - cover_height
        canvas.line((canvas.xcenter, y1), (canvas.xcenter, y2),
                    color='saddlebrown', width="40p")

    # 30–50: Branches appear one by one
    branching = mg.clamped_linear(x, 30, 50)
    if branching > 0:
        start_y = canvas.ybottom - 0.3
        branches = [
            ((canvas.xcenter, start_y - 0.3), (canvas.xcenter - 0.2, start_y - 0.5)),
            ((canvas.xcenter, start_y), (canvas.xcenter + 0.3, start_y - 0.5)),
            ((canvas.xcenter, start_y - 0.4), (canvas.xcenter + 0.15, start_y - 0.6)),
            ((canvas.xcenter + 0.15, start_y - 0.25), (canvas.xcenter + 0.1, start_y - 0.4)),
        ]
        total_branches = len(branches)
        for i, (start, end) in enumerate(branches):
            branch_start = (i / total_branches) * 100
            branch_end = ((i + 1) / total_branches) * 100
            branch_progress = (branching - branch_start) / (branch_end - branch_start) * 100
            branch_progress = np.clip(branch_progress, 0, 100)
            if branch_progress > 0:
                x_current = mg.lerp(branch_progress, start[0], end[0])
                y_current = mg.lerp(branch_progress, start[1], end[1])
                canvas.line(start, (x_current, y_current), color='saddlebrown', width='20p')

    # 50–65: Leaves grow at the ends of branches
    leafing = mg.clamped_linear(x, 50, 65)
    if leafing > 0:
        branch_ends = [
            (canvas.xcenter - 0.2, canvas.ybottom - 0.8),
            (canvas.xcenter + 0.3, canvas.ybottom - 0.8),
            (canvas.xcenter + 0.15, canvas.ybottom - 0.95),
            (canvas.xcenter + 0.1, canvas.ybottom - 0.7),
        ]
        total_leaves = len(branch_ends)
        for i, center in enumerate(branch_ends):
            leaf_start = (i / total_leaves) * 100
            leaf_end = ((i + 1) / total_leaves) * 100
            leaf_progress = (leafing - leaf_start) / (leaf_end - leaf_start) * 100
            leaf_progress = np.clip(leaf_progress, 0, 100)
            if leaf_progress > 0:
                radius = mg.lerp(leaf_progress, 0.01, 0.05)
                canvas.circle(center=center, radius=radius, color='forestgreen')

    # 65–80: Crown grows and trunk thickens
    crown_growth = mg.clamped_linear(x, 65, 80)
    if crown_growth > 0:
        trunk_thickness = mg.lerp(crown_growth, 40, 300)
        canvas.line(
            (canvas.xcenter, canvas.ybottom - 0.1),
            (canvas.xcenter, canvas.ycenter - 0.2 + mg.lerp(crown_growth, 0.01, 0.7)),
            color='saddlebrown', width=f"{trunk_thickness}p", linecap="round"
        )
        crown_radius = mg.lerp(crown_growth, 0.01, 0.7)
        canvas.circle(center=(canvas.xcenter, canvas.ycenter - 0.2),
                      radius=crown_radius, color='forestgreen')

    # 80–90: Flowers grow one by one
    flower_growth = mg.clamped_linear(x, 80, 90)
    if flower_growth > 0:
        angles = np.linspace(0, 2*np.pi, 12)
        for i, angle in enumerate(angles):
            flower_start = (i / len(angles)) * 100
            flower_end = ((i + 1) / len(angles)) * 100
            flower_progress = (flower_growth - flower_start) / (flower_end - flower_start) * 100
            flower_progress = np.clip(flower_progress, 0, 100)
            if flower_progress > 0:
                x_flower = canvas.xcenter + 0.6 * np.cos(angle)
                y_flower = (canvas.ycenter - 0.2) + 0.6 * np.sin(angle)
                radius = mg.lerp(flower_progress, 0.01, 0.04)
                canvas.circle(center=(x_flower, y_flower), radius=radius, color='pink')

    # 90–100: Apples grow one by one
    apple_growth = mg.clamped_linear(x, 90, 100)
    if apple_growth > 0:
        angles = np.linspace(0, 2*np.pi, 12)
        for i, angle in enumerate(angles):
            apple_start = (i / len(angles)) * 100
            apple_end = ((i + 1) / len(angles)) * 100
            apple_progress = (apple_growth - apple_start) / (apple_end - apple_start) * 100
            apple_progress = np.clip(apple_progress, 0, 100)
            if apple_progress > 0:
                x_apple = canvas.xcenter + 0.6 * np.cos(angle)
                y_apple = (canvas.ycenter - 0.2) + 0.6 * np.sin(angle)
                radius = mg.lerp(apple_progress, 0.01, 0.06)
                canvas.circle(center=(x_apple, y_apple), radius=radius, color='red')


# author: Ondřej Lukášek
def flower(x: float, canvas: mg.Canvas) -> None:
    c_green = "#00800a" #RGB
    c_tomato = "#ff6347" #RGB
    c_gold = "#ffd710" #RGB
    # TRUNK
    canvas.line(p1=(canvas.xcenter, canvas.ybottom),
                p2=(canvas.xcenter, mg.lerp(mg.clamped_linear(x, 0, 40), canvas.ybottom, canvas.ycenter)),
                color=c_green, width="90p", linecap="round")
    # PETALS
    petal_growth = mg.clamped_linear(x, 20, 50)
    canvas.circle(center=(canvas.xcenter-0.18, canvas.ycenter+0.10), radius=mg.lerp(petal_growth, 0, 0.20), color=c_tomato)
    petal_growth = mg.clamped_linear(x, 30, 60)
    canvas.circle(center=(canvas.xcenter-0.23, canvas.ycenter-0.15), radius=mg.lerp(petal_growth, 0, 0.20), color=c_tomato)
    petal_growth = mg.clamped_linear(x, 40, 70)
    canvas.circle(center=(canvas.xcenter, canvas.ycenter-0.33), radius=mg.lerp(petal_growth, 0, 0.20), color=c_tomato)
    petal_growth = mg.clamped_linear(x, 50, 80)
    canvas.circle(center=(canvas.xcenter+0.23, canvas.ycenter-0.15), radius=mg.lerp(petal_growth, 0, 0.20), color=c_tomato)
    petal_growth = mg.clamped_linear(x, 60, 90)
    canvas.circle(center=(canvas.xcenter+0.18, canvas.ycenter+0.10), radius=mg.lerp(petal_growth, 0, 0.20), color=c_tomato)
    # YELLOW CENTER
    center_growth = mg.clamped_linear(x, 70, 100)
    canvas.circle(center=(canvas.xcenter, canvas.ycenter-0.09), radius=mg.lerp(center_growth, 0, 0.17), color=c_gold)
    # LEAF LEFT
    leaf_growth = mg.clamped_linear(x, 0, 20)
    canvas.ellipse(center=(canvas.xcenter-0.25, canvas.ycenter+0.6), 
                   rx=mg.lerp(leaf_growth, 0, 0.5),
                   ry=mg.lerp(leaf_growth, 0, 0.2),
                   color=c_green)
    # LEAF RIGHT
    leaf_growth = mg.clamped_linear(x, 20, 40)
    canvas.ellipse(center=(canvas.xcenter+0.25, canvas.ycenter+0.6),
                   rx=mg.lerp(leaf_growth, 0, 0.5),
                   ry=mg.lerp(leaf_growth, 0, 0.2), color=c_green)


# author: Michaela Macková
def circular_progressbar_ticks_color(x: float, canvas: mg.Canvas) -> None:
    def draw_partial_ring(x: float, canvas: mg.Canvas, inner_radius: float, outer_radius: float, segments: int = 100, color: str = "black", width: str = "20p", style: str = "fill") -> None:
        segments_lerp = int(mg.lerp(x, 0, segments))
        angle_offset = -np.pi / 2  # Start at the top of the circle
        # Outer circle points
        outer_points = [
            (canvas.xcenter + outer_radius * np.cos(2 * np.pi * i / segments + angle_offset),
            canvas.ycenter + outer_radius * np.sin(2 * np.pi * i / segments + angle_offset))
            for i in range(segments_lerp + 1)
        ]
        # Inner circle points (reversed to create a hole)
        inner_points = [
            (canvas.xcenter + inner_radius * np.cos(2 * np.pi * i / segments + angle_offset),
            canvas.ycenter + inner_radius * np.sin(2 * np.pi * i / segments + angle_offset))
            for i in range(segments_lerp, -1, -1)
        ]
        # Combine outer and inner points to form the ring
        points = outer_points + inner_points
        canvas.polygon(points, width=width, color=color, closed=True, style=style, linejoin='round')

    radius_outer = 0.5 * canvas.xsize
    radius_inner = 0.25 * canvas.xsize
    segments = 2000
    cm = mg.ColorMap({0: '#FBDDF3', 30: '#ec38bc', 65: '#7303c0', 100: '#1D0157'})
    draw_partial_ring(x, canvas, inner_radius=radius_inner, outer_radius=radius_outer, segments=segments, color=cm.get_color(x), style="fill")
    draw_partial_ring(segments, canvas, inner_radius=radius_inner, outer_radius=radius_outer, segments=segments, color="black", style="stroke", width="25p")
    tick_radius_center = (radius_inner + radius_outer) / 2
    tick_big_size = 0.15*canvas.xsize
    tick_small_size = 0.08*canvas.xsize
    for tick in range(16):
        tick_radius = (tick_radius_center - tick_big_size/2, tick_radius_center + tick_big_size/2)
        if ((tick % 2) == 1):
            tick_radius = (tick_radius_center - tick_small_size/2, tick_radius_center + tick_small_size/2)
        canvas.line(mg.orbit(canvas.center, tick*np.pi/8, tick_radius[0]),
                    mg.orbit(canvas.center, tick*np.pi/8, tick_radius[1]),
                    width="30p", linecap='round', color=(1, 1, 1, 1))
        canvas.line(mg.orbit(canvas.center, tick*np.pi/8, tick_radius[0]),
                    mg.orbit(canvas.center, tick*np.pi/8, tick_radius[1]),
                    width="10p", linecap='round', color='black')


# author: Faith Naz
def lightbulb(x:float, canvas:mg.Canvas) -> None:
    # ── Setup and basic parameters ──────────────────────────────────
    offset = canvas.ysize * 0.1
    cx, cy = canvas.center
    cy = cy + offset
    R = min(canvas.xsize, canvas.ysize) * 0.35
    ry = R * 1.4
    brightness = max(0, min(x / 100, 1))  # normalize brightness between 0 and 1
    inner_r = R * 0.6

    # Soft gradient color map for the glass
    cm = mg.ColorMap({
        0: '#fcf6c1',
        50: '#f9ef94',
        100: '#fadb82'
    })
    glass_color = cm.get_color(x)

    # ── 1) Glass and Inner Opacity + Reflection ─────────────────────
    canvas.ellipse((cx, cy), rx=inner_r, ry=inner_r * 0.6,
                   color='lightgoldenrodyellow', style='fill')

    highlight_rx = R * 0.9
    highlight_ry = ry * 0.25
    highlight_cy = cy - ry * 0.6
    canvas.ellipse((cx, highlight_cy), rx=highlight_rx, ry=highlight_ry,
                   color='white', style='fill')

    canvas.ellipse((cx, cy), rx=R, ry=ry, color=glass_color, style='fill')
    canvas.ellipse((cx, cy), rx=R, ry=ry, color='white', style='stroke', width='5p')
    canvas.ellipse((cx, cy), rx=R, ry=ry, color='darkslategray', style='stroke', width='15p')

    # Extra highlight spot if brightness > 0.3
    if brightness > 0.3:
        spot_r = R * 0.25
        spot_cx = cx - R * 0.15
        spot_cy = cy - ry * 0.25
        canvas.ellipse((spot_cx, spot_cy),
                       rx=spot_r, ry=spot_r * 0.7,
                       color='#f5edd3', style='fill')

    # ── 2) Filament Support Legs (angled) ───────────────────────────
    coil_r = R * 0.18
    coil_dx = coil_r * 1.1
    coil_y = cy + R * 0.32
    leg_len = R * 0.35
    leg_width = '10p'
    leg_y = coil_y + R * 0.32

    left_leg_x = cx - (coil_dx * 0.6)
    right_leg_x = cx + (coil_dx * 0.6)

    # Left leg
    x0, y0 = left_leg_x, leg_y
    x1 = x0 - leg_len * np.cos(np.radians(60))
    y1 = y0 - leg_len * np.sin(np.radians(60))
    canvas.line((x0, y0), (x1, y1), color='#ffa500', style='stroke', width=leg_width)

    # Right leg
    x0, y0 = right_leg_x, leg_y
    x1 = x0 + leg_len * np.cos(np.radians(60))
    y1 = y0 - leg_len * np.sin(np.radians(60))
    canvas.line((x0, y0), (x1, y1), color='#ffa500', style='stroke', width=leg_width)

    # ── 3) Circular Filament Rings ──────────────────────────────────
    filament_width = f"{int(10 + 20 * brightness)}p"
    coil_cm = mg.ColorMap({
        0.0: '#ffa500',
        0.5: '#eec95d',
        0.8: '#eeb75d',
        1.0: '#eeac5d'
    })
    coil_color = coil_cm.get_color(brightness)

    for dx in (-coil_dx, 0, coil_dx):
        canvas.ellipse((cx + dx, coil_y),
                       rx=coil_r, ry=coil_r * 0.7,
                       color=coil_color, style='stroke', width=filament_width)

    # ── 4) Inner Rotating Light Ring ─────────────────────────────────
    if brightness > 0.6:
        ring_r = R * 0.70
        tick = R * 0.10
        n_ticks = 8  # number of ticks on the ring
        phase_offset = brightness * 2 * np.pi

        for i in range(n_ticks):
            ang = phase_offset + (i * 2 * np.pi / n_ticks)
            p0 = (cx + np.cos(ang) * ring_r,
                  cy + np.sin(ang) * ring_r)
            p1 = (cx + np.cos(ang) * (ring_r + tick),
                  cy + np.sin(ang) * (ring_r + tick))
            canvas.line(p0, p1, color='gold', style='stroke', width='10p')

    # ── 5) Outer Light Rays ──────────────────────────────────────────
    if brightness > 0.8:
        n_rays = int(8 + 8 * (brightness - 0.8) / 0.2)  # 8 to 16 rays
        ray_rx = R * 1.30
        ray_ry = ry * 1.30
        start_angle = np.pi * 0.8
        end_angle = 2 * np.pi * 1.1

        for i in range(n_rays):
            ang = start_angle + (i / (n_rays - 1)) * (end_angle - start_angle)
            p0 = (cx + np.cos(ang) * R, cy + np.sin(ang) * ry)
            p1 = (cx + np.cos(ang) * ray_rx, cy + np.sin(ang) * ray_ry)
            canvas.line(p0, p1, color='gold', style='stroke', width='10p')

    # ── 6) Bulb Socket ───────────────────────────────────────────────
    socket_w = R * 0.6
    socket_h = R * 0.18
    socket_top = cy + ry - R * 0.75

    canvas.rect((cx - socket_w / 2, socket_top),
                (cx + socket_w / 2, socket_top + socket_h),
                color='gray', style='fill')

    for i in range(1, 4):
        y = socket_top + i * socket_h / 4
        canvas.line((cx - socket_w / 2, y), (cx + socket_w / 2, y),
                    color='dimgray', style='stroke', width='3p')
        

#author: Ondřej Áč
# Foam polygon using LUT
n_pts = 100
random.seed(5485)
def gen_lut(npts : int = 40, ks : int = 5):
    X = [random.uniform(0.04, 0.08) for i in range(npts)]
    res = [None] * npts
    for i in range(npts):
        acc = 0.0
        cnt = 0.0
        for j in range(ks):
            k = i + j - ks // 2
            if(k > 0 and k < npts):
                acc += X[k]
        res[i] = acc / ks
    return res

lut = gen_lut(n_pts, 7)
foam_scale = 2

def beer_glyph(t: float, canvas: mg.Canvas) -> None:
    t = t * 0.01 + 0.01
    t = np.pow(t, 0.8)
    center = canvas.center
    top_w = 0.6 * t
    base_w = 0.4 * t
    glass_h = 1.75 * t
    num_lines = 9
    beer_h = glass_h * t
    beer_w = t * (top_w - base_w)
    # Compute base offsets
    base_p = canvas.bottom_center
    base_p = (base_p[0], base_p[1] + 0.05)
    top_p = (base_p[0], base_p[1] + beer_h)
    # Compute vertices
    l_base = (base_p[0] - base_w, base_p[1])
    r_base = (base_p[0] + base_w, base_p[1])
    bl_top = (l_base[0] - beer_w, top_p[1])
    br_top = (r_base[0] + beer_w, top_p[1])
    gl_top = (base_p[0] - top_w, base_p[1] + glass_h)
    gr_top = (base_p[0] + top_w, base_p[1] + glass_h)
    beer_vert = [l_base, r_base, br_top, bl_top]
    glass_vert = [l_base, r_base, gr_top, gl_top]

    # Flip Y
    beer_vert = [(x, 2 - y) for x, y in beer_vert]
    glass_vert = [(x, 2 - y) for x, y in glass_vert]

    # === FOAM POLYGON ===
    x_vals = [bl_top[0] + (br_top[0] - bl_top[0]) * i / (n_pts - 1) for i in range(n_pts)]
    bottom = [(1.0 * x, top_p[1]) for x in x_vals]
    top = [(1 * x, top_p[1] + foam_scale * t * lut[i]) for i, x in enumerate(x_vals)]
    foam_vert = bottom + top[::-1]

    # Flip Y
    foam_vert = [(x, 2 - y) for x, y in foam_vert]

    #canvas.rect(canvas.top_left, canvas.bottom_right, color=(0.15, 0.25, 0.0)) # BG
    canvas.polygon(foam_vert, (0.93,0.9,0.8,0.9), style='fill', closed=True) # Foam
    #canvas.polygon(foam_vert, (1.0,1.0,0.9,0.9), style='fill', closed=True) # Foam
    canvas.polygon(glass_vert, (0.4, 0.7, 1,0.2), style='fill') # Glass BG
    canvas.polygon(beer_vert, 'goldenrod', style='fill', closed=True) # Beer
    canvas.polygon(glass_vert, (0.4, 0.7, 1,0.7), width=0.02*t, style='stroke') # Glass outline
    # Measuring lines
    
    for i in range(num_lines):
        y = 2 - (base_p[1] + glass_h * (i + 1) / num_lines)
        y2 =  2 - (base_p[1] + glass_h * (i + 0.5) / num_lines)
        x1 = base_p[0] - base_w / 4
        x2 = base_p[0] + base_w / 4
        x3 = base_p[0] - base_w / 8
        x4 = base_p[0] + base_w / 8
        if(i < num_lines - 1):
            canvas.line((x1,y), (x2,y), (0,0,0,0.7), width=0.01*t)
        canvas.line((x3,y2), (x4,y2), (0,0,0,0.6), width=0.01*t) 


# created by: Julie Provaznikova
def candle_glyph(x: float, canvas: mg.Canvas) -> None:
    # svicka
    bg_color = mg.ColorMap({0: "#4C4C4C", 100: "#D0D0BF"})

    bgx = x
    if x > 70:
        t = (x - 70) / 15 * 100
        bgx = mg.lerp(t, 70.0, 0.0)

    canvas.circle((0, 0), 5, style='fill', color=bg_color.get_color(bgx))

    canvas.rounded_rect((-0.1, -0.3), (0.1, 0.8), 0.05, 0.05, 0.05, 0.05, style='fill', color="#D8D0A0")

    # knot
    canvas.line((0, -0.5), (0, -0.3), width='10p', color='black', linecap='round')
    
    # plamen, bude menit velikost a barvu, bude mit dve casti - vnitrni a vnejsi, vnejsi se objevi pozdeji
    if x <= 70: # plamen se zvetsuje
        f = mg.lerp(x, 0.0, 100.0)
    elif x <= 85: # plamen se zmensuje
        t = (x - 70) / 15 * 100
        f = mg.lerp(t, 100.0, 0.0)
    else: # plamen neni
        f = 0.0

    #vnitrni plamen
    canvas.ellipse((0, -0.5), mg.lerp(f, 0.0, 0.1), mg.lerp(f, 0.1, 0.2), style='fill', color=mg.ColorMap({0: "#FFFF47", 70: "#FFE07A", 85: "#E3AD70"}).get_color(x))
    #vnejsi plamen
    canvas.ellipse((0, -0.5), mg.lerp(f, 0.0, 0.8), mg.lerp(f, 0.15, 0.8), style='fill', color=(1, 0.55, 0, 0.2))
    
    wax_x = x
    if wax_x > 85:
        wax_x = 85

    canvas.ellipse((0.1, -0.2), mg.lerp(wax_x, 0.0, 0.15), mg.lerp(wax_x, 0.1, 0.2), style='fill', color="#DBDBCA")
    canvas.ellipse((-0.1, -0.1), mg.lerp(wax_x, 0.0, 0.15), mg.lerp(wax_x, 0.1, 0.5), style='fill', color="#DBDBCA")
    
    if x >= 85:
        s = (x - 85) / 15 * 100 
        if s > 100:
            s = 100

        y0 = mg.lerp(s, -0.45, -0.95)

        a = mg.lerp(s, 0.10, 0.60)

        wobble = 0.05 * np.sin(x * 0.35)

        c1 = (0.85, 0.85, 0.85, a)
        c2 = (0.90, 0.90, 0.90, a * 0.85)
        c3 = (0.95, 0.95, 0.95, a * 0.70)

        canvas.ellipse((0.02 + wobble, y0), mg.lerp(s, 0.04, 0.14), mg.lerp(s, 0.05, 0.20), style='fill', color=c1)

        canvas.ellipse((-0.04 - wobble * 0.8, y0 - 0.18), mg.lerp(s, 0.035, 0.12), mg.lerp(s, 0.045, 0.18), style='fill', color=c2)

        canvas.ellipse((0.03 + wobble * 0.6, y0 - 0.34), mg.lerp(s, 0.03, 0.10), mg.lerp(s, 0.04, 0.15), style='fill', color=c3)

    # paprsky u plaminku
    if f > 60:
        b = (f - 60) / 40
        if b > 1:
            b = 1

        n_rays = int(8 + 10 * b)         
        r0 = 0.18                         
        r1 = 0.18 + 0.20 * b              
        ray_w = f"{int(4 + 10 * b)}p"     

        phase = x * 0.08

        for i in range(n_rays):
            ang = phase + i * 2*np.pi / n_rays
            p0 = mg.orbit((0, -0.5), ang, r0)
            p1 = mg.orbit((0, -0.5), ang, r1)
            canvas.line(p0, p1, width=ray_w, color='gold', linecap='round')


# created by: Grace Otuya
def ripple_wave_glyph(x: float, canvas: mg.Canvas) -> None:
    frequency = mg.lerp(x, 3, 10)      # How many ripple waves
    amplitude = mg.lerp(x, 0.1, 0.3)   # How big the waves are
    deformity = mg.lerp(x, 0.0, 0.4)   # How uneven the wave is

    points = []
    steps = 200      # How smooth the circle is
    for i in range(steps + 1):
        angle = (2 * np.pi) * (i / steps)
        base_radius = 0.5 + amplitude * np.sin(frequency * angle)

        # Add random deformity
        random_shift = (random.uniform(-1, 1) * deformity)
        radius = base_radius + random_shift * amplitude
        points.append((radius * np.cos(angle), radius * np.sin(angle)))

    # Draw the ripple wave shape
    for idx in range(len(points) - 1):
        canvas.line(points[idx], points[idx + 1], color='navy', width='15p', linecap='round')


# --------------------------- MODEL FIT -------------------------
def ui_to_x(u, gamma):
    u = max(1, min(100, int(u)))
    p = u / 100.0
    x = 100.0 * (p ** (1.0 / gamma))
    return x

GAMMA = 1.469

B = 1.2
C = -1.902

def tree_growth_gamma(u: float, canvas: mg.Canvas) -> None:
    # u = "UI hodnota" 0..100 nebo 1..100 (podle toho co používáš)
    # převedeme na fyzické x
    p = max(0.0, min(1.0, u / 100.0))
    x = 100.0 * (p ** (1.0 / GAMMA))
    tree_growth(x, canvas)


def ui_to_x_cc(u: float, b_par: float, c_par: float) -> float:
    u = max(1, min(100, int(round(u))))
    p = (u - 1) / 99.0               # 0..1
    x01 = float(inv_cubic_constrained(p, b_par, c_par)[0])  # 0..1
    return 100.0 * x01               # 0..100 pro mglyph


def sun_graph_cc(u: float, canvas: mg.Canvas) -> None:
    x = ui_to_x_cc(u, B, C)
    sun_graph(x, canvas)


SIMPLE_GLYPHS = {
    "line": horizontal_line,
    "square": simple_scaled_square,
    "circle": simple_scaled_circle,
    "star": simple_scaled_star,
    "polygon": simple_polygon,
}

ADVANCED_GLYPHS = {
    #"sun": sun_graph,
    #"tree_growth": tree_growth,
    #"flower": flower,
    #"circular_progressbar": circular_progressbar_ticks_color,
    #"beer": beer_glyph,
    #"candle": candle_glyph,
    #"ripple_wave": ripple_wave_glyph,
    #"tree_growth_gamma": tree_growth_gamma,
    "sun_cc": sun_graph_cc,
}