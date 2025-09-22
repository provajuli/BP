import mglyph as mg
import math
import colorsys
import numpy as np

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
    radius = mg.lerp(x, 0, canvas.ysize/2)
    vertices = []
    for segment in range(5):
        vertices.append(mg.orbit(canvas.center, segment * 2*math.pi/5, radius))
        vertices.append(mg.orbit(canvas.center, (segment + 0.5) * 2*math.pi/5, math.cos(2*math.pi/5)/math.cos(math.pi/5) * radius))
    canvas.polygon(vertices, color=(1,0,0,0.25), closed=True, width='30p')

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
        


def simple_colour_patch(x: float, canvas: mg.Canvas) -> None:
    hue = x / 100
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    canvas.rect(
        (canvas.xleft, canvas.ytop),
        (canvas.xright, canvas.ybottom),
        color=color,
        style="fill",
    )

SIMPLE_GLYPHS = {
    "line": horizontal_line,
    "square": simple_scaled_square,
    "circle": simple_scaled_circle,
    "star": simple_scaled_star,
    "polygon": simple_polygon,
}

ADVANCED_GLYPHS = {
    "sun": sun_graph,
    "tree_growth": tree_growth,
    "flower": flower,
    "circular_progressbar": circular_progressbar_ticks_color,
    "lightbulb": lightbulb,
}