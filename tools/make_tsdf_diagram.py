''' Draw media/diagrams/tsdf_fusion.svg (and .png when cairosvg is installed): how
depth from many views is fused in a truncated signed distance function (TSDF)
on a voxel grid, and what gets thrown out. '''

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "media" / "diagrams"
VIEW_COLORS = ["#2F6FB0", "#2E9E5B", "#D9822B"]
CELL = 24  # px per 1.5mm voxel
SURFACE_X = 6.4  # true wall position, in cells from the grid's left edge


def text(x, y, s, cls="desc", anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def panel_frame(x, title, subtitle):
    return [f'<rect x="{x}" y="70" width="410" height="470" rx="14" fill="#FFFFFF" stroke="#BCCCDC" stroke-width="1.5"/>',
            text(x + 20, 100, title, "name"), text(x + 20, 122, subtitle)]


def grid(x0, y0, cols, rows, fills):
    parts = []
    for r in range(rows):
        for c in range(cols):
            parts.append(f'<rect x="{x0 + c * CELL}" y="{y0 + r * CELL}" width="{CELL}" height="{CELL}" '
                         f'fill="{fills.get((c, r), "#FFFFFF")}" stroke="#D9E2EC" stroke-width="1"/>')
    return parts


def main():
    rng = np.random.default_rng(3)
    p = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="790" viewBox="0 0 1320 790" '
         'font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#4A5568"/></marker><style>'
         '.title{font-size:26px;font-weight:700;fill:#1F2933}.name{font-size:17px;font-weight:700;fill:#1F2933}'
         '.desc{font-size:13px;fill:#52606D}.small{font-size:12px;fill:#52606D}.bold{font-size:13px;font-weight:700;fill:#1F2933}'
         '.lane{font-size:14px;font-weight:700;fill:#616E7C;letter-spacing:0.5px}'
         '.ray{stroke-width:1.4;stroke-dasharray:5 4;fill:none}.truth{stroke:#1F2933;stroke-width:3}'
         '</style></defs>',
         '<rect x="0" y="0" width="1320" height="790" rx="16" fill="#FFFFFF"/>',
         text(660, 42, "Fusion: how many noisy depth views become one surface (TSDF: truncated signed distance function)", "title", "middle")]

    # Panel 1: each view measures the wall with noise
    x = 20
    p += panel_frame(x, "1. Each view measures the wall, with noise", "cross-section through the object's wall")
    gx, gy = x + 60, 160
    wall_x = gx + SURFACE_X * CELL
    p.append(f'<rect x="{wall_x}" y="{gy}" width="{gx + 14 * CELL - wall_x}" height="{14 * CELL}" fill="#E4F2E8"/>')
    p.append(text(wall_x + 60, gy + 14 * CELL - 12, "inside the object", "small"))
    p.append(f'<line x1="{wall_x}" y1="{gy}" x2="{wall_x}" y2="{gy + 14 * CELL}" class="truth"/>')
    cams = [(x + 30, 175), (x + 22, 300), (x + 30, 430)]
    for (cx, cy), color in zip(cams, VIEW_COLORS):
        p.append(f'<rect x="{cx - 12}" y="{cy - 9}" width="24" height="18" rx="3" fill="{color}"/>')
        for k in range(6):
            ty = gy + 20 + k * 55 + rng.uniform(-8, 8)
            p.append(f'<line x1="{cx + 12}" y1="{cy}" x2="{wall_x}" y2="{ty}" class="ray" stroke="{color}" opacity="0.35"/>')
            for _ in range(3):
                my = ty + rng.uniform(-10, 10)
                mx = wall_x + rng.normal(0, 7)
                p.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="3" fill="{color}"/>')
    # flying pixels near the silhouette, crossed out
    for fx, fy in ((wall_x - 40, gy + 8), (wall_x - 62, gy + 26)):
        p.append(f'<circle cx="{fx}" cy="{fy}" r="3" fill="#9AA5B1"/><path d="M{fx-6},{fy-6} L{fx+6},{fy+6} M{fx-6},{fy+6} L{fx+6},{fy-6}" stroke="#C0392B" stroke-width="2"/>')
    p.append(text(x + 20, 520, "Dots: each view's depth readings (±1-2mm noise).", "small"))
    p.append(text(x + 20, 536, "Red crosses: flying pixels at the silhouette, dropped first.", "small"))

    # Panel 2: one view writes signed distances into a band of voxels
    x = 455
    p += panel_frame(x, "2. One view writes into a band of voxels", "along each ray, around the surface that view measured")
    gx, gy = x + 37, 160
    measured = SURFACE_X + 0.35  # this view's (noisy) reading of the wall, in cells
    fills = {}
    for r in range(14):
        for c in range(14):
            d = (measured - (c + 0.5)) * 1.5  # mm in front (+) or behind (−) the measured surface
            if d > 6:
                fills[(c, r)] = "#EAF2FB"  # free space: clamped at +1
            elif d >= 0:
                fills[(c, r)] = "#9CC3EA"
            elif d >= -6:
                fills[(c, r)] = "#F2B8B5"
    p += grid(gx, gy, 14, 14, fills)
    mx = gx + measured * CELL
    p.append(f'<line x1="{mx}" y1="{gy}" x2="{mx}" y2="{gy + 14 * CELL}" stroke="{VIEW_COLORS[0]}" stroke-width="3"/>')
    p.append(f'<rect x="{x + 6}" y="{gy + 150}" width="24" height="18" rx="3" fill="{VIEW_COLORS[0]}"/>')
    p.append(f'<path d="M{x + 30},{gy + 159} L{gx - 4},{gy + 159}" stroke="#4A5568" stroke-width="1.5" marker-end="url(#arrow)"/>')
    band_l, band_r = mx - 4 * CELL, mx + 4 * CELL
    p.append(f'<path d="M{band_l},{gy - 10} L{band_r},{gy - 10}" stroke="#4A5568" stroke-width="1.5" marker-start="url(#arrow)" marker-end="url(#arrow)"/>')
    p.append(text((band_l + band_r) / 2, gy - 16, "±6mm truncation band", "small", "middle"))
    # Scale, inside the empty (not updated) cells at the bottom right.
    p.append(f'<rect x="{gx + 13 * CELL}" y="{gy + 13 * CELL}" width="{CELL}" height="{CELL}" fill="none" stroke="#1F2933" stroke-width="2"/>')
    p.append(text(gx + 13 * CELL - 6, gy + 13 * CELL + 16, "1 voxel = 1.5mm", "small", "end"))
    for i, (color, label) in enumerate((("#EAF2FB", "free space: clamped at +1"), ("#9CC3EA", "in front of the reading: +"),
                                         ("#F2B8B5", "behind the reading: −"), ("#FFFFFF", "over 6mm behind: not updated"))):
        lx, ly = x + 20 + (i % 2) * 185, 514 + (i // 2) * 18
        p.append(f'<rect x="{lx}" y="{ly - 10}" width="13" height="13" fill="{color}" stroke="#BCCCDC"/>')
        p.append(text(lx + 18, ly + 1, label, "small"))

    # Panel 3: one row of voxels across the wall, with each view's value and the average
    x = 890
    p += panel_frame(x, "3. Average all views; the surface is where it's 0", "one row of 7 voxels across the wall (values in mm)")
    centers = np.array([-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5])  # voxel centers; the true wall is at 0
    readings = {"view 1": 0.5, "view 2": -0.6, "view 3": 0.2}  # where each view measured the wall
    rows = {name: np.clip(r - centers, -6, 6) for name, r in readings.items()}  # + in front, − behind
    rows["average"] = np.mean(list(rows.values()), axis=0)
    col_w, row_h, left, top = 44, 34, x + 92, 196

    def shade(value):
        t = min(1.0, abs(value) / 6)
        base = np.array([0x9C, 0xC3, 0xEA]) if value > 0 else np.array([0xF2, 0xB8, 0xB5])
        color = (255 - (255 - base) * (0.35 + 0.65 * t)).astype(int)
        return "#%02X%02X%02X" % tuple(color)

    p.append(text(left - 10, top - 26, "voxel center:", "small", "end"))
    for k, c in enumerate(centers):
        p.append(text(left + k * col_w + col_w / 2, top - 26, f"{c:+.1f}", "small", "middle"))
    p.append(text(left, top - 44, "← outside", "small"))
    p.append(text(left + 7 * col_w, top - 44, "inside →", "small", "end"))
    p.append(text(left + 3 * col_w + col_w / 2, top - 58, "true wall", "small", "middle"))
    for r, (name, values) in enumerate(rows.items()):
        y = top + r * row_h + (10 if name == "average" else 0)
        color = VIEW_COLORS[r] if r < 3 else "#1F2933"
        p.append(text(left - 10, y + row_h / 2 + 4, name, "bold" if name == "average" else "desc", "end"))
        if r < 3:
            p.append(f'<rect x="{left - 86}" y="{y + row_h / 2 - 5}" width="10" height="10" rx="2" fill="{color}"/>')
        for k, value in enumerate(values):
            p.append(f'<rect x="{left + k * col_w}" y="{y}" width="{col_w}" height="{row_h}" fill="{shade(value)}" stroke="#FFFFFF" stroke-width="2"/>')
            p.append(text(left + k * col_w + col_w / 2, y + row_h / 2 + 4, f"{value:+.1f}", "bold" if name == "average" else "small", "middle"))
        if r < 3:  # where this view put the wall: ticks on the row's edges, clear of the numbers
            wx = left + (readings[name] - centers[0]) / 1.5 * col_w + col_w / 2
            p.append(f'<path d="M{wx},{y} L{wx},{y + 7} M{wx},{y + row_h - 7} L{wx},{y + row_h}" stroke="{color}" stroke-width="3"/>')
    # The average crosses zero between two voxels: linear interpolation gives the surface.
    avg = rows["average"]
    k = int(np.where(np.diff(np.sign(avg)) != 0)[0][0])
    surface = centers[k] + avg[k] / (avg[k] - avg[k + 1]) * 1.5
    sx = left + (surface - centers[0]) / 1.5 * col_w + col_w / 2
    true_x = left + (0 - centers[0]) / 1.5 * col_w + col_w / 2
    y_avg = top + 3 * row_h + 10
    p.append(f'<path d="M{sx},{y_avg - 6} L{sx},{y_avg + 8} M{sx},{y_avg + row_h - 8} L{sx},{y_avg + row_h + 6}" stroke="#1F2933" stroke-width="3"/>')
    p.append(text(sx, y_avg + row_h + 20, "fused surface", "small", "middle"))
    p.append(f'<line x1="{true_x}" y1="{top - 52}" x2="{true_x}" y2="{top - 32}" stroke="#2E9E5B" stroke-width="1.5" stroke-dasharray="4 3"/>')
    # How many views wrote to each voxel: the weight that "seen by at least 2 views" checks.
    y_w = y_avg + row_h + 30
    p.append(text(left - 10, y_w + 17, "views", "desc", "end"))
    for k in range(len(centers)):
        p.append(f'<rect x="{left + k * col_w}" y="{y_w}" width="{col_w}" height="24" fill="#F0F4F8" stroke="#FFFFFF" stroke-width="2"/>')
        p.append(text(left + k * col_w + col_w / 2, y_w + 17, "3", "small", "middle"))
    errors = ", ".join(f"{abs(r):.1f}" for r in readings.values())
    p.append(text(x + 20, y_w + 52, f"Ticks: where each view put the wall ({errors}mm off).", "desc"))
    p.append(text(x + 20, y_w + 70, f"The average crosses 0 at {abs(surface):.2f}mm from the true wall:", "desc"))
    p.append(text(x + 20, y_w + 88, "that is the fused surface. A surface is kept only where", "desc"))
    p.append(text(x + 20, y_w + 106, "at least 2 views wrote (the views row).", "desc"))
    p.append(text(x + 20, 532, "Blue: in front of a reading (+). Red: behind it (−).", "small"))

    # Bottom strip: what is thrown out, and where
    p.append('<rect x="20" y="560" width="1280" height="212" rx="14" fill="#F0F4F8" stroke="#BCCCDC" stroke-width="1.5"/>')
    p.append(text(40, 588, "WHAT GETS THROWN OUT, AND WHERE", "lane"))
    rows = [("Per view, before fusion", ["5 frames at each stop: per-pixel median; a pixel valid in fewer than half the frames is dropped.",
                                         "Pixels next to a depth jump (more than 5mm or 2%) are dropped: flying pixels at silhouettes.",
                                         "Points outside the object's box + 12mm, or less than 2.5mm above that view's own table fit."]),
            ("During fusion", ["Each reading only updates voxels within ±6mm of it along its ray; farther behind is left alone",
                               "(another view may see that region). Readings are averaged, never thrown out, so noise cancels."]),
            ("When the surface is extracted", ["Only where at least 2 views wrote (one view, alone, keeps all it saw).",
                                               "Fragments under 5% of the largest piece are removed; light smoothing (5 Taubin steps)."])]
    y = 616
    for heading, lines in rows:
        p.append(text(40, y, heading, "bold"))
        for k, line in enumerate(lines):
            p.append(text(260, y + k * 18, line))
        y += 18 * len(lines) + 10
    p.append("</svg>")
    svg = OUT / "tsdf_fusion.svg"
    svg.write_text("\n".join(p))
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(OUT / "tsdf_fusion.png"))
    except ImportError:
        pass
    print("wrote", svg)


if __name__ == "__main__":
    main()
