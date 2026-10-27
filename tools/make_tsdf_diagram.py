"""Draw media/diagrams/tsdf_fusion.svg (and .png when cairosvg is installed): how
depth from many views is fused in a voxel grid, and what gets thrown out."""

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
         text(660, 42, "Fusion: how many noisy depth views become one surface (TSDF, 1.5mm voxels, ±6mm band)", "title", "middle")]

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

    # Panel 3: averaging across views; zero crossing = fused surface
    x = 890
    p += panel_frame(x, "3. Average all views; surface = where it's 0", "weighted average of signed distances per voxel")
    gx, gy = x + 37, 160
    readings = [SURFACE_X + 0.35, SURFACE_X - 0.4, SURFACE_X + 0.15]
    fills = {}
    for r in range(14):
        for c in range(14):
            values = [max(-1, min(1, (m - (c + 0.5)) * 1.5 / 6)) for m in readings if (m - (c + 0.5)) * 1.5 >= -6]
            if r < 3:  # top rows: only one view reached here
                values = values[:1]
            if values:
                v = np.mean(values)
                fills[(c, r)] = "#9CC3EA" if v > 0.02 else ("#F2B8B5" if v < -0.02 else "#FFFFFF")
    p += grid(gx, gy, 14, 14, fills)
    p.append(f'<rect x="{gx}" y="{gy}" width="{14 * CELL}" height="{3 * CELL}" fill="#FFF5D6" opacity="0.65"/>')
    p.append(text(gx + 8, gy + 3 * CELL - 8, "seen by 1 view only: not kept in observed.obj", "small"))
    for m, color in zip(readings, VIEW_COLORS):
        p.append(f'<line x1="{gx + m * CELL}" y1="{gy + 3 * CELL}" x2="{gx + m * CELL}" y2="{gy + 14 * CELL}" stroke="{color}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    fused = gx + np.mean(readings) * CELL
    p.append(f'<line x1="{fused}" y1="{gy + 3 * CELL}" x2="{fused}" y2="{gy + 14 * CELL}" stroke="#1F2933" stroke-width="3"/>')
    p.append(text(x + 20, 520, "Dashed: three views' readings. Solid: fused surface,", "small"))
    p.append(text(x + 20, 536, "kept where at least 2 views wrote (weight ≥ 2).", "small"))

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
