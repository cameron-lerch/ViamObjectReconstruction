''' Draw media/diagrams/open_table_ring.svg (and .png when cairosvg is installed): the
ring of table the survey checks around each object to tell a scan target from clutter.

    python tools/make_ring_diagram.py

Panels 1 and 2 classify a real 5mm grid with the same rule as recon/scene.py
(table_support): bare table, something else, or unseen. The percentages printed
under them are computed from the drawn cells. Panel 3 shows why a tall object
leaves some of its ring unseen.
'''

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "media" / "diagrams"
SCALE = 1.9  # px per mm
CELL = 5.0  # the rule's grid spacing (mm)
INNER, OUTER = 5.0, 60.0  # the ring's inner and outer distance from the object's box (mm)
COLORS = {"object": "#D9822B", "table": "#E4F2E8", "else": "#F2C14E", "unseen": "#7B8794"}


def text(x, y, s, cls="desc", anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def panel_frame(x, title, subtitle):
    return [f'<rect x="{x}" y="70" width="410" height="520" rx="14" fill="#FFFFFF" stroke="#BCCCDC" stroke-width="1.5"/>',
            text(x + 20, 102, title, "name"), text(x + 20, 124, subtitle)]


def ring_cells(half: float) -> np.ndarray:
    ''' Centers (mm, relative to the object's center) of the 5mm cells in the ring around a box of half-size `half`. '''
    reach = half + OUTER
    u = np.arange(-reach, reach + 1e-9, CELL)
    grid = np.array([(x, y) for x in u for y in u])
    return grid[np.any(np.abs(grid) > half + INNER, axis=1)]


def shadowed(cells: np.ndarray, half: float, height: float, camera: np.ndarray, camera_height: float) -> np.ndarray:
    ''' Cells whose line of sight from the camera passes through the object (a box `height` tall). '''
    hidden = np.zeros(len(cells), bool)
    for z in np.linspace(0.0, height, 40):
        point = camera + (camera_height - z) / camera_height * (cells - camera)  # where the ray is at height z
        hidden |= np.all(np.abs(point) <= half, axis=1)
    return hidden


def draw_ring(x0, y0, half, classes, cells, extra=()):
    ''' The ring's cells, the object, and the ring's outline, centered at (x0, y0) in px. '''
    parts = list(extra)
    size = CELL * SCALE
    for (cx, cy), kind in zip(cells, classes):
        px, py = x0 + cx * SCALE - size / 2, y0 - cy * SCALE - size / 2
        parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{size:.1f}" height="{size:.1f}" fill="{COLORS[kind]}" stroke="#FFFFFF" stroke-width="0.6"/>')
    side = 2 * half * SCALE
    parts.append(f'<rect x="{x0 - side / 2:.1f}" y="{y0 - side / 2:.1f}" width="{side:.1f}" height="{side:.1f}" rx="6" fill="{COLORS["object"]}" stroke="#8C4A12" stroke-width="1.5"/>')
    for distance, dash in ((INNER, "4 3"), (OUTER, "")):
        edge = 2 * (half + distance) * SCALE
        parts.append(f'<rect x="{x0 - edge / 2:.1f}" y="{y0 - edge / 2:.1f}" width="{edge:.1f}" height="{edge:.1f}" fill="none" '
                     f'stroke="#1F2933" stroke-width="1.4" stroke-dasharray="{dash}"/>')
    return parts


def verdict(classes):
    classes = np.array(classes)
    seen = classes != "unseen"
    bare = (classes[seen] == "table").mean()
    unseen = 1 - seen.mean()
    target = bare >= 0.9 and unseen <= 0.25
    return bare, unseen, target


def verdict_lines(x, bare, unseen, target):
    return [text(x + 205, 560, f"{bare:.0%} of the seen ring is bare table, {unseen:.0%} unseen", "bold", "middle"),
            text(x + 205, 580, "scan target" if target else "obstacle (kept out of the planner's way)", "pass" if target else "fail", "middle")]


def main():
    p = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="700" viewBox="0 0 1320 700" '
         'font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#4A5568"/></marker><style>'
         '.title{font-size:26px;font-weight:700;fill:#1F2933}.name{font-size:17px;font-weight:700;fill:#1F2933}'
         '.desc{font-size:13px;fill:#52606D}.small{font-size:12px;fill:#52606D}.bold{font-size:13px;font-weight:700;fill:#1F2933}'
         '.pass{font-size:14px;font-weight:700;fill:#2E7D4F}.fail{font-size:14px;font-weight:700;fill:#B83232}'
         '</style></defs>',
         '<rect x="0" y="0" width="1320" height="700" rx="16" fill="#FFFFFF"/>',
         text(660, 44, "The ring: is this object standing in open table?", "title", "middle")]

    # 1. A tall object in open table: its own shadow is unseen, everything seen is table.
    x = 20
    p += panel_frame(x, "1. Open table, tall object", "Seen from above, camera off to the lower left")
    half = 40.0
    cells = ring_cells(half)
    camera = np.array([-50.0, -50.0])  # 70mm off to the side, about as for the real pumpkin
    hidden = shadowed(cells, half, 200.0, camera, 610.0)
    classes = ["unseen" if h else "table" for h in hidden]
    p += draw_ring(x + 205, 330, half, classes, cells)
    p.append(f'<line x1="{x + 62}" y1="{506}" x2="{x + 100}" y2="{468}" stroke="#4A5568" stroke-width="1.6" marker-end="url(#arrow)"/>')
    p.append(f'<rect x="{x + 26}" y="{509}" width="150" height="16" fill="#FFFFFF" fill-opacity="0.9"/>')
    p.append(text(x + 30, 521, "toward the survey camera", "small"))
    bare, unseen, target = verdict(classes)
    p += verdict_lines(x, bare, unseen, target)

    # 2. An object at the table's edge with a cable beside it: lots unseen, and not all table.
    x = 455
    p += panel_frame(x, "2. At the table's edge, beside a cable", "Seen from straight above")
    half = 40.0
    cells = ring_cells(half)
    edge_x = half + 12.0  # the table ends 12mm right of the object; the floor beyond is out of range
    classes = []
    for cx, cy in cells:
        if cx > edge_x:
            classes.append("unseen")
        elif abs(cy + 0.45 * cx + 52) < 9:  # a cable running past the lower left
            classes.append("else")
        else:
            classes.append("table")
    floor = [f'<rect x="{x + 205 + edge_x * SCALE:.1f}" y="{330 - (half + OUTER + 8) * SCALE:.1f}" width="{(OUTER - 12 + 8) * SCALE:.1f}" '
             f'height="{2 * (half + OUTER + 8) * SCALE:.1f}" fill="#CBD2D9"/>']
    p += draw_ring(x + 205, 330, half, classes, cells, floor)
    label_x = x + 205 + (edge_x + 28) * SCALE
    p.append(f'<rect x="{label_x - 42:.1f}" y="{330 - 12}" width="84" height="36" rx="4" fill="#FFFFFF" fill-opacity="0.92"/>')
    p.append(text(label_x, 330 + 2, "floor, out", "bold", "middle"))
    p.append(text(label_x, 330 + 18, "of range", "bold", "middle"))
    bare, unseen, target = verdict(classes)
    p += verdict_lines(x, bare, unseen, target)

    # 3. Side view: why a tall object hides some of its ring.
    x = 890
    p += panel_frame(x, "3. Why some of the ring is unseen", "Side view: a tall object's shadow")
    table_y, s = 455, 0.46  # px per mm in this panel
    cam_x, cam_y = x + 120, table_y - 610 * s
    obj_left, obj_right, obj_top = x + 130, x + 130 + 150 * s, table_y - 200 * s
    p.append(f'<line x1="{x + 20}" y1="{table_y}" x2="{x + 395}" y2="{table_y}" stroke="#1F2933" stroke-width="2"/>')
    p.append(f'<rect x="{cam_x - 16}" y="{cam_y - 12}" width="32" height="20" rx="4" fill="#3E4C59"/>')
    p.append(text(cam_x + 24, cam_y + 3, "survey camera, 600mm up", "small"))
    p.append(f'<rect x="{obj_left:.1f}" y="{obj_top:.1f}" width="{obj_right - obj_left:.1f}" height="{table_y - obj_top:.1f}" rx="18" '
             f'fill="{COLORS["object"]}" stroke="#8C4A12" stroke-width="1.5"/>')
    p.append(text((obj_left + obj_right) / 2, obj_top + 30, "200mm", "bold", "middle"))
    # The ray grazing the object's top far corner lands on the table here; between it and the object is hidden.
    land = cam_x + (obj_right - cam_x) * (table_y - cam_y) / (obj_top - cam_y)
    p.append(f'<line x1="{cam_x}" y1="{cam_y + 8}" x2="{land:.1f}" y2="{table_y}" stroke="#4A5568" stroke-width="1.3" stroke-dasharray="5 4"/>')
    p.append(f'<line x1="{cam_x}" y1="{cam_y + 8}" x2="{obj_left:.1f}" y2="{table_y}" stroke="#4A5568" stroke-width="1.3" stroke-dasharray="5 4"/>')
    p.append(f'<rect x="{obj_right:.1f}" y="{table_y - 7}" width="{land - obj_right:.1f}" height="7" fill="{COLORS["unseen"]}"/>')
    p.append(text((obj_right + land) / 2, table_y + 22, "hidden table", "small", "middle"))
    ring_left, ring_right = obj_right + 5 * s, obj_right + 60 * s
    p.append(f'<line x1="{ring_left:.1f}" y1="{table_y + 40}" x2="{ring_right:.1f}" y2="{table_y + 40}" stroke="#1F2933" stroke-width="1.4"/>')
    for end in (ring_left, ring_right):
        p.append(f'<line x1="{end:.1f}" y1="{table_y + 34}" x2="{end:.1f}" y2="{table_y + 46}" stroke="#1F2933" stroke-width="1.4"/>')
    p.append(text((ring_left + ring_right) / 2, table_y + 60, "the ring (5-60mm out)", "small", "middle"))
    p.append(text(x + 205, 572, "Hidden table is unseen, not 'something else'", "bold", "middle"))

    # Legend and the rule.
    legend = [("object", "object (its box)"), ("table", "bare table"), ("else", "something else (cable, object)"), ("unseen", "unseen (shadow, floor out of range)")]
    lx = 120
    for key, label in legend:
        p.append(f'<rect x="{lx}" y="612" width="16" height="16" fill="{COLORS[key]}" stroke="#BCCCDC"/>')
        p.append(text(lx + 24, 625, label, "desc"))
        lx += 70 + 8 * len(label)
    p.append(text(660, 668, "A scan target needs at least 90% of the cells it saw to be bare table, and at most 25% of its ring unseen. "
                            "The ring is a 5mm grid from 5mm (dashed) to 60mm (solid) outside the object's box.", "desc", "middle"))
    p.append("</svg>")

    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / "open_table_ring.svg"
    svg.write_text("\n".join(p))
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(svg.with_suffix(".png")))
    except ImportError:
        pass
    print("wrote", svg)


if __name__ == "__main__":
    main()
