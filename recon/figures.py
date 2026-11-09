''' Figures made from saved results: rows of rendered panels with labels, as SVG
(plus a PNG copy when cairosvg is installed). Used after each scan and for the README. '''

import base64
import json
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .progression import POINTS_VOXEL
from .render import render, render_points, save_png

PANEL = 300
MEASURED = np.array([0.62, 0.68, 0.76])
FILLED = np.array([0.93, 0.55, 0.20])


def png_base64(image: np.ndarray) -> str:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "panel.png"
        save_png(image, path)
        return base64.b64encode(path.read_bytes()).decode()


def write_svg(svg: Path, parts: list[str]) -> None:
    svg.write_text("\n".join(parts))
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(svg.with_suffix(".png")))
    except ImportError:
        pass


def row_figure(svg: Path, title: str, panels: list[tuple[np.ndarray, str, str]], note: str = "") -> None:
    ''' A row of panels with a label and a sub-label under each. '''
    gap, top, bottom = 14, 52, 70 if note else 54
    width = len(panels) * PANEL + (len(panels) + 1) * gap
    height = top + PANEL + bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
             f'<rect width="{width}" height="{height}" rx="14" fill="#FFFFFF"/>',
             f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="700" fill="#1F2933">{title}</text>']
    for index, (image, label, sub) in enumerate(panels):
        x = gap + index * (PANEL + gap)
        parts.append(f'<image x="{x}" y="{top}" width="{PANEL}" height="{PANEL}" href="data:image/png;base64,{png_base64(image)}"/>')
        parts.append(f'<text x="{x + PANEL / 2}" y="{top + PANEL + 20}" text-anchor="middle" font-size="16" font-weight="700" fill="#1F2933">{label}</text>')
        parts.append(f'<text x="{x + PANEL / 2}" y="{top + PANEL + 40}" text-anchor="middle" font-size="13" fill="#52606D">{sub}</text>')
    if note:
        parts.append(f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="13" font-style="italic" fill="#3E4C59">{note}</text>')
    parts.append("</svg>")
    write_svg(svg, parts)


def comparison_figure(svg: Path, title: str, columns: list[tuple[str, list[np.ndarray], list[str]]],
                      takeaway: str, note: str = "") -> None:
    ''' Side-by-side setups of the same object (uncalibrated vs. calibrated camera, one
    camera vs. another): each column is a label, its renders and lines of metrics, with
    one takeaway line across the bottom. '''
    gap, top = 14, 52
    lines = max(len(metrics) for _, _, metrics in columns)
    column_widths = [len(images) * PANEL + (len(images) - 1) * gap for _, images, _ in columns]
    width = sum(column_widths) + (len(columns) + 1) * 2 * gap
    metrics_top = top + PANEL + 30
    takeaway_top = metrics_top + lines * 20 + 18
    height = takeaway_top + 46 + (24 if note else 0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
             f'<rect width="{width}" height="{height}" rx="14" fill="#FFFFFF"/>',
             f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="700" fill="#1F2933">{title}</text>']
    x = 2 * gap
    for (label, images, metrics), column_width in zip(columns, column_widths):
        for k, image in enumerate(images):
            parts.append(f'<image x="{x + k * (PANEL + gap)}" y="{top}" width="{PANEL}" height="{PANEL}" '
                         f'href="data:image/png;base64,{png_base64(image)}"/>')
        middle = x + column_width / 2
        parts.append(f'<text x="{middle}" y="{top + PANEL + 14}" text-anchor="middle" font-size="18" font-weight="700" fill="#1F2933">{label}</text>')
        for k, line in enumerate(metrics):
            parts.append(f'<text x="{middle}" y="{metrics_top + 14 + k * 20}" text-anchor="middle" font-size="14" fill="#3E4C59">{line}</text>')
        x += column_width + 2 * gap
    parts.append(f'<rect x="{2 * gap}" y="{takeaway_top}" width="{width - 4 * gap}" height="36" rx="8" fill="#E8F0FA"/>')
    parts.append(f'<text x="{width / 2}" y="{takeaway_top + 24}" text-anchor="middle" font-size="16" font-weight="700" fill="#1F2933">{takeaway}</text>')
    if note:
        parts.append(f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="13" font-style="italic" fill="#3E4C59">{note}</text>')
    parts.append("</svg>")
    write_svg(svg, parts)


def color_measured(mesh: o3d.geometry.TriangleMesh, measured: np.ndarray) -> o3d.geometry.TriangleMesh:
    ''' Gray-blue where the closed mesh lies on measured data, orange where it was filled in. '''
    gaps, _ = cKDTree(measured).query(np.asarray(mesh.vertices))
    colored = o3d.geometry.TriangleMesh(mesh)
    colored.vertex_colors = o3d.utility.Vector3dVector(np.where((gaps < 2.0)[:, None], MEASURED, FILLED))
    return colored


def framing(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, float]:
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    return center, float(np.linalg.norm(vertices - center, axis=1).max())



def progression_counts(n: int) -> list[int]:
    ''' The view counts shown in the progression figure: 1, 8, 16, 32 and all of them. '''
    return [k for k in (1, 8, 16, 32) if k < n] + [n]


def grid_figure(svg: Path, title: str, rows: list[list[tuple[np.ndarray, str, str]]], note: str = "") -> None:
    ''' Rows of panels, each with a label and a sub-label under it. '''
    gap, top, under = 14, 52, 50
    columns = max(len(row) for row in rows)
    width = columns * PANEL + (columns + 1) * gap
    height = top + len(rows) * (PANEL + under) + (24 if note else 6)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
             f'<rect width="{width}" height="{height}" rx="14" fill="#FFFFFF"/>',
             f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="700" fill="#1F2933">{title}</text>']
    for r, row in enumerate(rows):
        y = top + r * (PANEL + under)
        for c, (image, label, sub) in enumerate(row):
            x = gap + c * (PANEL + gap)
            parts.append(f'<image x="{x}" y="{y}" width="{PANEL}" height="{PANEL}" href="data:image/png;base64,{png_base64(image)}"/>')
            parts.append(f'<text x="{x + PANEL / 2}" y="{y + PANEL + 18}" text-anchor="middle" font-size="15" font-weight="700" fill="#1F2933">{label}</text>')
            parts.append(f'<text x="{x + PANEL / 2}" y="{y + PANEL + 36}" text-anchor="middle" font-size="13" fill="#52606D">{sub}</text>')
    if note:
        parts.append(f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="13" font-style="italic" fill="#3E4C59">{note}</text>')
    parts.append("</svg>")
    write_svg(svg, parts)


def progression_figure(folder: Path, svg: Path, title: str, counts: list[int],
                       elevation: float = 20.0, azimuth: float = -60.0) -> None:
    ''' The reconstruction from each number of views in `counts` (from `progression`'s output
    in `folder`), in three rows: the cumulative point cloud of those views, the observed mesh
    (only what at least 2 of them measured), and the final watertight mesh (orange: filled in). The
    point clouds are thinned to one point per POINTS_VOXEL cube, as `progression` saves them. '''
    rows = {str(r["views"]): r for r in json.loads((folder / "summary.json").read_text())}
    last = max(counts)

    def mesh(k: int, name: str) -> o3d.geometry.TriangleMesh:
        path = folder / f"views_{k:02d}" / name  # missing when that many views saw too little (2 views can)
        return o3d.io.read_triangle_mesh(str(path)) if path.exists() else o3d.geometry.TriangleMesh()

    final, final_observed = mesh(last, "mesh.ply"), mesh(last, "observed.ply")
    center, radius = framing(final if len(final.vertices) else final_observed)
    clouds, observed_row, final_row = [], [], []
    for k in counts:
        views = f"{k} view{'s' if k > 1 else ''}"
        row = rows[str(k)]
        path = folder / f"views_{k:02d}" / "points.ply"
        if not path.exists() and k == 1:
            path = folder / "views_01" / "raw_points.ply"
        cloud = o3d.io.read_point_cloud(str(path)) if path.exists() else o3d.geometry.PointCloud()
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else None
        clouds.append((render_points(points, PANEL, elevation, azimuth, center, radius, dot=2, colors=colors),
                       f"{views}: cumulative point cloud", f"{len(points):,} points (using {POINTS_VOXEL:g}mm voxels)"))
        observed = mesh(k, "observed.ply")
        observed_row.append((render(observed, PANEL, elevation, azimuth, center, radius), f"{views}: open mesh", ""))
        closed = mesh(k, "mesh.ply")
        if len(closed.triangles) and len(observed.vertices):
            final_row.append((render(color_measured(closed, np.asarray(observed.vertices)), PANEL, elevation, azimuth, center,
                                     radius, use_colors=True),
                              f"{views}: watertight mesh", f"{1 - row['measured_share']:.0%} filled in"))
        else:
            final_row.append((render(o3d.geometry.TriangleMesh(), PANEL), f"{views}: watertight mesh", "too little seen to close"))
    grid_figure(svg, title, [clouds, observed_row, final_row],
                "Open meshes keep surface at least 2 views measured (1 view keeps all it saw); red lines mark open edges. "
                "Watertight meshes: blue measured, orange filled in.")


def facing_the_arm(center) -> float:
    ''' Render azimuth (deg) looking at an object from the arm's base, the side the
    scan's cameras could reach best (far sides can be out of reach). '''
    return float(np.degrees(np.arctan2(-center[1], -center[0])))


def doubling_counts(n: int) -> list[int]:
    ''' 1, 2, 4, 8, ... up to (and always including) n. '''
    counts = [1]
    while counts[-1] * 2 < n:
        counts.append(counts[-1] * 2)
    return sorted(set(counts + [n]))
