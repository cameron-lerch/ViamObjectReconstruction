"""Figures made from saved results: rows of rendered panels with labels, as SVG
(plus a PNG copy when cairosvg is installed). Used after each scan and for the README."""

import base64
import json
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

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
    """A row of panels with a label and a sub-label under each."""
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


def color_measured(mesh: o3d.geometry.TriangleMesh, measured: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Gray-blue where the closed mesh lies on measured data, orange where it was filled in."""
    gaps, _ = cKDTree(measured).query(np.asarray(mesh.vertices))
    colored = o3d.geometry.TriangleMesh(mesh)
    colored.vertex_colors = o3d.utility.Vector3dVector(np.where((gaps < 2.0)[:, None], MEASURED, FILLED))
    return colored


def framing(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, float]:
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    return center, float(np.linalg.norm(vertices - center, axis=1).max())


def progression_figure(folder: Path, svg: Path, title: str, counts: list[int],
                       elevation: float = 20.0, azimuth: float = -60.0) -> None:
    """One view's raw points, the observed mesh from each number of views in `counts`
    (from `progression`'s output in `folder`), and the closed mesh from the most."""
    rows = {str(r["views"]): r for r in json.loads((folder / "summary.json").read_text())}
    last = max(counts)
    closed = o3d.io.read_triangle_mesh(str(folder / f"views_{last:02d}" / "mesh.ply"))
    center, radius = framing(closed)
    raw = np.asarray(o3d.io.read_point_cloud(str(folder / "views_01" / "raw_points.ply")).points)
    panels = [(render_points(raw, PANEL, elevation, azimuth, center, radius), "1 view: raw points",
               f"roughness {rows['raw']['roughness_mm']:.2f}mm")]
    for k in counts:
        observed = o3d.io.read_triangle_mesh(str(folder / f"views_{k:02d}" / "observed.ply"))
        row = rows[str(k)]
        sub = (f"{row.get('completeness', 0):.0%} seen, roughness {row['roughness_mm']:.2f}mm"
               if row.get("roughness_mm") == row.get("roughness_mm") and row.get("roughness_mm") is not None else "nothing seen")
        panels.append((render(observed, PANEL, elevation, azimuth, center, radius), f"{k} view{'s' if k > 1 else ''}: observed", sub))
    cloud = np.asarray(o3d.io.read_triangle_mesh(str(folder / f"views_{last:02d}" / "observed.ply")).vertices)
    panels.append((render(color_measured(closed, cloud), PANEL, elevation, azimuth, center, radius, use_colors=True),
                   f"{last} views: closed", f"{rows[str(last)]['measured_share']:.0%} measured (orange = filled in)"))
    row_figure(svg, title, panels, "Observed meshes: 1 view keeps all it saw; 2+ views keep surface at least 2 of them agree on. Red lines mark open edges.")


def doubling_counts(n: int) -> list[int]:
    """1, 2, 4, 8, ... up to (and always including) n."""
    counts = [1]
    while counts[-1] * 2 < n:
        counts.append(counts[-1] * 2)
    return sorted(set(counts + [n]))
