"""Make the README's result figures from saved sessions (offscreen; no windows).

    python tools/make_figures.py [--real data/<session>] [--real-object 1] [--synthetic data/demo-<...>]

Figures are SVGs (rendered panels plus labels) in media/images/; when cairosvg
is installed (pip install cairosvg) a PNG copy is written next to each.
Run `python scan.py progression ...` and `python scan.py demo --progression`
first: the progression meshes are read from disk.
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from recon.config import Config, load_config  # noqa: E402
from recon.pipeline import scanned_object  # noqa: E402
from recon.reconstruct import reconstruct  # noqa: E402
from recon.rgbd import load_views  # noqa: E402
from recon.session import Session  # noqa: E402
from render_mesh import render, render_points, save_png  # noqa: E402

OUT = ROOT / "media" / "images"
PANEL = 300
MEASURED = np.array([0.62, 0.68, 0.76])
FILLED = np.array([0.93, 0.55, 0.20])


def png_base64(image: np.ndarray) -> str:
    path = OUT / "_panel.png"
    save_png(image, path)
    data = base64.b64encode(path.read_bytes()).decode()
    path.unlink()
    return data


def figure(name: str, title: str, panels: list[tuple[np.ndarray, str, str]], note: str = "") -> None:
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
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(parts))
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(OUT / f"{name}.png"))
    except ImportError:
        pass
    print("wrote", svg)


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


def progression_figure(name: str, title: str, folder: Path, counts: list[int], elevation: float, azimuth: float) -> None:
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
    figure(name, title, panels, "Observed meshes: 1 view keeps all it saw; 2+ views keep surface at least 2 of them agree on. Red lines mark open edges.")


def measured_figure(folder: Path, counts: list[int], elevation: float, azimuth: float) -> None:
    rows = {str(r["views"]): r for r in json.loads((folder / "summary.json").read_text())}
    center, radius = framing(o3d.io.read_triangle_mesh(str(folder / f"views_{max(counts):02d}" / "mesh.ply")))
    panels = []
    for k in counts:
        closed = o3d.io.read_triangle_mesh(str(folder / f"views_{k:02d}" / "mesh.ply"))
        observed = np.asarray(o3d.io.read_triangle_mesh(str(folder / f"views_{k:02d}" / "observed.ply")).vertices)
        if len(closed.triangles) == 0 or len(observed) == 0:
            continue
        for el, az, side in ((elevation, azimuth, "side"), (-55, azimuth, "underside")):
            panels.append((render(color_measured(closed, observed), PANEL, el, az, center, radius, use_colors=True, wire_boundary=False),
                           f"{k} view{'s' if k > 1 else ''}, {side}", f"{rows[str(k)]['measured_share']:.0%} measured, {rows[str(k)]['volume_cm3']:.1f}cm³"))
    figure("measured_vs_filled", "Closed meshes: measured surface (gray-blue) vs. filled in (orange)", panels,
           "The flat bottom is always filled in; with few views, so are most walls (straight down from the outline seen from above).")


def calibration_figure(session: Session, object_id: int) -> None:
    logging.disable(logging.WARNING)
    config = load_config(ROOT / "config.toml")
    obj = scanned_object(session, object_id)
    _, plane = session.load_scene()
    views = load_views(session.views_dir(object_id))
    calibration = json.loads((session.root / "mount_calibration.json").read_text())
    swap = np.linalg.inv(np.array(calibration["original"])) @ np.array(calibration["calibrated"])
    calibrated = [type(v)(v.color, v.depth, v.T_world_cam @ swap, v.intrinsics, v.meta, v.name) for v in views]
    before = reconstruct(views, obj, plane, config)
    after = reconstruct(calibrated, obj, plane, config)
    center, radius = framing(after.mesh)
    panels = []
    for label, result in (("configured camera mount", before), ("calibrated camera mount", after)):
        deviation = result.report["median_single_view_deviation_mm"]
        for az in (-60, 120):
            panels.append((render(result.observed, PANEL, 20, az, center, radius), label,
                           f"views deviate {deviation:.2f}mm (median)"))
    figure("calibration_before_after", "Real cylinder, 25 views: camera mount as configured vs. calibrated", panels,
           "Same 25 captures, same alignment and fusion; only the camera mount differs (21mm / 2.2° apart).")


def alignment_figure(old_session: Session) -> None:
    logging.disable(logging.WARNING)
    objects, plane = old_session.load_scene()
    obj = objects[1]
    old = o3d.io.read_triangle_mesh(str(old_session.recon_dir(1) / "mesh.ply"))
    new = reconstruct(load_views(old_session.views_dir(1)), obj, plane, Config())
    center, radius = framing(new.mesh)
    panels = [(render(old, PANEL, 15, az, center, radius, wire_boundary=False), "pairwise ICP + pose graph", "first approach")
              for az in (-60, 60)]
    panels += [(render(new.observed, PANEL, 15, az, center, radius, wire_boundary=False), "joint ICP with arm prior", "current approach")
               for az in (-60, 60)]
    figure("alignment_before_after", "Same 37 synthetic views, two ways of aligning them", panels,
           "Misaligned views average into ripples and double walls; well-aligned ones fuse into a smooth surface.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", default="data/20261026-143006")
    parser.add_argument("--real-object", type=int, default=1)
    parser.add_argument("--synthetic", default="data/demo-20261027-113147")
    parser.add_argument("--old-synthetic", default="data/demo-20261026-130458", help="a demo built with the first alignment approach")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    real = Session(ROOT / args.real)
    real_folder = real.recon_dir(args.real_object) / "progression"
    progression_figure("progression_real", "Real cylinder: one view vs. many", real_folder, [1, 3, 8, 25], 20, -60)
    measured_figure(real_folder, [1, 8, 25], 20, -60)
    synthetic = Session(ROOT / args.synthetic)
    progression_figure("progression_synthetic", "Synthetic cylinder (known shape): one view vs. many",
                       synthetic.recon_dir(1) / "progression", [1, 3, 8, 37], 20, -60)
    calibration_figure(real, args.real_object)
    alignment_figure(Session(ROOT / args.old_synthetic))


if __name__ == "__main__":
    main()
