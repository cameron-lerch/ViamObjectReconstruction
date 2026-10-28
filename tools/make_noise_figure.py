"""How noisy one RealSense view is: the color image and the raw point cloud of a few
views of one object, next to all views together and the fused mesh (offscreen; no windows).

    python tools/make_noise_figure.py [--session data/<session>] [--object 0] [--views 4]

Each view's cloud is its saved depth (the per-pixel median of the frames captured
at that pose) back-projected with nothing removed: no edge filter, no outlier
removal. Every cloud is drawn from the same side-on direction, so the panels
compare directly. Writes media/images/<--name>.svg (and .png with cairosvg).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon.config import load_config  # noqa: E402
from recon.pipeline import scanned_object  # noqa: E402
from recon.rgbd import load_views, pixel_rays  # noqa: E402
from recon.session import Session  # noqa: E402
from recon.figures import png_base64, write_svg  # noqa: E402
from recon.render import render, render_points  # noqa: E402

OUT = ROOT / "media" / "images"

PANEL = 280
SIDE_ELEVATION, SIDE_AZIMUTH = 15.0, -60.0  # how every cloud is drawn


def raw_points(view, obj, margin: float = 25.0):
    """Every depth reading of the view near the object (plus a little table), with its color."""
    v, u, rays = pixel_rays(view.intrinsics)
    depth = view.depth.astype(float)
    keep = depth > 0
    points = rays[keep] * depth[keep][:, None]
    points = points @ view.T_world_cam[:3, :3].T + view.T_world_cam[:3, 3]
    colors = view.color[v[keep], u[keep]] / 255.0
    near = obj.contains(points, margin) & (points[:, 2] > obj.bottom_z - 10)
    return points[near], colors[near]


def color_crop(view, obj, pad: float = 0.6) -> np.ndarray:
    """The color image around the object, square, scaled to PANEL."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * obj.size / 2 + obj.center
    world_to_camera = np.linalg.inv(view.T_world_cam)
    local = corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    k = view.intrinsics
    u = k.fx * local[:, 0] / local[:, 2] + k.cx
    v = k.fy * local[:, 1] / local[:, 2] + k.cy
    half = (1 + pad) * max(np.ptp(u), np.ptp(v)) / 2
    cu, cv = u.mean(), v.mean()
    u0, v0 = int(max(0, cu - half)), int(max(0, cv - half))
    crop = view.color[v0 : int(min(k.height, cv + half)), u0 : int(min(k.width, cu + half))]
    rows = (np.arange(PANEL) * crop.shape[0] / PANEL).astype(int)
    cols = (np.arange(PANEL) * crop.shape[1] / PANEL).astype(int)
    return np.ascontiguousarray(crop[rows][:, cols])


def pick_views(views, count: int):
    """Spread over elevation and azimuth: the highest view first, then the views
    farthest (in viewing direction) from those already picked."""
    directions = np.array([v.T_world_cam[:3, 2] for v in views])
    order = [int(np.argmin(directions[:, 2]))]  # most nearly straight down
    while len(order) < min(count, len(views)):
        nearest = np.max(directions @ directions[order].T, axis=1)
        nearest[order] = np.inf
        order.append(int(np.argmin(nearest)))
    return [views[k] for k in order]


def grid_svg(name: str, title: str, columns: list[tuple[np.ndarray, np.ndarray, str, str]], note: str) -> None:
    """Columns of (top image, bottom image, label, sub-label)."""
    gap, top, bottom = 14, 52, 74
    width = len(columns) * PANEL + (len(columns) + 1) * gap
    height = top + 2 * PANEL + gap + bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
             f'<rect width="{width}" height="{height}" rx="14" fill="#FFFFFF"/>',
             f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="700" fill="#1F2933">{title}</text>']
    for index, (upper, lower, label, sub) in enumerate(columns):
        x = gap + index * (PANEL + gap)
        for row, image in enumerate((upper, lower)):
            y = top + row * (PANEL + gap)
            parts.append(f'<image x="{x}" y="{y}" width="{PANEL}" height="{PANEL}" href="data:image/png;base64,{png_base64(image)}"/>')
            parts.append(f'<rect x="{x}" y="{y}" width="{PANEL}" height="{PANEL}" fill="none" stroke="#CBD2D9"/>')
        y = top + 2 * PANEL + gap
        parts.append(f'<text x="{x + PANEL / 2}" y="{y + 22}" text-anchor="middle" font-size="16" font-weight="700" fill="#1F2933">{label}</text>')
        parts.append(f'<text x="{x + PANEL / 2}" y="{y + 42}" text-anchor="middle" font-size="13" fill="#52606D">{sub}</text>')
    parts.append(f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="13" font-style="italic" fill="#3E4C59">{note}</text>')
    parts.append("</svg>")
    write_svg(OUT / f"{name}.svg", parts)
    print("wrote", OUT / f"{name}.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=Path, help="default: the latest session")
    parser.add_argument("--object", type=int, default=0)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--name", default="raw_view_noise", help="output file name in media/images (no extension)")
    args = parser.parse_args()
    config = load_config(ROOT / "config.toml")
    session = Session(args.session) if args.session else Session.latest(config.resolve(config.data_dir))
    obj = scanned_object(session, args.object)
    views = load_views(session.views_dir(args.object))
    center = obj.center
    radius = 0.5 * np.linalg.norm(obj.size) + 10.0  # the object fills the panel; the table ring is cropped

    def side(points, colors=None):
        return render_points(points, PANEL, SIDE_ELEVATION, SIDE_AZIMUTH, center, radius, dot=1, colors=colors)

    columns = []
    for view in pick_views(views, args.views):
        points, colors = raw_points(view, obj)
        elevation = view.meta.get("elevation_deg", 90.0)
        azimuth = view.meta.get("azimuth_deg", 0.0)
        columns.append((color_crop(view, obj), side(points, colors),
                        f"{elevation:.0f}° up, {azimuth:.0f}° around", f"{len(points):,} raw points"))

    clouds = [raw_points(view, obj) for view in views]
    everything = np.concatenate([p for p, _ in clouds])
    shown = np.random.default_rng(0).permutation(len(everything))[:200_000]  # drawing every point is slow
    mesh_path = session.recon_dir(args.object) / "mesh.ply"
    if mesh_path.exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        fused = render(mesh, PANEL, SIDE_ELEVATION, SIDE_AZIMUTH, center, radius)
        fused_label = "below: the fused mesh"
    else:
        fused = np.full((PANEL, PANEL, 3), 255, np.uint8)
        fused_label = "(run reconstruct for the fused mesh)"
    columns.append((side(everything[shown]), fused, f"All {len(views)} views together",
                    f"{len(everything) / 1e6:.1f}M points; {fused_label}"))
    named = f" ({obj.color_name})" if obj.color_name else ""
    grid_svg(args.name, f"What the RealSense gives us, one view at a time: object {args.object}{named}",
             columns, "Top: the color image. Bottom: that view's depth as points, unfiltered, all drawn from the same side-on direction.")


if __name__ == "__main__":
    main()
