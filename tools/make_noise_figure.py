"""How noisy one RealSense view is: the color image and the raw point cloud of a few
views of one object, next to all views together and the fused mesh (offscreen; no windows).

    python tools/make_noise_figure.py [--session data/<session>] [--object 0] [--views 4]

Each view's cloud is its saved depth (the per-pixel median of the frames captured
at that pose) back-projected with nothing removed: no edge filter, no outlier
removal. Every cloud is drawn from the same side-on direction, so the panels
compare directly; each color image is turned (in 90° steps) so world up points
up, like the renders, and every panel has a small world x/y/z axis triad. Writes
media/images/<--name>.svg (and .png with cairosvg).
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
from recon.transforms import look_at  # noqa: E402

OUT = ROOT / "media" / "images"

PANEL = 280
SIDE_ELEVATION, SIDE_AZIMUTH = 15.0, -60.0  # how every cloud is drawn
AXES = np.eye(3)  # world x, y, z
AXIS_COLORS = ("#D64545", "#2F9E44", "#2F6FD6")


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


def color_crop(view, obj, pad: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """The color image around the object, square, scaled to PANEL, and the world
    axes as they appear in it (one 2D panel-pixel direction per row, y down)."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * obj.size / 2 + obj.center
    world_to_camera = np.linalg.inv(view.T_world_cam)
    k = view.intrinsics

    def project(points):
        local = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        return np.column_stack([k.fx * local[:, 0] / local[:, 2] + k.cx, k.fy * local[:, 1] / local[:, 2] + k.cy])

    uv = project(corners)
    half = (1 + pad) * max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / 2
    cu, cv = uv.mean(axis=0)
    u0, v0 = int(max(0, cu - half)), int(max(0, cv - half))
    crop = view.color[v0 : int(min(k.height, cv + half)), u0 : int(min(k.width, cu + half))]
    rows = (np.arange(PANEL) * crop.shape[0] / PANEL).astype(int)
    cols = (np.arange(PANEL) * crop.shape[1] / PANEL).astype(int)
    scale = np.array([PANEL / crop.shape[1], PANEL / crop.shape[0]])
    ends = project(np.vstack([obj.center, obj.center + 30.0 * AXES]))
    axes = (ends[1:] - ends[0]) * scale
    return np.ascontiguousarray(crop[rows][:, cols]), axes


def render_axes() -> np.ndarray:
    """The world axes as they appear in the side-on point and mesh renders."""
    a, e = np.radians(SIDE_AZIMUTH), np.radians(SIDE_ELEVATION)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    rotation = look_at(direction, np.zeros(3))[:3, :3]  # columns: image right, image down, forward
    return (AXES @ rotation)[:, :2]


def turned(image: np.ndarray, axes: np.ndarray, quarter_turns: int) -> tuple[np.ndarray, np.ndarray]:
    """The image turned counterclockwise in 90° steps, and its axis directions with it."""
    for _ in range(quarter_turns % 4):
        axes = np.column_stack([axes[:, 1], -axes[:, 0]])  # (right, down) -> (down, -right)
    return np.ascontiguousarray(np.rot90(image, quarter_turns)), axes


def match_turn(axes: np.ndarray) -> int:
    """Quarter turns (counterclockwise) that make world up (+z) point up in the image, so
    the table is at the bottom as in the renders. A view looking straight down has no
    up in the picture (z points at the camera), so it isn't turned."""
    z = axes[2]
    if np.hypot(*z) < 0.3 * np.linalg.norm(axes, axis=1).max():
        return 0
    return int(np.argmax([-turned(np.zeros((1, 1, 3)), axes, k)[1][2, 1] for k in range(4)]))


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


def triad(x: float, y: float, axes: np.ndarray, length: float = 20.0) -> list[str]:
    """SVG for small world x/y/z arrows from the bottom-left corner of a panel.
    An axis pointing (nearly) straight at or away from the viewer is a dot or a cross in a circle."""
    ox, oy = x + 26, y + PANEL - 26
    longest = max(np.linalg.norm(axes, axis=1).max(), 1e-9)
    parts = [f'<rect x="{x + 2}" y="{y + PANEL - 54}" width="52" height="52" rx="6" fill="#FFFFFF" fill-opacity="0.75"/>']
    for (dx, dy), color, name in zip(axes / longest * length, AXIS_COLORS, "xyz"):
        if np.hypot(dx, dy) < 0.3 * length:
            parts.append(f'<circle cx="{ox}" cy="{oy}" r="5" fill="none" stroke="{color}" stroke-width="1.6"/>')
            parts.append(f'<circle cx="{ox}" cy="{oy}" r="1.6" fill="{color}"/>')
            parts.append(f'<text x="{ox + 7}" y="{oy + 12}" font-size="11" font-weight="700" fill="{color}">{name}</text>')
            continue
        parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + dx:.1f}" y2="{oy + dy:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        parts.append(f'<text x="{ox + 1.3 * dx:.1f}" y="{oy + 1.3 * dy + 4:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="{color}">{name}</text>')
    return parts


def grid_svg(name: str, title: str, columns: list[tuple], note: str) -> None:
    """Columns of (top image, its axes, bottom image, its axes, label, sub-label)."""
    gap, top, bottom = 14, 52, 74
    width = len(columns) * PANEL + (len(columns) + 1) * gap
    height = top + 2 * PANEL + gap + bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
             f'<rect width="{width}" height="{height}" rx="14" fill="#FFFFFF"/>',
             f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="700" fill="#1F2933">{title}</text>']
    for index, (upper, upper_axes, lower, lower_axes, label, sub) in enumerate(columns):
        x = gap + index * (PANEL + gap)
        for row, (image, axes) in enumerate(((upper, upper_axes), (lower, lower_axes))):
            y = top + row * (PANEL + gap)
            parts.append(f'<image x="{x}" y="{y}" width="{PANEL}" height="{PANEL}" href="data:image/png;base64,{png_base64(image)}"/>')
            parts.append(f'<rect x="{x}" y="{y}" width="{PANEL}" height="{PANEL}" fill="none" stroke="#CBD2D9"/>')
            parts += triad(x, y, axes)
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

    side_axes = render_axes()
    columns = []
    for view in pick_views(views, args.views):
        points, colors = raw_points(view, obj)
        elevation = view.meta.get("elevation_deg", 90.0)
        azimuth = view.meta.get("azimuth_deg", 0.0)
        image, axes = color_crop(view, obj)
        quarter_turns = match_turn(axes)
        image, axes = turned(image, axes, quarter_turns)
        print(f"  {view.name}: color image turned {90 * quarter_turns} deg counterclockwise")
        columns.append((image, axes, side(points, colors), side_axes,
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
    columns.append((side(everything[shown]), side_axes, fused, side_axes, f"All {len(views)} views together",
                    f"{len(everything) / 1e6:.1f}M points; {fused_label}"))
    named = f" ({obj.color_name})" if obj.color_name else ""
    grid_svg(args.name, f"What the RealSense gives us, one view at a time: object {args.object}{named}",
             columns, "Top: the color image, turned so the table is at the bottom. Bottom: that view's depth as points, unfiltered, all drawn from the same side-on direction.")


if __name__ == "__main__":
    main()
