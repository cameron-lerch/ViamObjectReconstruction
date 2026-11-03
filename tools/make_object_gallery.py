''' The README's gallery of scanned objects: each object's final colored mesh beside the
points from all of its views, and a table of how each scan measured up (offscreen; no windows).

    python tools/make_object_gallery.py              # every object in OBJECTS
    python tools/make_object_gallery.py --only mug   # one object, by its key

Every scan is rebuilt with the current pipeline (config.toml), so the objects compare
directly; nothing in the session folders is changed. Writes media/images/objects/<key>.png
and media/images/objects/summary.json (the table's numbers).
'''

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon.config import load_config  # noqa: E402
from recon.experiment import Variant, evaluate, variant_views  # noqa: E402
from recon.bottom import table_scraps  # noqa: E402
from recon.figures import facing_the_arm  # noqa: E402
from recon.measure import load_truth  # noqa: E402
from recon.reconstruct import local_table_plane, object_mask, reconstruct  # noqa: E402
from recon.render import render, render_points  # noqa: E402
from recon.rgbd import back_project  # noqa: E402
from recon.scene import DetectedObject, TablePlane  # noqa: E402

OUT = ROOT / "media" / "images" / "objects"
# Each object's most recent scan (full 50-frame scans where they exist): key, name, session, object id,
# ground-truth entry (None: no calipers or CAD).
OBJECTS = [
    ("orange_block", "Orange block", "20261102-141308-orange-block", 0, "orange_block"),
    ("green_cylinder", "Green cylinder", "20261103-091105-green-cylinder", 0, "green_cylinder"),
    ("printed_block", "3D-printed block", "20261029-120949", 0, "printed_block"),
    ("printed_sphere", "3D-printed sphere", "20261029-124044", 0, "printed_sphere"),
    ("printed_bowl", "3D-printed bowl", "20261030-162717", 0, "printed_bowl"),
    ("pumpkin", "Pumpkin", "20261030-135302", 0, None),
    ("mug", "Mug", "20261030-141836", 0, None),
    ("cup", "Cup", "20261030-145232", 0, None),
]
PANEL = 520  # pixels per panel
RENDER_TRIANGLES = 40000  # the closed mesh is simplified to this many for drawing only
POINT_VOXEL = 1.0  # all views' points, thinned to one per 1mm cube for drawing


def colored(mesh: o3d.geometry.TriangleMesh, observed: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    ''' The closed mesh colored like the measured surface: each vertex takes the color of the
    nearest observed vertex (Poisson closing carries no colors). '''
    mesh = o3d.geometry.TriangleMesh(mesh)
    if observed.has_vertex_colors() and len(observed.vertices):
        _, nearest = cKDTree(np.asarray(observed.vertices)).query(np.asarray(mesh.vertices))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(observed.vertex_colors)[nearest])
    return mesh


def all_points(views, poses, obj: DetectedObject, plane, config) -> o3d.geometry.PointCloud:
    ''' Every view's points on the object, placed by the aligned poses and cropped the way
    fusion crops them (the object's box, above each view's own table). The low ring of
    table at the edge of the box is still in it (`without_table` removes it). '''
    clouds = o3d.geometry.PointCloud()
    for view, pose in zip(views, poses):
        table = local_table_plane(view, pose, obj, plane, config)
        points, colors, _ = back_project(view, 2, config.camera.min_depth, config.camera.max_depth, pose,
                                         edge_grow=config.reconstruct.edge_grow)
        keep = object_mask(points, obj, table, config.reconstruct)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep]))
        cloud.colors = o3d.utility.Vector3dVector(colors[keep])
        clouds += cloud
    return clouds.voxel_down_sample(POINT_VOXEL)


def without_table(cloud: o3d.geometry.PointCloud, obj: DetectedObject) -> o3d.geometry.PointCloud:
    ''' The cloud without the ring of table around the object: the same scraps the closing
    drops (low points outside the outline of everything higher; bottom.table_scraps). '''
    table = TablePlane(np.array([0.0, 0.0, 1.0]), -(obj.center[2] - obj.size[2] / 2))
    return cloud.select_by_index(np.flatnonzero(~table_scraps(np.asarray(cloud.points), table)))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/liberation/{name}", size)
    except OSError:
        return ImageFont.load_default()


def figure(name: str, mesh_image: np.ndarray, points_image: np.ndarray, subtitle: str) -> Image.Image:
    ''' Two panels side by side under a title: the final mesh and the cumulative point cloud. '''
    gap, top = 16, 92
    sheet = Image.new("RGB", (2 * PANEL + gap, PANEL + top), "white")
    sheet.paste(Image.fromarray(mesh_image), (0, top))
    sheet.paste(Image.fromarray(points_image), (PANEL + gap, top))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), name, fill=(31, 41, 51), font=font(30, bold=True))
    draw.text((12, 48), subtitle, fill=(82, 96, 109), font=font(18))
    for x, label in ((12, "Final mesh"), (PANEL + gap + 12, "Cumulative point cloud")):
        draw.text((x, top + 8), label, fill=(62, 76, 89), font=font(18, bold=True))
    return sheet


def build(entry, config, truth) -> dict:
    key, name, session, object_id, truth_name = entry
    variant = Variant(name, ROOT / "data" / session, object_id)
    views, obj, plane = variant_views(variant, config)
    result = reconstruct(views, obj, plane, config)
    row, _ = evaluate(result, truth_name, truth, ROOT) if truth_name else evaluate(result, None, {}, ROOT)
    fitted = DetectedObject.from_dict(result.report["object"])
    cloud = all_points(views, result.refined_poses, fitted, plane, config)

    mesh = colored(result.mesh, result.observed)
    if len(mesh.triangles) > RENDER_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(RENDER_TRIANGLES)
    mesh.compute_vertex_normals()
    center = np.asarray(mesh.vertices).mean(axis=0) if len(mesh.vertices) else fitted.center
    points = np.asarray(cloud.points)
    radius = float(np.linalg.norm(points - center, axis=1).max()) * 1.02 if len(points) else None
    azimuth = facing_the_arm(fitted.center)
    mesh_image = render(mesh, PANEL, 22, azimuth, center, radius, use_colors=True, wire_boundary=False)
    frames = views[0].meta.get("frames", 5)
    subtitle = f"{len(views)} views, {frames} frames per view"
    OUT.mkdir(parents=True, exist_ok=True)
    shown = without_table(cloud, fitted)
    points_image = render_points(np.asarray(shown.points), PANEL, 22, azimuth, center, radius, dot=2,
                                 colors=np.asarray(shown.colors))
    figure(name, mesh_image, points_image, subtitle).save(OUT / f"{key}.png", optimize=True)
    summary = {"key": key, "name": name, "session": session, "object": object_id, "views": len(views),
               "frames": frames, "points_drawn": len(points), "extent_mm": result.report["mesh"]["extent_mm"]}
    summary |= {k: v for k, v in row.items() if isinstance(v, (int, float, str, bool)) or v is None}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="one object's key, e.g. mug")
    args = parser.parse_args()
    config = load_config(ROOT / "config.toml")
    truth = load_truth(ROOT / "ground_truth.toml")
    entries = [e for e in OBJECTS if args.only in (None, e[0])]
    if not entries:
        raise SystemExit(f"No object {args.only!r}; keys: {', '.join(e[0] for e in OBJECTS)}")
    summary_path = OUT / "summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for entry in entries:
        print(f"{entry[1]} ({entry[2]})", flush=True)
        summaries[entry[0]] = build(entry, config, truth)
        summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
