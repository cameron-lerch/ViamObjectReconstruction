"""Compare reconstructions of one object from 1, 2, ... n views.

All views are aligned once (with every view available), then the first k views
are fused for each k, so each subset uses the same best-available camera poses
and only the number of views changes. For each k this writes a closed mesh
(flat bottom) and the observed surface, and measures:

- roughness: how far the surface wanders from a local plane (median RMS over
  3mm patches), a direct measure of leftover sensor noise. It needs no ground
  truth; a little of it is the object's real curvature. It is measured only on
  the part of the object the first view saw, which every subset includes, so
  rows compare like with like (a flat top is smoother than a curved wall).
- deviation from the n-view result: how far the observed surface lies from
  the final closed mesh (median and 90th percentile).
- completeness: the share of the n-view observed surface that this subset saw.
- measured share: the share of the closed mesh's surface that lies on measured
  data, as opposed to the flat-bottom and straight-wall fill.
- volume of the closed mesh.

Every subset uses the pipeline's own "seen by at least N views" filter, so with
fewer views than N the mesh is empty, and with few views it covers only where
they overlap. That is the point: one viewpoint cannot separate surface from
noise.
- error against the true shape, when ground truth is given (synthetic data),
  after the best rigid fit, so it measures shape rather than placement.

A first row ("raw") scores the raw points of the first view, i.e. one camera
frame with no fusion at all.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .config import Config
from .reconstruct import build_mesh, object_mask, refined_views, surface_distances, write_meshes
from .rgbd import View, back_project, to_point_cloud
from .scene import DetectedObject, TablePlane

log = logging.getLogger(__name__)

ROUGHNESS_RADIUS = 3.0
COMPLETENESS_TOLERANCE = 2.0


def spread_order(views: list[View], center: np.ndarray) -> list[int]:
    """Start with the view from highest above, then keep adding the view pointing
    most differently from those already chosen."""
    directions = np.array([(v.camera_position - center) / np.linalg.norm(v.camera_position - center) for v in views])
    order = [int(np.argmax(directions[:, 2]))]
    while len(order) < len(views):
        closeness = (directions @ directions[order].T).max(axis=1)
        closeness[order] = np.inf
        order.append(int(np.argmin(closeness)))
    return order


def roughness(points: np.ndarray, radius: float = ROUGHNESS_RADIUS, samples: int = 4000) -> float:
    """Median RMS distance of local patches from their best-fit plane."""
    if len(points) < 20:
        return float("nan")
    points = np.asarray(to_point_cloud(points).voxel_down_sample(radius / 4).points)
    tree = cKDTree(points)
    picks = np.random.default_rng(0).permutation(len(points))[:samples]
    spreads = []
    for neighbors in tree.query_ball_point(points[picks], radius):
        if len(neighbors) >= 8:
            patch = points[neighbors] - points[neighbors].mean(axis=0)
            spreads.append(np.sqrt(max(np.linalg.eigvalsh(patch.T @ patch / len(patch))[0], 0.0)))
    return float(np.median(spreads)) if spreads else float("nan")


def rigid_fit(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Best rigid transform of `points` onto `mesh`, to score shape apart from placement."""
    target = mesh.sample_points_uniformly(30000)
    target.estimate_normals()
    source = to_point_cloud(points)
    method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    return o3d.pipelines.registration.registration_icp(source, target, 5.0, np.eye(4), method).transformation


def shape_error(points: np.ndarray, truth: o3d.geometry.TriangleMesh) -> np.ndarray:
    fit = rigid_fit(points, truth)
    return surface_distances(truth, points @ fit[:3, :3].T + fit[:3, 3])


def near(points: np.ndarray, region: np.ndarray, distance: float = COMPLETENESS_TOLERANCE) -> np.ndarray:
    if len(points) == 0 or len(region) == 0:
        return points[:0]
    gaps, _ = cKDTree(region).query(points, distance_upper_bound=distance)
    return points[np.isfinite(gaps)]


def _score(points: np.ndarray, reference: o3d.geometry.TriangleMesh, truth: o3d.geometry.TriangleMesh | None,
           common: np.ndarray) -> dict:
    row = {"roughness_mm": roughness(near(points, common))}
    if len(points) and len(reference.triangles):
        d = surface_distances(reference, points)
        row |= {"deviation_median_mm": float(np.median(d)), "deviation_p90_mm": float(np.percentile(d, 90))}
    if truth is not None and len(points):
        e = shape_error(points, truth)
        row |= {"truth_median_mm": float(np.median(e)), "truth_p90_mm": float(np.percentile(e, 90))}
    return row


def measured_share(mesh: o3d.geometry.TriangleMesh, measured: np.ndarray) -> float:
    """Share of a closed mesh's surface lying on measured data; the rest is the
    flat-bottom and straight-wall fill."""
    samples = np.asarray(mesh.sample_points_uniformly(20000).points)
    gaps, _ = cKDTree(measured).query(samples, distance_upper_bound=COMPLETENESS_TOLERANCE)
    return float(np.isfinite(gaps).mean())


def progression(views: list[View], obj: DetectedObject, plane: TablePlane, config: Config, folder: Path,
                counts: list[int] | None = None, order: str = "spread", refine: bool = True,
                truth: o3d.geometry.TriangleMesh | None = None, min_views: int | None = None) -> list[dict]:
    """`min_views` defaults to the pipeline's own setting, so with fewer views than
    that the mesh is empty: one view cannot pass a "seen by at least 2 views" filter."""
    min_views = config.reconstruct.min_views if min_views is None else min_views
    folder = Path(folder)
    log.info("Aligning all %d views once", len(views))
    obj, poses, planes, _ = refined_views(views, obj, plane, config, refine)
    posed = [View(v.color, v.depth, pose, v.intrinsics, v.meta, v.name) for v, pose in zip(views, poses)]
    sequence = spread_order(posed, obj.center) if order == "spread" else list(range(len(views)))
    counts = sorted({k for k in (counts or range(1, len(views) + 1)) if 1 <= k <= len(views)})

    full = build_mesh(posed, poses, planes, obj, config)
    reference = full.mesh
    reference_samples = (np.asarray(full.observed.sample_points_uniformly(20000).points)
                         if len(full.observed.triangles) else np.zeros((0, 3)))

    first = sequence[0]
    raw, _, _ = back_project(posed[first], 2, config.camera.min_depth, config.camera.max_depth, poses[first])
    raw = raw[object_mask(raw, obj, planes[first], config.reconstruct)]
    (folder / "views_01").mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(folder / "views_01" / "raw_points.ply"), to_point_cloud(raw))
    # The patch every subset has seen: the final surface where the first view saw it.
    common = near(reference_samples, raw)
    rows = [{"views": "raw", "chosen": posed[first].name, **_score(raw, reference, truth, common)}]

    for k in counts:
        chosen = sequence[:k]
        fused = build_mesh([posed[i] for i in chosen], [poses[i] for i in chosen], [planes[i] for i in chosen],
                           obj, config, min_views=min_views)
        out = folder / f"views_{k:02d}"
        out.mkdir(parents=True, exist_ok=True)
        write_meshes(out, fused.mesh, fused.observed)
        observed_points = np.asarray(fused.observed.vertices)
        row = {"views": k, "chosen": " ".join(posed[i].name.removeprefix("view_") for i in chosen)}
        row |= _score(observed_points, reference, truth, common)
        summary = fused.summary()
        row["observed_triangles"] = summary["observed_triangles"]
        row["closed_triangles"] = summary["triangles"]
        row["volume_cm3"] = summary["volume_mm3"] / 1000 if summary["volume_mm3"] else None
        if len(fused.mesh.triangles) and len(fused.cloud.points):
            row["measured_share"] = measured_share(fused.mesh, np.asarray(fused.cloud.points))
        if len(reference_samples) and len(fused.observed.triangles):
            row["completeness"] = float((surface_distances(fused.observed, reference_samples) < COMPLETENESS_TOLERANCE).mean())
        if truth is not None and len(fused.mesh.triangles):
            e = shape_error(np.asarray(fused.mesh.vertices), truth)
            row |= {"closed_truth_median_mm": float(np.median(e)), "closed_truth_p90_mm": float(np.percentile(e, 90))}
        rows.append(row)
        log.info("  %s", format_row(row))

    write_summary(rows, folder)
    return rows


COLUMNS = [("views", "views", "{}"), ("observed_triangles", "seen tris", "{}"), ("roughness_mm", "rough", "{:.3f}"),
           ("deviation_median_mm", "dev med", "{:.2f}"), ("deviation_p90_mm", "dev p90", "{:.2f}"),
           ("completeness", "complete", "{:.0%}"), ("measured_share", "measured", "{:.0%}"), ("volume_cm3", "vol cm3", "{:.1f}"),
           ("truth_median_mm", "true med", "{:.2f}"), ("truth_p90_mm", "true p90", "{:.2f}"),
           ("closed_truth_median_mm", "closed true med", "{:.2f}")]


def format_row(row: dict) -> str:
    parts = []
    for key, label, fmt in COLUMNS:
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            parts.append(f"{label} {fmt.format(value)}")
    return ", ".join(parts)


def write_summary(rows: list[dict], folder: Path) -> None:
    (folder / "summary.json").write_text(json.dumps(rows, indent=2))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (folder / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict]) -> str:
    """Rows as a fixed-width text table, for the terminal."""
    present = [(key, label, fmt) for key, label, fmt in COLUMNS if any(row.get(key) is not None for row in rows)]
    lines = ["  ".join(f"{label:>8}" for _, label, _ in present)]
    for row in rows:
        cells = []
        for key, _, fmt in present:
            value = row.get(key)
            cells.append(f"{'-' if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value):>8}")
        lines.append("  ".join(cells))
    return "\n".join(lines)
