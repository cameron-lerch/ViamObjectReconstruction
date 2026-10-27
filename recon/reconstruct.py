"""Fuse the views of one object into a mesh.

1. Place every view in the world using the arm's pose at capture time.
2. Refine those poses with one joint point-to-plane ICP over all overlapping
   view pairs, holding each view near its arm pose with a prior. Registration
   uses the object plus a ring of table, which pins down height and tilt even
   for featureless or symmetric objects.
3. Integrate the depth maps, cropped to the object, into a TSDF volume. The
   signed-distance average over many views is what removes most sensor noise,
   and surfaces seen by too few views are dropped.
4. Extract the observed surface, then build a closed mesh with a flat bottom
   on the table (objects rest flat on it; see bottom.py), and measure how far
   each raw view lies from the observed surface.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .bottom import boundary_edge_count, close_with_flat_bottom, enclosed_volume
from .config import Config, ReconstructConfig
from .rgbd import View, back_project, to_point_cloud
from .scene import DetectedObject, TablePlane, min_area_rectangle
from .transforms import make_transform, rotation_angle_deg

log = logging.getLogger(__name__)


@dataclass
class Reconstruction:
    mesh: o3d.geometry.TriangleMesh  # closed, flat bottom
    observed: o3d.geometry.TriangleMesh  # only the surface the camera saw
    cloud: o3d.geometry.PointCloud
    refined_poses: list[np.ndarray]
    report: dict


def object_mask(points: np.ndarray, obj: DetectedObject, plane: TablePlane, settings: ReconstructConfig) -> np.ndarray:
    return obj.contains(points, settings.crop_margin) & (plane.height(points) > settings.table_clearance)


def local_table_plane(view: View, pose: np.ndarray, obj: DetectedObject, fallback: TablePlane,
                      config: Config) -> TablePlane:
    """The table as this view sees it, from a ring of table around the object.

    The survey's table comes from far away and inherits the camera calibration
    error, so near the object the real table can sit millimeters above it and
    leak into the crop. Fitting the table per view, in the same (refined) pose
    used for cropping, cancels that.
    """
    settings = config.reconstruct
    points, _, _ = back_project(view, 4, config.camera.min_depth, config.camera.max_depth, pose)
    ring = (obj.contains(points, settings.crop_margin + settings.table_ring)
            & ~obj.contains(points, settings.crop_margin) & (np.abs(fallback.height(points)) < 20))
    if ring.sum() < 200:
        return fallback
    (a, b, c, d), inliers = to_point_cloud(points[ring]).segment_plane(2.0, 3, 300)
    normal = np.array([a, b, c]) * np.sign(c)
    if len(inliers) < 100 or np.degrees(np.arccos(min(1.0, normal[2]))) > config.workspace.max_table_tilt_deg:
        return fallback
    return TablePlane(normal, float(d * np.sign(c)))


def registration_cloud(view: View, obj: DetectedObject, config: Config) -> o3d.geometry.PointCloud:
    settings = config.reconstruct
    points, colors, _ = back_project(view, settings.pixel_stride, config.camera.min_depth, config.camera.max_depth)
    keep = obj.contains(points, settings.context_margin)
    cloud = to_point_cloud(points[keep], colors[keep]).voxel_down_sample(settings.registration_voxel)
    cloud, _ = cloud.remove_statistical_outlier(20, 2.0)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * settings.registration_voxel, max_nn=30))
    cloud.orient_normals_towards_camera_location(view.camera_position)
    return cloud


def _view_direction(view: View, target: np.ndarray) -> np.ndarray:
    direction = view.camera_position - target
    return direction / np.linalg.norm(direction)


def _twist_to_matrix(twist: np.ndarray) -> np.ndarray:
    return make_transform(Rotation.from_rotvec(twist[:3]).as_matrix(), twist[3:])


def _matrix_to_twist(transform: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]])


def refine_poses(views: list[View], obj: DetectedObject, config: Config) -> tuple[list[np.ndarray], dict]:
    """Jointly align all views with point-to-plane ICP, using the arm poses as a prior.

    Each view gets a small rigid correction (in a frame centered on the object,
    so rotations turn about the object). Every pair of views facing roughly the
    same way contributes point-to-plane residuals; each correction is also
    pulled toward zero with the arm's expected accuracy. Gauss-Newton solves for
    all corrections together. Motions the geometry cannot observe, such as a
    cylinder spinning on its axis, simply stay at the arm pose.
    Returns world-frame corrections (apply as correction @ T_world_cam) and diagnostics.
    """
    settings = config.reconstruct
    to_object = make_transform(np.eye(3), -obj.center)
    clouds = [registration_cloud(view, obj, config).transform(to_object) for view in views]
    points = [np.asarray(c.points) for c in clouds]
    normals = [np.asarray(c.normals) for c in clouds]
    trees = [cKDTree(p) if len(p) else None for p in points]
    rng = np.random.default_rng(0)
    samples = [rng.permutation(len(p))[: settings.points_per_pair] for p in points]

    directions = [_view_direction(view, obj.center) for view in views]
    max_cos = np.cos(np.radians(settings.max_pair_angle_deg))
    pairs = [(i, j) for i in range(len(views)) for j in range(i + 1, len(views))
             if directions[i] @ directions[j] >= max_cos and trees[j] is not None and len(samples[i])]

    # Rotations are about the object's center; the third axis is (nearly) vertical, the
    # spin a round object cannot constrain, so it gets its own, tighter prior.
    tilt, spin = np.radians(settings.prior_rotation_deg), np.radians(settings.prior_spin_deg)
    prior = np.array([1 / tilt**2, 1 / tilt**2, 1 / spin**2] + [1 / settings.prior_translation_mm**2] * 3)
    transforms = [np.eye(4) for _ in views]
    count = 6 * len(views)
    pair_stats = {}
    for iteration in range(settings.icp_iterations):
        # Shrink the correspondence distance over the iterations, from coarse to fine.
        progress = iteration / max(1, settings.icp_iterations - 1)
        max_distance = settings.icp_max_distance * (2 - progress)
        hessian = np.diag(np.tile(prior, len(views)))
        gradient = np.concatenate([prior * _matrix_to_twist(T) for T in transforms])
        for i, j in pairs:
            source = points[i][samples[i]] @ transforms[i][:3, :3].T + transforms[i][:3, 3]
            local = (source - transforms[j][:3, 3]) @ transforms[j][:3, :3]
            distances, indices = trees[j].query(local, distance_upper_bound=max_distance, workers=-1)
            found = np.isfinite(distances)
            target_normals = normals[j][indices[found]] @ transforms[j][:3, :3].T
            source_normals = normals[i][samples[i][found]] @ transforms[i][:3, :3].T
            agree = (target_normals * source_normals).sum(axis=1) > 0.5
            if agree.sum() < settings.min_pair_overlap * len(samples[i]):
                pair_stats.pop((i, j), None)
                continue
            p = source[found][agree]
            n = target_normals[agree]
            q = points[j][indices[found][agree]] @ transforms[j][:3, :3].T + transforms[j][:3, 3]
            residual = ((p - q) * n).sum(axis=1)
            weight = np.minimum(1.0, settings.huber_mm / np.maximum(np.abs(residual), 1e-9))
            # Moving view j moves both its point and its normal; together that is
            # exactly the opposite of moving view i, so a shared motion changes nothing.
            jacobian_i = np.hstack([np.cross(p, n), n])
            jacobian_j = -jacobian_i
            si, sj = slice(6 * i, 6 * i + 6), slice(6 * j, 6 * j + 6)
            weighted_i, weighted_j = jacobian_i * weight[:, None], jacobian_j * weight[:, None]
            hessian[si, si] += weighted_i.T @ jacobian_i
            hessian[sj, sj] += weighted_j.T @ jacobian_j
            hessian[si, sj] += weighted_i.T @ jacobian_j
            hessian[sj, si] += weighted_j.T @ jacobian_i
            gradient[si] += weighted_i.T @ residual
            gradient[sj] += weighted_j.T @ residual
            pair_stats[(i, j)] = {"source": i, "target": j, "matches": int(agree.sum()),
                                  "rmse_mm": float(np.sqrt(np.mean(residual**2)))}
        step = np.linalg.solve(hessian + 1e-6 * np.eye(count), -gradient)
        transforms = [_twist_to_matrix(step[6 * k : 6 * k + 6]) @ T for k, T in enumerate(transforms)]
        steps = step.reshape(-1, 6)
        if np.abs(steps[:, :3]).max() < 1e-5 and np.abs(steps[:, 3:]).max() < 0.01:
            break

    corrections, per_view = [], []
    from_object = np.linalg.inv(to_object)
    for index, local in enumerate(transforms):
        shift = float(np.linalg.norm(local[:3, 3]))  # movement of the object center
        angle = rotation_angle_deg(local[:3, :3])
        rejected = shift > settings.max_correction_mm or angle > settings.max_correction_deg
        if rejected:
            log.warning("View %d correction %.1fmm / %.1fdeg is too large; keeping the arm pose", index, shift, angle)
            local = np.eye(4)
        corrections.append(from_object @ local @ to_object)
        per_view.append({"view": views[index].name, "shift_mm": shift, "rotation_deg": angle, "rejected": rejected})
    return corrections, {"views": per_view, "pairs": list(pair_stats.values()), "iterations": iteration + 1}


def fusion_device(name: str) -> o3d.core.Device:
    if name == "auto":
        name = "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"
    return o3d.core.Device(name)


def fuse(views: list[View], poses: list[np.ndarray], obj: DetectedObject, planes: list[TablePlane],
         config: Config) -> o3d.t.geometry.VoxelBlockGrid:
    """Integrate each view's object pixels into a TSDF voxel grid (millimeter units)."""
    settings = config.reconstruct
    device = fusion_device(settings.device)
    o3c = o3d.core
    grid = o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight", "color"), (o3c.float32, o3c.float32, o3c.float32), (1, 1, 3),
        settings.voxel, 16, 50000, device)
    trunc_voxels = settings.sdf_trunc / settings.voxel
    for view, pose, plane in zip(views, poses, planes):
        points, _, pixels = back_project(view, 1, config.camera.min_depth, config.camera.max_depth, pose)
        keep = object_mask(points, obj, plane, settings)
        depth = np.zeros_like(view.depth)
        rows, cols = pixels[keep].T
        depth[rows, cols] = view.depth[rows, cols]
        depth_image = o3d.t.geometry.Image(o3c.Tensor(depth)).to(device)
        color_image = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(view.color))).to(device)
        intrinsic = o3c.Tensor(view.intrinsics.matrix, o3c.float64)
        extrinsic = o3c.Tensor(np.linalg.inv(pose), o3c.float64)
        blocks = grid.compute_unique_block_coordinates(
            depth_image, intrinsic, extrinsic, 1.0, config.camera.max_depth, trunc_voxels)
        grid.integrate(blocks, depth_image, color_image, intrinsic, intrinsic, extrinsic,
                       1.0, config.camera.max_depth, trunc_voxels)
    return grid


def clean_mesh(mesh: o3d.geometry.TriangleMesh, settings: ReconstructConfig) -> o3d.geometry.TriangleMesh:
    """Drop small floating fragments and lightly smooth what remains."""
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    small = counts < settings.min_component_fraction * counts.max()
    mesh.remove_triangles_by_mask(small[labels])
    mesh.remove_unreferenced_vertices()
    if settings.smoothing_iterations:
        mesh = mesh.filter_smooth_taubin(settings.smoothing_iterations)
    mesh.compute_vertex_normals()
    return mesh


def surface_distances(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def view_deviation(views: list[View], poses: list[np.ndarray], mesh: o3d.geometry.TriangleMesh,
                   obj: DetectedObject, planes: list[TablePlane], config: Config) -> list[dict]:
    """How far each single view's raw object points lie from the fused surface."""
    results = []
    for view, pose, plane in zip(views, poses, planes):
        points, _, _ = back_project(view, 4, config.camera.min_depth, config.camera.max_depth, pose)
        points = points[object_mask(points, obj, plane, config.reconstruct)]
        if len(points) == 0 or len(mesh.triangles) == 0:
            continue
        distances = surface_distances(mesh, points)
        results.append({"view": view.name, "points": len(points),
                        "median_mm": float(np.median(distances)), "p90_mm": float(np.percentile(distances, 90))})
    return results


def combined_plane(planes: list[TablePlane], center: np.ndarray) -> TablePlane:
    """One table plane from the per-view ones: mean normal, through their mean height at the object."""
    normal = np.mean([p.normal for p in planes], axis=0)
    normal /= np.linalg.norm(normal)
    z = float(np.mean([p.z_at(*center[:2]) for p in planes]))
    return TablePlane(normal, -float(normal @ np.array([center[0], center[1], z])))


@dataclass
class FusedMesh:
    observed: o3d.geometry.TriangleMesh  # only what the camera saw (open where unseen)
    mesh: o3d.geometry.TriangleMesh  # closed, with a flat bottom on the table
    cloud: o3d.geometry.PointCloud
    table: TablePlane

    def summary(self) -> dict:
        mesh = self.mesh
        closed = len(mesh.triangles) > 0 and boundary_edge_count(mesh) == 0
        return {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                "extent_mm": (mesh.get_axis_aligned_bounding_box().get_extent().tolist() if len(mesh.vertices) else None),
                "closed": closed, "volume_mm3": enclosed_volume(mesh) if closed else None,
                "observed_triangles": len(self.observed.triangles),
                "observed_open_edges": boundary_edge_count(self.observed) if len(self.observed.triangles) else 0}


def build_mesh(views: list[View], poses: list[np.ndarray], planes: list[TablePlane], obj: DetectedObject,
               config: Config, min_views: int | None = None) -> FusedMesh:
    """Fuse posed views into the observed surface, then close it with a flat bottom."""
    settings = config.reconstruct
    # Each view adds 1 to a voxel's weight, and extraction keeps weights strictly
    # above the threshold, so "seen by at least n views" is a threshold of n - 0.5.
    weight = (settings.min_views if min_views is None else min_views) - 0.5
    grid = fuse(views, poses, obj, planes, config)
    observed = clean_mesh(grid.extract_triangle_mesh(weight).to_legacy(), settings)
    cloud = grid.extract_point_cloud(weight).to_legacy()
    table = combined_plane(planes, obj.center)
    mesh = (close_with_flat_bottom(cloud, table, settings.poisson_depth, settings.poisson_scale,
                                   settings.footprint_shrink, settings.base_band)
            if settings.close_bottom else observed)
    return FusedMesh(observed, mesh, cloud, table)


def refit_object(views: list[View], poses: list[np.ndarray], obj: DetectedObject, plane: TablePlane,
                 config: Config) -> DetectedObject:
    """Re-locate the object from the scan's own views.

    The box from the survey can be off by more than the crop margin (a survey
    taken with a miscalibrated camera mount placed the real cylinder 17mm
    away), and a box in the wrong place cuts a slice off the object in every
    view. Points above each view's own table, within a generous search margin
    of the old box, are clustered, and the box is refit around the main cluster.
    """
    settings = config.reconstruct
    gathered = []
    for view, pose in zip(views, poses):
        table = local_table_plane(view, pose, obj, plane, config)
        points, _, _ = back_project(view, 4, config.camera.min_depth, config.camera.max_depth, pose)
        keep = obj.contains(points, settings.refit_search) & (table.height(points) > settings.table_clearance + 3)
        gathered.append(points[keep])
    points = np.concatenate(gathered) if gathered else np.zeros((0, 3))
    if len(points) < 200:
        return obj
    cloud = to_point_cloud(points).voxel_down_sample(1.5)
    cloud, _ = cloud.remove_statistical_outlier(20, 2.0)
    points = np.asarray(cloud.points)
    labels = np.asarray(cloud.cluster_dbscan(4.0, 10))
    if labels.max() < 0:
        return obj
    # The biggest cluster whose center lies within the search region of the old box.
    clusters = [points[labels == k] for k in range(labels.max() + 1)]
    clusters = [c for c in clusters if np.linalg.norm(c.mean(axis=0)[:2] - obj.center[:2]) < obj.footprint_radius + settings.refit_search]
    if not clusters:
        return obj
    main = max(clusters, key=len)
    center_xy, extents, yaw = min_area_rectangle(main[:, :2])
    bottom = float(np.median([local_table_plane(v, T, obj, plane, config).z_at(*center_xy) for v, T in zip(views, poses)]))
    top = float(np.percentile(main[:, 2], 98))
    return DetectedObject(obj.id, np.array([*center_xy, (top + bottom) / 2]), np.array([*extents, top - bottom]),
                          yaw, len(main))


def refined_views(views: list[View], obj: DetectedObject, plane: TablePlane, config: Config,
                  refine: bool = True) -> tuple[DetectedObject, list[np.ndarray], list[TablePlane], dict]:
    """The object's box refit from the scan, camera poses after alignment, each view's
    local table, and the alignment report."""
    arm_poses = [view.T_world_cam for view in views]
    obj = refit_object(views, arm_poses, obj, plane, config)
    if refine and len(views) > 1:
        corrections, registration = refine_poses(views, obj, config)
        poses = [correction @ pose for correction, pose in zip(corrections, arm_poses)]
        obj = refit_object(views, poses, obj, plane, config)
    else:
        poses, registration = arm_poses, {}
    planes = [local_table_plane(view, pose, obj, plane, config) for view, pose in zip(views, poses)]
    return obj, poses, planes, registration


def reconstruct(views: list[View], obj: DetectedObject, plane: TablePlane, config: Config,
                refine: bool = True) -> Reconstruction:
    if not views:
        raise ValueError("No views to reconstruct from")
    survey_obj = obj
    obj, poses, planes, registration = refined_views(views, obj, plane, config, refine)
    fused = build_mesh(views, poses, planes, obj, config)
    deviation = view_deviation(views, poses, fused.observed, obj, planes, config)
    table_offsets = [float(p.z_at(*obj.center[:2]) - plane.z_at(*obj.center[:2])) for p in planes]
    report = {
        "object": obj.to_dict(),
        "survey_object": survey_obj.to_dict(),
        "views": len(views),
        "mesh": fused.summary(),
        "registration": registration,
        # Each view's local table height at the object relative to the survey's table.
        "table_offset_mm": table_offsets,
        "single_view_deviation": deviation,
        "median_single_view_deviation_mm": float(np.median([d["median_mm"] for d in deviation])) if deviation else None,
    }
    return Reconstruction(fused.mesh, fused.observed, fused.cloud, poses, report)


def write_meshes(folder: Path, mesh: o3d.geometry.TriangleMesh, observed: o3d.geometry.TriangleMesh) -> None:
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):  # OBJ drops normals; fine
        for name, item in (("mesh", mesh), ("observed", observed)):
            o3d.io.write_triangle_mesh(str(folder / f"{name}.ply"), item)
            o3d.io.write_triangle_mesh(str(folder / f"{name}.obj"), item)


def save_reconstruction(result: Reconstruction, folder: Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_meshes(folder, result.mesh, result.observed)
    o3d.io.write_point_cloud(str(folder / "cloud.ply"), result.cloud)
    (folder / "report.json").write_text(json.dumps(result.report, indent=2))
    (folder / "refined_poses.json").write_text(json.dumps([pose.tolist() for pose in result.refined_poses]))
