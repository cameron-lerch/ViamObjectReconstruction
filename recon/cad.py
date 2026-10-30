''' Compare a scan with the CAD model of a 3D-printed object (calibration/printed_objects/).

Calipers check a few dimensions; the CAD model is the true shape everywhere. The CAD
mesh is placed on the table where the scan found the object, then the scan is fitted
onto it (best rigid fit: rotation and shift only, never scale, so a scan that is too big
stays too big), and every measured point's signed distance from the true surface is
taken: positive outside (the scan is too big there), negative inside.
'''

import numpy as np
import open3d as o3d

from .scene import DetectedObject, TablePlane

YAW_STEPS = 12  # starting rotations about the vertical tried by the fit (every 30deg)
FIT_DISTANCES = (10.0, 4.0, 2.0)  # coarse-to-fine correspondence distances (mm)


def load_cad(path) -> o3d.geometry.TriangleMesh:
    ''' The printed object's mesh (millimeters), with duplicate vertices merged so it is
    watertight and signed distances work. '''
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"No triangles in {path}")
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _yaw(angle_deg: float, about: np.ndarray) -> np.ndarray:
    a = np.radians(angle_deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = about - T[:3, :3] @ about
    return T


def place(cad: o3d.geometry.TriangleMesh, obj: DetectedObject, table: TablePlane) -> np.ndarray:
    ''' The CAD-to-world transform that stands the model on the table at the object's
    center (models have their resting face at z = 0, see the folder's README). '''
    vertices = np.asarray(cad.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    T = np.eye(4)
    T[:2, 3] = obj.center[:2] - (low[:2] + high[:2]) / 2
    T[2, 3] = table.z_at(*obj.center[:2]) - low[2]
    return T


def fit(cad: o3d.geometry.TriangleMesh, points: np.ndarray, start: np.ndarray) -> np.ndarray:
    ''' The CAD-to-world transform that best fits the model to the scanned `points`:
    rigid ICP (point to plane) from every starting rotation about the vertical, keeping
    the closest. '''
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.0, max_nn=30))
    source = cad.sample_points_uniformly(20000, use_triangle_normal=True)
    center = start[:3, 3] + np.asarray(cad.get_center())
    method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    best, best_score = start, np.inf
    for step in range(YAW_STEPS):
        T = _yaw(360.0 * step / YAW_STEPS, center) @ start
        for distance in FIT_DISTANCES:
            result = o3d.pipelines.registration.registration_icp(source, target, distance, T, method)
            T = result.transformation
        # Score by how close the scan is to the placed model (not the other way round:
        # the scan never sees the bottom, the model always has one).
        distances = np.abs(signed_distances(cad, points, T))
        score = float(np.median(distances))
        if score < best_score:
            best, best_score = T, score
    return best


def signed_distances(cad: o3d.geometry.TriangleMesh, points: np.ndarray, T_world_cad: np.ndarray) -> np.ndarray:
    ''' Each point's distance from the CAD surface (mm): positive outside, negative inside. '''
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cad))
    world_to_cad = np.linalg.inv(T_world_cad)
    local = points @ world_to_cad[:3, :3].T + world_to_cad[:3, 3]
    return scene.compute_signed_distance(o3d.core.Tensor(local.astype(np.float32))).numpy()


def compare_with_cad(cad: o3d.geometry.TriangleMesh, observed: o3d.geometry.TriangleMesh, obj: DetectedObject,
                     table: TablePlane) -> tuple[dict, np.ndarray, np.ndarray]:
    ''' Fit the CAD model to a scan's observed (measured) surface and measure the error.
    Returns (metrics, each observed vertex's signed error, the CAD-to-world fit). '''
    points = np.asarray(observed.vertices)
    T = fit(cad, points, place(cad, obj, table))
    errors = signed_distances(cad, points, T)
    absolute = np.abs(errors)
    metrics = {"cad_median_mm": float(np.median(absolute)), "cad_p90_mm": float(np.percentile(absolute, 90)),
               "cad_bias_mm": float(np.median(errors))}
    return metrics, errors, T


def error_colors(errors: np.ndarray, limit: float = 3.0) -> np.ndarray:
    ''' Blue where the scan is inside the true surface, white on it, red outside (+-limit mm). '''
    t = np.clip(errors / limit, -1.0, 1.0)[:, None]
    white, red, blue = np.ones(3), np.array([0.84, 0.19, 0.15]), np.array([0.17, 0.42, 0.78])
    return np.where(t >= 0, white + t * (red - white), white - t * (blue - white))
