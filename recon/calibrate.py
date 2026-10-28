''' Estimate the camera mount (camera pose relative to the arm's end) from a scan.

A mount error moves the camera differently for every wrist orientation, so the
views of one static scene disagree with each other by an amount that depends
on the joints. One shared correction to the mount should make them all agree.
This finds it: point-to-plane distances between overlapping views (object plus
a wide ring of table) are minimized over the six mount parameters, with
correspondences recomputed as the estimate improves.
'''

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .config import Config
from .rgbd import OUTLIER_NEIGHBORS, OUTLIER_STD, View, back_project, to_point_cloud
from .scene import DetectedObject
from .transforms import make_transform, rotation_angle_deg

# Correspondence search distances, coarse to fine (mm): wide enough at first to pair
# up views that a 21mm mount error (found on armfarm14) pushes apart, ending at the
# 3mm the joint alignment uses.
SCHEDULE = (25.0, 15.0, 8.0, 5.0, 3.0)


@dataclass
class MountCalibration:
    mount: np.ndarray  # corrected camera pose relative to the arm's end
    original: np.ndarray
    rms_before_mm: float
    rms_after_mm: float
    matches: int

    @property
    def correction(self) -> np.ndarray:
        return np.linalg.inv(self.original) @ self.mount

    def describe(self) -> str:
        shift = np.linalg.norm(self.correction[:3, 3])
        angle = rotation_angle_deg(self.correction[:3, :3])
        return (f"mount correction {shift:.1f}mm / {angle:.2f}deg; view disagreement (RMS point-to-plane) "
                f"{self.rms_before_mm:.2f}mm -> {self.rms_after_mm:.2f}mm over {self.matches} matches")


def _camera_cloud(view: View, obj: DetectedObject, config: Config, ring: float):
    ''' Points (and normals) of the object and a ring of table, in the camera's own frame. '''
    points, _, _ = back_project(view, 2, config.camera.min_depth, config.camera.max_depth, edge_grow=config.reconstruct.edge_grow)
    keep = obj.contains(points, ring)
    cloud = to_point_cloud(points[keep]).voxel_down_sample(3.0)
    cloud, _ = cloud.remove_statistical_outlier(OUTLIER_NEIGHBORS, OUTLIER_STD)
    cloud.estimate_normals()
    cloud.orient_normals_towards_camera_location(view.camera_position)
    world_to_camera = np.linalg.inv(view.T_world_cam)
    cloud.transform(world_to_camera)
    return np.asarray(cloud.points), np.asarray(cloud.normals)


def _correction(params: np.ndarray) -> np.ndarray:
    return make_transform(Rotation.from_rotvec(params[:3]).as_matrix(), params[3:])


def calibrate_mount(views: list[View], mount: np.ndarray, obj: DetectedObject, config: Config,
                    ring: float = 60.0, samples: int = 400, max_pair_angle_deg: float = 70.0) -> MountCalibration:
    ''' `mount` is the camera pose relative to the arm's end used when the views were captured.

    `ring` is the table around the object (as the survey's clear_margin), `samples` the
    points used per view (enough for 6 parameters, fast), and only views within
    `max_pair_angle_deg` of each other are paired (farther apart they share little
    surface). All three are judgment, not tuned.
    '''
    ends = [view.T_world_cam @ np.linalg.inv(mount) for view in views]
    clouds = [_camera_cloud(view, obj, config, ring) for view in views]
    rng = np.random.default_rng(0)
    picks = [rng.permutation(len(points))[:samples] for points, _ in clouds]
    directions = [(v.camera_position - obj.center) / np.linalg.norm(v.camera_position - obj.center) for v in views]
    pairs = [(i, j) for i in range(len(views)) for j in range(len(views))
             if i != j and directions[i] @ directions[j] >= np.cos(np.radians(max_pair_angle_deg))]

    def world(index: int, points: np.ndarray, params: np.ndarray) -> np.ndarray:
        T = ends[index] @ mount @ _correction(params)
        return points @ T[:3, :3].T + T[:3, 3]

    def match(params: np.ndarray, distance: float):
        ''' Fixed correspondences (indices into each camera cloud) at the current estimate. '''
        trees = [cKDTree(world(k, clouds[k][0], params)) for k in range(len(views))]
        found = []
        for i, j in pairs:
            source = world(i, clouds[i][0][picks[i]], params)
            gaps, nearest = trees[j].query(source, distance_upper_bound=distance)
            ok = np.isfinite(gaps)
            if ok.sum() >= 20:  # a pair needs some overlap to say anything
                found.append((i, j, picks[i][ok], nearest[ok]))
        return found

    def residuals(params: np.ndarray, matches) -> np.ndarray:
        out = []
        for i, j, source_index, target_index in matches:
            T_j = ends[j] @ mount @ _correction(params)
            p = world(i, clouds[i][0][source_index], params)
            q = world(j, clouds[j][0][target_index], params)
            n = clouds[j][1][target_index] @ T_j[:3, :3].T
            out.append(((p - q) * n).sum(axis=1))
        return np.concatenate(out) if out else np.zeros(1)

    params = np.zeros(6)
    final = match(params, SCHEDULE[-1])
    rms_before = float(np.sqrt(np.mean(residuals(params, final) ** 2)))
    for distance in SCHEDULE:
        for _ in range(3):
            matches = match(params, distance)
            # Rotations are in radians, translations in mm: scale so both move comparably
            # (0.01rad turns a point 300mm away by 3mm). The robust loss (soft_l1) treats
            # residuals beyond about 1.5mm, a few times the views' noise, as likely mismatches.
            result = least_squares(residuals, params, args=(matches,), loss="soft_l1", f_scale=1.5,
                                   x_scale=np.array([0.01] * 3 + [1.0] * 3))
            converged = np.abs(result.x - params).max() < 1e-4
            params = result.x
            if converged:
                break
    final = match(params, SCHEDULE[-1])
    rms_after = float(np.sqrt(np.mean(residuals(params, final) ** 2)))
    return MountCalibration(mount @ _correction(params), mount, rms_before, rms_after,
                            int(sum(len(m[2]) for m in final)))


def apply_mount(views: list[View], old_mount: np.ndarray, new_mount: np.ndarray) -> list[View]:
    ''' Views with camera poses recomputed for a different mount (same arm poses). '''
    swap = np.linalg.inv(old_mount) @ new_mount
    return [View(v.color, v.depth, v.T_world_cam @ swap, v.intrinsics, v.meta, v.name) for v in views]
