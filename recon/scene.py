"""Find the table plane and the objects standing on it in a world-frame point cloud."""

import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from .config import WorkspaceConfig
from .transforms import make_transform, rotation_z


@dataclass
class TablePlane:
    normal: np.ndarray  # unit, pointing up
    offset: float  # normal @ p + offset == 0 on the table

    def height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset

    def z_at(self, x: float, y: float) -> float:
        a, b, c = self.normal
        return -(a * x + b * y + self.offset) / c

    def to_dict(self) -> dict:
        return {"normal": self.normal.tolist(), "offset": self.offset}

    @classmethod
    def from_dict(cls, data: dict) -> "TablePlane":
        return cls(np.array(data["normal"]), float(data["offset"]))


@dataclass
class DetectedObject:
    id: int
    center: np.ndarray  # world xyz of the box center
    size: np.ndarray  # box extents: length (along yaw), width, height
    yaw_deg: float
    n_points: int

    @property
    def top_z(self) -> float:
        return float(self.center[2] + self.size[2] / 2)

    @property
    def bottom_z(self) -> float:
        return float(self.center[2] - self.size[2] / 2)

    @property
    def radius(self) -> float:
        """Radius of the bounding sphere around the box."""
        return float(np.linalg.norm(self.size) / 2)

    @property
    def footprint_radius(self) -> float:
        return float(np.linalg.norm(self.size[:2]) / 2)

    @property
    def pose(self) -> np.ndarray:
        return make_transform(rotation_z(self.yaw_deg), self.center)

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        local = (points - self.center) @ rotation_z(self.yaw_deg)
        return np.all(np.abs(local) <= self.size / 2 + margin, axis=1)

    def moved_to(self, x: float, y: float) -> "DetectedObject":
        return DetectedObject(self.id, np.array([x, y, self.center[2]]), self.size.copy(), self.yaw_deg, self.n_points)

    def to_dict(self) -> dict:
        return {"id": self.id, "center": self.center.tolist(), "size": self.size.tolist(),
                "yaw_deg": self.yaw_deg, "n_points": self.n_points}

    @classmethod
    def from_dict(cls, data: dict) -> "DetectedObject":
        return cls(data["id"], np.array(data["center"]), np.array(data["size"]), data["yaw_deg"], data["n_points"])

    def describe(self) -> str:
        x, y, z = self.center
        length, width, height = self.size
        return (f"object {self.id}: center ({x:.0f}, {y:.0f}, {z:.0f}), "
                f"{length:.0f} x {width:.0f} x {height:.0f}mm, yaw {self.yaw_deg:.0f}deg")


def in_workspace(points: np.ndarray, workspace: WorkspaceConfig) -> np.ndarray:
    (x0, x1), (y0, y1) = workspace.x_range, workspace.y_range
    inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return inside & (np.hypot(points[:, 0], points[:, 1]) >= workspace.min_base_distance)


def fit_table_plane(points: np.ndarray, workspace: WorkspaceConfig) -> TablePlane:
    """RANSAC the dominant near-horizontal plane among workspace points."""
    remaining = to_cloud(points[in_workspace(points, workspace)])
    for _ in range(3):
        if len(remaining.points) < 1000:
            break
        (a, b, c, d), inliers = remaining.segment_plane(workspace.table_distance, 3, 1000)
        normal = np.array([a, b, c])
        if c < 0:
            normal, d = -normal, -d
        tilt = math.degrees(math.acos(min(1.0, normal[2] / np.linalg.norm(normal))))
        if tilt <= workspace.max_table_tilt_deg:
            return TablePlane(normal / np.linalg.norm(normal), d / np.linalg.norm(normal))
        remaining = remaining.select_by_index(inliers, invert=True)
    raise RuntimeError("No horizontal table plane found in the workspace")


def to_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def min_area_rectangle(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Smallest rectangle around 2D points: center, (length, width), yaw of the long side in degrees."""
    hull = xy[ConvexHull(xy).vertices]
    best = None
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end - start
        angle = math.atan2(edge[1], edge[0])
        c, s = math.cos(angle), math.sin(angle)
        local = hull @ np.array([[c, -s], [s, c]])
        low, high = local.min(axis=0), local.max(axis=0)
        area = np.prod(high - low)
        if best is None or area < best[0]:
            mid = (low + high) / 2
            best = (area, np.array([[c, -s], [s, c]]) @ mid, high - low, math.degrees(angle))
    _, center, extents, yaw = best
    if extents[1] > extents[0]:
        extents, yaw = extents[::-1], yaw + 90
    yaw = (yaw + 90) % 180 - 90  # a box looks the same after a half turn
    return center, extents, yaw


def merge_fragments(clusters: list[np.ndarray]) -> list[np.ndarray]:
    """Join clusters whose footprints overlap: one object split by shadows or missing depth."""
    clusters = [c for c in clusters if len(c) >= 3]
    merged = True
    while merged:
        merged = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                low = np.maximum(clusters[a][:, :2].min(0), clusters[b][:, :2].min(0))
                high = np.minimum(clusters[a][:, :2].max(0), clusters[b][:, :2].max(0))
                if np.all(low < high):
                    clusters[a] = np.vstack([clusters[a], clusters.pop(b)])
                    merged = True
                    break
            if merged:
                break
    return clusters


def detect_objects(points: np.ndarray, plane: TablePlane, workspace: WorkspaceConfig) -> list[DetectedObject]:
    heights = plane.height(points)
    keep = in_workspace(points, workspace) & (heights > workspace.min_object_height) & (heights < workspace.max_object_height)
    cloud = to_cloud(points[keep]).voxel_down_sample(2.0)
    if len(cloud.points) == 0:
        return []
    cloud, _ = cloud.remove_statistical_outlier(20, 2.0)
    labels = np.asarray(cloud.cluster_dbscan(workspace.cluster_distance, 10))
    candidates = np.asarray(cloud.points)
    objects = []
    for cluster in merge_fragments([candidates[labels == label] for label in range(labels.max() + 1)]):
        if len(cluster) < workspace.min_object_points:
            continue
        center_xy, extents, yaw = min_area_rectangle(cluster[:, :2])
        bottom = plane.z_at(*center_xy)
        top = float(np.percentile(cluster[:, 2], 98))
        if top - bottom < workspace.min_object_top:
            continue
        objects.append(DetectedObject(
            id=len(objects),
            center=np.array([*center_xy, (top + bottom) / 2]),
            size=np.array([*extents, top - bottom]),
            yaw_deg=yaw,
            n_points=len(cluster),
        ))
    return objects
