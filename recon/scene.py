"""Find the table plane and the objects standing on it in a world-frame point cloud."""

import logging
import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree

from .config import WorkspaceConfig
from .transforms import make_transform, rotation_z

log = logging.getLogger(__name__)


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
    color: tuple[float, float, float] | None = None  # median RGB (0-1) of its points

    @property
    def color_name(self) -> str:
        return color_name(self.color) if self.color is not None else ""

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
        return DetectedObject(self.id, np.array([x, y, self.center[2]]), self.size.copy(), self.yaw_deg, self.n_points, self.color)

    def to_dict(self) -> dict:
        return {"id": self.id, "center": self.center.tolist(), "size": self.size.tolist(),
                "yaw_deg": self.yaw_deg, "n_points": self.n_points, "color": self.color, "color_name": self.color_name}

    @classmethod
    def from_dict(cls, data: dict) -> "DetectedObject":
        color = tuple(data["color"]) if data.get("color") else None
        return cls(data["id"], np.array(data["center"]), np.array(data["size"]), data["yaw_deg"], data["n_points"], color)

    def describe(self) -> str:
        x, y, z = self.center
        length, width, height = self.size
        named = f" ({self.color_name})" if self.color_name else ""
        return (f"object {self.id}{named}: center ({x:.0f}, {y:.0f}, {z:.0f}), "
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


def detect_objects(points: np.ndarray, plane: TablePlane, workspace: WorkspaceConfig,
                   colors: np.ndarray | None = None) -> list[DetectedObject]:
    heights = plane.height(points)
    keep = in_workspace(points, workspace) & (heights > workspace.min_object_height) & (heights < workspace.max_object_height)
    cloud = to_cloud(points[keep])
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors[keep])
    cloud = cloud.voxel_down_sample(2.0)
    if len(cloud.points) == 0:
        return []
    cloud, _ = cloud.remove_statistical_outlier(20, 2.0)
    labels = np.asarray(cloud.cluster_dbscan(workspace.cluster_distance, 10))
    candidates = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors) if cloud.has_colors() else None
    objects = []
    for cluster in merge_fragments([candidates[labels == label] for label in range(labels.max() + 1)]):
        if len(cluster) < workspace.min_object_points:
            continue
        center_xy, extents, yaw = min_area_rectangle(cluster[:, :2])
        bottom = plane.z_at(*center_xy)
        top = float(np.percentile(cluster[:, 2], 98))
        color = None
        if rgb is not None:
            # Colors of the candidate points that belong to this cluster.
            _, nearest = cKDTree(candidates).query(cluster)
            color = tuple(float(c) for c in np.median(rgb[nearest], axis=0))
        candidate = DetectedObject(len(objects), np.array([*center_xy, (top + bottom) / 2]),
                                   np.array([*extents, top - bottom]), yaw, len(cluster), color)
        reason = not_a_target(candidate, points, heights, workspace)
        if reason:
            log.info("  skipped something at (%.0f, %.0f), %.0f x %.0f x %.0fmm: %s",
                     *center_xy, *candidate.size, reason)
            continue
        objects.append(candidate)
    return objects


def table_support(obj: DetectedObject, points: np.ndarray, heights: np.ndarray, inner: float = 5.0,
                  outer: float = 60.0, spacing: float = 5.0) -> float:
    """Share of a ring around the object's footprint (from `inner` to `outer` mm out)
    that is bare table: points within 5mm of the table plane.

    Scan targets stand in open table. Cables at the table's edge or beside a power
    strip, and things in a gap between tables, have floor, gaps or other things
    around part of them.
    """
    table_xy = points[np.abs(heights) < 5.0][:, :2]
    if len(table_xy) == 0:
        return 0.0
    half = obj.size[:2] / 2 + outer
    u, v = np.meshgrid(np.arange(-half[0], half[0] + spacing, spacing), np.arange(-half[1], half[1] + spacing, spacing))
    local = np.column_stack([u.ravel(), v.ravel()])
    in_ring = np.any(np.abs(local) > obj.size[:2] / 2 + inner, axis=1)
    c, s = np.cos(np.radians(obj.yaw_deg)), np.sin(np.radians(obj.yaw_deg))
    ring = local[in_ring] @ np.array([[c, s], [-s, c]]) + obj.center[:2]
    gaps, _ = cKDTree(table_xy).query(ring, distance_upper_bound=spacing)
    return float(np.isfinite(gaps).mean())


def not_a_target(obj: DetectedObject, points: np.ndarray, heights: np.ndarray, workspace: WorkspaceConfig) -> str:
    """Why something on the table isn't a scan target ("" if it is one)."""
    if obj.size[2] < workspace.min_object_top:
        return f"under {workspace.min_object_top:.0f}mm tall (a cord or flat clutter)"
    if obj.size[0] > workspace.max_object_size:
        return f"over {workspace.max_object_size:.0f}mm across"
    support = table_support(obj, points, heights, outer=workspace.clear_margin)
    if support < workspace.min_table_support:
        return (f"not in open table (only {support:.0%} of the {workspace.clear_margin:.0f}mm around it is bare table): "
                "at an edge, in a gap, or beside something else")
    return ""


def merge_views(detections: list[tuple[np.ndarray, list[DetectedObject]]], gap: float = 15.0) -> list[DetectedObject]:
    """Combine per-view detections: each object comes from the view that sees it
    closest to straight below the camera, where it is least distorted.

    `detections` holds (camera xy, objects found in that view). A detection whose
    center lies within another, better-seen object's footprint (plus `gap`) is
    the same object, or a fragment of it.
    """
    ranked = sorted(((np.linalg.norm(obj.center[:2] - camera_xy), obj)
                     for camera_xy, objects in detections for obj in objects), key=lambda item: item[0])
    kept: list[DetectedObject] = []
    for _, obj in ranked:
        if all(np.linalg.norm(obj.center[:2] - other.center[:2]) > other.footprint_radius + gap for other in kept):
            kept.append(obj)
    return [DetectedObject(index, o.center, o.size, o.yaw_deg, o.n_points, o.color) for index, o in enumerate(kept)]


def color_name(rgb) -> str:
    """A rough color name from RGB (0-1), for telling objects apart in the survey output."""
    import colorsys
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    if saturation < 0.25:
        return "white" if value > 0.75 else ("black" if value < 0.2 else "gray")
    hue *= 360
    for limit, name in ((8, "red"), (30, "orange"), (75, "yellow"), (170, "green"), (260, "blue"), (340, "purple")):
        if hue < limit:
            return name
    return "red"
