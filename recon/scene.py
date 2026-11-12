''' Find the table plane and the objects standing on it in a world-frame point cloud. '''

import logging
import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from .config import WorkspaceConfig
from .rgbd import OUTLIER_NEIGHBORS, OUTLIER_STD
from .transforms import make_transform, rotation_z

log = logging.getLogger(__name__)

# An object's top is the 98th percentile of its points' heights, not the highest:
# flying pixels and noise float above real tops (judgment, not tuned).
TOP_PERCENTILE = 98
# An object whose middle is past min_base_distance keeps its points up to this much
# closer to the base: a 150mm pumpkin 380mm out reached 305mm, and cutting it at 350mm
# left a narrow, off-center box whose cut-off part counted as 'not bare table' around it.
BASE_REACH = 100.0
# A box ignores this share of an object's points, the farthest from its middle, so a
# few stray points can't stretch or turn it (they stretched a pumpkin's 150mm to 165-207mm).
BOX_TRIM = 0.02
# Joining pieces of one object (a cable's plug and its coil): a pixel is "not table" when
# its color is this far (RGB, 0-1) from the table's color around it. Judgment, from the
# 2026-11-11 cable survey: the black cable is about 1 away, shadows and marks under 0.2.
NOT_TABLE_COLOR = 0.3
TABLE_HEIGHT = 3.0  # points within this height (mm) of the table give the table's color
# Growing a target's box over the object-colored pixels connected to it (a cable's flat
# loops): only points from GROW_MIN_HEIGHT (mm) up to the survey's min_object_height (the
# flat parts depth alone leaves out), at most GROW_LIMIT (mm) past the box from depth, and
# only when they add at least GROW_MIN_SHARE more points (and GROW_MIN_POINTS): shading
# at a solid object's base adds a few percent, a cable's loops more than the plug itself.
GROW_MIN_HEIGHT = 1.0
GROW_LIMIT = 150.0
GROW_MIN_POINTS = 30
GROW_MIN_SHARE = 0.25


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
        ''' Radius of the bounding sphere around the box. '''
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


def in_workspace(points: np.ndarray, workspace: WorkspaceConfig, reach: float = 0.0) -> np.ndarray:
    ''' Points in the table area (all of them before the survey has found it), at least
    min_base_distance - `reach` from the arm's base. '''
    inside = np.ones(len(points), dtype=bool)
    if workspace.x_range and workspace.y_range:
        (x0, x1), (y0, y1) = workspace.x_range, workspace.y_range
        inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return inside & (np.hypot(points[:, 0], points[:, 1]) >= workspace.min_base_distance - reach)


def fit_table_plane(points: np.ndarray, workspace: WorkspaceConfig) -> TablePlane:
    ''' RANSAC the dominant near-horizontal plane among workspace points. '''
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
    ''' Smallest rectangle around 2D points: center, (length, width), yaw of the long side in degrees. '''
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


def object_box(xy: np.ndarray, trim: float = BOX_TRIM) -> tuple[np.ndarray, np.ndarray, float]:
    ''' min_area_rectangle without the `trim` share of points farthest from the middle. '''
    distance = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
    inner = xy[distance <= np.quantile(distance, 1.0 - trim)]
    return min_area_rectangle(inner if len(inner) >= 3 else xy)


def merge_fragments(clusters: list[np.ndarray]) -> list[np.ndarray]:
    ''' Join clusters whose footprints overlap: one object split by shadows or missing depth. '''
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
                   colors: np.ndarray | None = None, skipped: list[DetectedObject] | None = None) -> list[DetectedObject]:
    ''' Scan targets standing on the table. Anything else standing on it (too big, too
    flat, crowded) is appended to `skipped`: not worth scanning, but still an obstacle. '''
    heights = plane.height(points)
    keep = in_workspace(points, workspace, BASE_REACH) & (heights > workspace.min_object_height) & (heights < workspace.max_object_height)
    cloud = to_cloud(points[keep])
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors[keep])
    cloud = cloud.voxel_down_sample(2.0)
    if len(cloud.points) == 0:
        return []
    cloud, _ = cloud.remove_statistical_outlier(OUTLIER_NEIGHBORS, OUTLIER_STD)
    labels = np.asarray(cloud.cluster_dbscan(workspace.cluster_distance, 10))
    candidates = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors) if cloud.has_colors() else None
    objects = []
    for cluster in merge_fragments([candidates[labels == label] for label in range(labels.max() + 1)]):
        if len(cluster) < workspace.min_object_points:
            continue
        center_xy, extents, yaw = object_box(cluster[:, :2])
        if np.hypot(*center_xy) < workspace.min_base_distance:
            continue  # the arm and the cables around its base
        bottom = plane.z_at(*center_xy)
        top = float(np.percentile(cluster[:, 2], TOP_PERCENTILE))
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
            if skipped is not None:
                skipped.append(candidate)
            continue
        objects.append(candidate)
    return objects


def table_support(obj: DetectedObject, points: np.ndarray, heights: np.ndarray, inner: float = 5.0,
                  outer: float = 60.0, spacing: float = 5.0) -> tuple[float, float]:
    ''' How open the table around an object is, over a ring around its footprint (from
    `inner` to `outer` mm out): (share of the ring the camera saw that is bare table,
    share of the ring the camera didn't see at all).

    Scan targets stand in open table. Cables at the table's edge or beside a power
    strip, and things in a gap between tables, have floor, gaps or other things
    around part of them. The floor beyond the table's edge is out of the camera's
    range, so it shows up as unseen. A tall object also hides some table behind it
    from the survey camera (a 200mm pumpkin hid 11% of its ring from 600mm up), so
    unseen cells aren't counted as 'not table'; instead, too many of them is its own test.
    '''
    table_xy = points[np.abs(heights) < 5.0][:, :2]  # 5mm: the survey's table noise from 600mm up
    if len(table_xy) == 0:
        return 0.0, 1.0
    half = obj.size[:2] / 2 + outer
    u, v = np.meshgrid(np.arange(-half[0], half[0] + spacing, spacing), np.arange(-half[1], half[1] + spacing, spacing))
    local = np.column_stack([u.ravel(), v.ravel()])
    in_ring = np.any(np.abs(local) > obj.size[:2] / 2 + inner, axis=1)
    c, s = np.cos(np.radians(obj.yaw_deg)), np.sin(np.radians(obj.yaw_deg))
    ring = local[in_ring] @ np.array([[c, s], [-s, c]]) + obj.center[:2]
    on_table = np.isfinite(cKDTree(table_xy).query(ring, distance_upper_bound=spacing)[0])
    seen = np.isfinite(cKDTree(points[:, :2]).query(ring, distance_upper_bound=spacing)[0])
    if not seen.any():
        return 0.0, 1.0
    return float(on_table[seen].mean()), float(1.0 - seen.mean())


def not_a_target(obj: DetectedObject, points: np.ndarray, heights: np.ndarray, workspace: WorkspaceConfig) -> str:
    ''' Why something on the table isn't a scan target ("" if it is one). '''
    if obj.size[2] < workspace.min_object_top:
        return f"under {workspace.min_object_top:.0f}mm tall (a cord or flat clutter)"
    support, unseen = table_support(obj, points, heights, outer=workspace.clear_margin)
    if support < workspace.min_table_support:
        return (f"not in open table (only {support:.0%} of the {workspace.clear_margin:.0f}mm around it is bare table): "
                "beside something else, or in a gap")
    if unseen > workspace.max_unseen_share:
        return (f"can't see enough table around it ({unseen:.0%} of the {workspace.clear_margin:.0f}mm around it is unseen): "
                "at the table's edge, or seen from too far to the side")
    return ""


def not_table_pixels(color: np.ndarray, pixels: np.ndarray, heights: np.ndarray, cell: int = 40) -> np.ndarray:
    ''' Pixels whose color stands out from the table around them (HxW bool).

    The table's color comes only from pixels whose depth puts them on the table
    (`pixels` and `heights` of the view's points), averaged over `cell`-pixel squares
    so slow shading across the table doesn't count; squares with too little table
    take the table's overall color. Objects never tint it, however big or dark.
    '''
    image = color.astype(float) / 255.0
    rows, cols = image.shape[:2]
    on = np.abs(heights) < TABLE_HEIGHT
    r, c = pixels[on, 0], pixels[on, 1]
    seen = image[r, c]
    overall = np.median(seen, axis=0) if len(seen) else np.median(image.reshape(-1, 3), axis=0)
    grid = (-(-rows // cell), -(-cols // cell))
    index = (r // cell) * grid[1] + c // cell
    count = np.bincount(index, minlength=grid[0] * grid[1])
    table = np.stack([np.bincount(index, seen[:, i], minlength=grid[0] * grid[1]) for i in range(3)], axis=1)
    table = np.where(count[:, None] >= 10, table / np.maximum(count, 1)[:, None], overall).reshape(*grid, 3)
    full = np.stack([ndimage.zoom(table[..., i], cell, order=1)[:rows, :cols] for i in range(3)], axis=-1)
    return np.linalg.norm(image - full, axis=2) > NOT_TABLE_COLOR


def join_connected(objects: list[DetectedObject], skipped: list[DetectedObject], points: np.ndarray,
                   pixels: np.ndarray, heights: np.ndarray, color: np.ndarray,
                   workspace: WorkspaceConfig) -> tuple[list[DetectedObject], list[DetectedObject]]:
    ''' Join skipped pieces to the scan target they're connected to in the image, and
    grow each target's box over the object-colored pixels connected to it.

    A cable's plug can stand tall enough to be a target while its coil is too flat
    and is skipped; the thin cable between them barely rises above the table's
    depth noise, so depth alone sees two things. In color the cable is unbroken.
    Strands lying flat (under `min_object_height`) aren't part of any piece at all,
    so the box also grows over those low points whose pixels belong to the target's
    run of not-table pixels, up to GROW_LIMIT mm past the box from depth.

    Color alone decides what's connected: a patch of bad depth beside an object has
    the table's color, so it never joins. Things with bare table between them stay
    apart, however close; a run of pixels touching two targets grows neither, and a
    piece touching two targets stays an obstacle. Returns (targets, still skipped).
    '''
    if not objects:
        return objects, skipped
    standing = heights > workspace.min_object_height
    mask = ndimage.binary_closing(not_table_pixels(color, pixels, heights), iterations=2)
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))

    def members(obj: DetectedObject) -> np.ndarray:
        return standing & obj.contains(points, 2.0)

    def regions(inside: np.ndarray) -> set[int]:
        found = labels[pixels[inside, 0], pixels[inside, 1]]
        return set(found[found > 0].tolist())

    target_regions = [regions(members(obj)) for obj in objects]
    joined = [members(obj) for obj in objects]
    left = []
    for piece in skipped:
        inside = members(piece)
        touching = [i for i, r in enumerate(target_regions) if r & regions(inside)]
        if len(touching) == 1:
            joined[touching[0]] |= inside
            target_regions[touching[0]] |= regions(inside)
            log.info("  joined the piece at (%.0f, %.0f), %.0f x %.0f x %.0fmm, to the object at (%.0f, %.0f): "
                     "connected in the image", *piece.center[:2], *piece.size, *objects[touching[0]].center[:2])
        else:
            left.append(piece)
    # Grow over the flat parts: points on the table's surface or above it (not below)
    # whose pixels are in a run of not-table pixels that only this target touches.
    shared = set()
    for i, r in enumerate(target_regions):
        shared |= {region for region in r if any(region in other for j, other in enumerate(target_regions) if j != i)}
    point_labels = labels[pixels[:, 0], pixels[:, 1]]
    lying = (heights > GROW_MIN_HEIGHT) & ~standing
    result = []
    for obj, inside, owned in zip(objects, joined, target_regions):
        own = list(owned - shared)
        if own:
            near = obj.contains(np.column_stack([points[:, :2], np.full(len(points), obj.center[2])]), GROW_LIMIT)
            grown = lying & near & np.isin(point_labels, own)
            new = (grown & ~inside).sum()
            if new >= max(GROW_MIN_POINTS, GROW_MIN_SHARE * inside.sum()):
                inside = inside | grown
        if inside.sum() == members(obj).sum():
            result.append(obj)
            continue
        cluster = points[inside]
        center_xy, extents, yaw = object_box(cluster[:, :2])
        bottom = obj.center[2] - obj.size[2] / 2
        top = max(obj.center[2] + obj.size[2] / 2, float(np.percentile(cluster[:, 2], TOP_PERCENTILE)))
        grew = DetectedObject(obj.id, np.array([*center_xy, (top + bottom) / 2]),
                              np.array([*extents, top - bottom]), yaw, len(cluster), obj.color)
        if np.any(grew.size[:2] > obj.size[:2] + 5):
            log.info("  object at (%.0f, %.0f): box grows from %.0f x %.0f to %.0f x %.0fmm over the connected "
                     "object-colored pixels", *obj.center[:2], *obj.size[:2], *grew.size[:2])
        result.append(grew)
    return result, left


def merge_views(detections: list[tuple[np.ndarray, list[DetectedObject]]], gap: float = 15.0) -> list[DetectedObject]:
    ''' Combine per-view detections: each object comes from the view that sees it
    closest to straight below the camera, where it is least distorted.

    `detections` holds (camera xy, objects found in that view). A detection whose
    center lies within another, better-seen object's footprint (plus `gap`) is
    the same object, or a fragment of it.
    '''
    ranked = sorted(((np.linalg.norm(obj.center[:2] - camera_xy), obj)
                     for camera_xy, objects in detections for obj in objects), key=lambda item: item[0])
    kept: list[DetectedObject] = []
    for _, obj in ranked:
        if all(np.linalg.norm(obj.center[:2] - other.center[:2]) > other.footprint_radius + gap for other in kept):
            kept.append(obj)
    return [DetectedObject(index, o.center, o.size, o.yaw_deg, o.n_points, o.color) for index, o in enumerate(kept)]


def views_agree(obj: DetectedObject, clouds: list[tuple[np.ndarray, np.ndarray]],
                cell: float = 5.0) -> tuple[int, int]:
    ''' How many survey views saw `obj`'s spot, and how many of them saw something standing there.

    `clouds` holds each view's points and their heights above that view's table. A view
    saw the spot if its points cover at least half of the object's box seen from above
    (in `cell`-mm squares), so a view that catches only a corner of a big box doesn't
    vote. It saw something standing if at least 10 of those points reach half the
    object's height: the whole box, not just its middle, so a bowl's rim counts (its
    middle is its floor), and a count, not a share, since a coiled cable covers only a
    small part of its box and its tallest part (a plug) may sit far from the center.
    '''
    total = max(1, int(np.ceil(obj.size[0] / cell)) * int(np.ceil(obj.size[1] / cell)))
    seen = standing = 0
    for points, heights in clouds:
        inside = obj.contains(np.column_stack([points[:, :2], np.full(len(points), obj.center[2])]))
        if inside.sum() < 5:
            continue
        local = (points[inside, :2] - obj.center[:2]) @ rotation_z(obj.yaw_deg)[:2, :2] + obj.size[:2] / 2
        covered = len(np.unique(np.floor(local / cell).astype(int), axis=0))
        if covered >= 0.5 * total:
            seen += 1
            standing += bool((heights[inside] >= 0.5 * obj.size[2]).sum() >= 10)
    return seen, standing


def clutter_around(objects: list[DetectedObject], skipped: list[DetectedObject], gap: float = 15.0) -> list[DetectedObject]:
    ''' The skipped things that aren't part of (or a fragment of) a scan target. '''
    return [c for c in skipped
            if all(np.linalg.norm(c.center[:2] - o.center[:2]) > o.footprint_radius + gap for o in objects)]


def color_name(rgb) -> str:
    ''' A rough color name from RGB (0-1), for telling objects apart in the survey output. '''
    import colorsys
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
    # Measured on armfarm14: the table and cables have saturation under 0.1, the test
    # objects' hues are about 11deg (orange), 43deg (yellow) and 154deg (green). The
    # boundaries below are the usual color wheel's, placed between those.
    # Near black, saturation means nothing: a black cable at (0.06, 0.06, 0.05) has
    # saturation 0.25 and read as "yellow" (2026-11-12), so darkness decides first.
    # Black cables measure 0.06-0.09 bright; the dark blue tape roll 0.15-0.2.
    if value < 0.12:
        return "black"
    if saturation < 0.25:
        return "white" if value > 0.75 else ("black" if value < 0.2 else "gray")
    hue *= 360
    for limit, name in ((8, "red"), (30, "orange"), (75, "yellow"), (170, "green"), (260, "blue"), (340, "purple")):
        if hue < limit:
            return name
    return "red"
