''' RGB-D frames: decoding, denoising, back-projection and the on-disk view format.

Each captured view is a folder holding color.jpg, depth.png (uint16, mm) and
view.json (camera pose in the world, intrinsics, joints, timestamps).
'''

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage

from .transforms import transform_points

_DEPTH_HEADER = b"DEPTHMAP"

# Open3D statistical outlier removal: a point is dropped when its mean distance to its
# 20 nearest neighbors is more than 2 standard deviations above the cloud's average.
# These are the values from Open3D's point-cloud tutorial; they were not tuned here.
OUTLIER_NEIGHBORS, OUTLIER_STD = 20, 2.0


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    @classmethod
    def from_viam(cls, params) -> "Intrinsics":
        return cls(
            params.width_px, params.height_px,
            params.focal_x_px, params.focal_y_px, params.center_x_px, params.center_y_px,
        )


@dataclass
class View:
    color: np.ndarray  # HxWx3 uint8
    depth: np.ndarray  # HxW uint16, millimeters, 0 = no reading
    T_world_cam: np.ndarray
    intrinsics: Intrinsics
    meta: dict = field(default_factory=dict)
    name: str = ""

    @property
    def camera_position(self) -> np.ndarray:
        return self.T_world_cam[:3, 3]


def decode_depth(data: bytes) -> np.ndarray:
    ''' Decode Viam's image/vnd.viam.dep format: header, int64 width and height, uint16 pixels (all big-endian). '''
    # Byte layout: 0-7 the 'DEPTHMAP' header, 8-15 width, 16-23 height, then the pixels.
    if not data.startswith(_DEPTH_HEADER):
        raise ValueError("Not a Viam depth map")
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    return np.frombuffer(data, ">u2", offset=24, count=width * height).reshape(height, width).astype(np.uint16)


def decode_depth_png(data: bytes) -> np.ndarray:
    ''' A depth image sent as a 16-bit PNG (millimeters), as some camera modules do. '''
    import cv2

    depth = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Depth image is not a single-channel 16-bit PNG")
    return depth


def median_depth(frames: list[np.ndarray], min_valid_fraction: float = 0.5) -> np.ndarray:
    ''' Per-pixel median over valid readings; pixels valid in too few frames become 0. '''
    stack = np.stack(frames).astype(np.float32)
    valid = stack > 0
    stack[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixels with no reading in any frame
        result = np.nanmedian(stack, axis=0)
    result[valid.mean(axis=0) < min_valid_fraction] = 0
    return np.nan_to_num(result).round().astype(np.uint16)


def incidence_deg(depth: np.ndarray, intrinsics: Intrinsics, step: int = 2) -> np.ndarray:
    ''' For each pixel, the angle (deg) between the surface's normal and the line of sight:
    0 seen head-on, near 90 seen edge-on. The normal comes from the depth image, from the
    points `step` pixels to either side; pixels without a reading (or a neighbor) get 90. '''
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = depth.astype(np.float64)
    points = np.stack([(u - intrinsics.cx) / intrinsics.fx * z, (v - intrinsics.cy) / intrinsics.fy * z, z], axis=-1)
    across = np.zeros_like(points)
    down = np.zeros_like(points)
    across[:, step:-step] = points[:, 2 * step:] - points[:, :-2 * step]
    down[step:-step] = points[2 * step:] - points[:-2 * step]
    normal = np.cross(across, down)
    length = np.linalg.norm(normal, axis=-1)
    ray = points / np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-9)
    cosine = np.abs(np.sum(normal * ray, axis=-1)) / np.maximum(length, 1e-9)
    angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    valid = (z > 0) & (length > 0)
    for shift in (-step, step):  # a neighbor without a reading spoils the normal
        valid &= np.roll(z, shift, axis=0) > 0
        valid &= np.roll(z, shift, axis=1) > 0
    return np.where(valid, angle, 90.0)


def edge_mask(depth: np.ndarray, relative_jump: float = 0.02, min_jump: float = 5.0, grow: int = 0) -> np.ndarray:
    ''' Pixels next to a depth discontinuity, widened by `grow` pixels.

    RealSense "flying pixels" sit right at the jump, floating between the near and
    far surface. A few pixels further in, stereo matching still blends the two
    ("edge fattening"), pushing an object's outline outward; `grow` drops those too.
    '''
    # A pixel is an edge when one of its 4 neighbors differs by more than min_jump (5mm)
    # or relative_jump (2%) of its depth, whichever is larger. 5mm is several times the
    # RealSense's noise between neighboring pixels on a smooth surface at scan range,
    # so surfaces aren't flagged; the 2% grows with distance, as stereo depth error
    # does. Both are judgment for the RealSense, not tuned (only `grow` was, see
    # ReconstructConfig.edge_grow); test again for another camera.
    padded = np.pad(depth.astype(np.float32), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    jump = np.zeros_like(center)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        neighbor = padded[dy : dy + depth.shape[0], dx : dx + depth.shape[1]]
        jump = np.maximum(jump, np.where(neighbor > 0, np.abs(neighbor - center), 0))
    # A pixel without a reading (a dropout) isn't an edge, however different its neighbors are.
    edges = (jump > np.maximum(min_jump, relative_jump * center)) & (center > 0)
    return ndimage.binary_dilation(edges, iterations=grow) if grow else edges


def pixel_rays(intrinsics: Intrinsics, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v, u = np.mgrid[0 : intrinsics.height : stride, 0 : intrinsics.width : stride]
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (v - intrinsics.cy) / intrinsics.fy
    return v, u, np.stack([x, y, np.ones_like(x)], axis=-1)


def back_project(view: View, stride: int = 1, min_depth: float = 0.0, max_depth: float = np.inf,
                 T_world_cam: np.ndarray | None = None, edge_grow: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' World points, their colors (0-1) and their (row, col) pixels for every valid depth
    reading away from depth edges (and `edge_grow` pixels beyond them). '''
    v, u, rays = pixel_rays(view.intrinsics, stride)
    depth = view.depth[::stride, ::stride].astype(float)
    edges = edge_mask(view.depth, grow=edge_grow)[::stride, ::stride]
    keep = (depth > min_depth) & (depth < max_depth) & (depth > 0) & ~edges
    points_cam = rays[keep] * depth[keep][:, None]
    pose = view.T_world_cam if T_world_cam is None else T_world_cam
    colors = view.color[v[keep], u[keep]].astype(float) / 255.0
    return transform_points(pose, points_cam), colors, np.stack([v[keep], u[keep]], axis=1)


def to_point_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def save_view(folder: Path, color: bytes | np.ndarray, depth: np.ndarray, T_world_cam: np.ndarray,
              intrinsics: Intrinsics, meta: dict | None = None, color_ext: str = "jpg") -> Path:
    ''' Save a view; `color` is either encoded image bytes (with `color_ext`) or an RGB array. '''
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(color, bytes):
        (folder / f"color.{color_ext}").write_bytes(color)
    else:
        o3d.io.write_image(str(folder / "color.jpg"), o3d.geometry.Image(np.ascontiguousarray(color)), quality=95)  # color is used for colors and the board, not geometry
    o3d.io.write_image(str(folder / "depth.png"), o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.uint16))))
    o3d.io.write_image(str(folder / "depth_preview.jpg"), o3d.geometry.Image(depth_preview(depth)), quality=85)
    record = {"T_world_cam": T_world_cam.tolist(), "intrinsics": asdict(intrinsics), **(meta or {})}
    (folder / "view.json").write_text(json.dumps(record, indent=2))
    return folder


def depth_preview(depth: np.ndarray) -> np.ndarray:
    ''' A depth image for looking at: near is red, far is blue (the "turbo" color map),
    no reading is black.

    depth.png holds millimeters as 16-bit numbers (the usual way to store depth, as
    the RealSense tools and RGB-D datasets do), so a viewer shows 300-600mm as 0.5-1%
    of full brightness: almost black. Colors follow each depth's rank among this
    image's readings (histogram equalization, as the RealSense Viewer does), so a
    patch of far floor doesn't squeeze everything else into one color. This copy is
    only for viewing; nothing reads it.
    '''
    import cv2

    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), np.uint8)
    ordered = np.sort(depth[valid])
    rank = np.searchsorted(ordered, depth, side="right") / len(ordered)
    level = ((1.0 - rank) * 255).astype(np.uint8)
    colored = np.ascontiguousarray(cv2.applyColorMap(level, cv2.COLORMAP_TURBO)[..., ::-1])
    colored[~valid] = 0
    return colored


def height_preview(view: "View", plane) -> np.ndarray:
    ''' The depth image colored by height above the table, for looking at: the table one
    flat light gray (within 8mm of it: the survey's plane is fitted in one place, and the
    real table wanders a few mm from it), things on it from blue (low) to red (their tops,
    the 98th percentile of what stands up), the floor and anything well below the table
    dark gray, no reading black. Unlike the depth preview, a flat table stays one color
    whatever the camera's angle. `plane` is the measured table (scene.TablePlane). '''
    import cv2

    _, _, rays = pixel_rays(view.intrinsics)
    depth = view.depth.astype(np.float64)
    camera_points = rays * depth[..., None]
    points = camera_points @ view.T_world_cam[:3, :3].T + view.T_world_cam[:3, 3]
    height = plane.height(points.reshape(-1, 3)).reshape(depth.shape)
    valid = depth > 0
    band = 8.0
    standing = valid & (height > band)
    top = max(30.0, float(np.percentile(height[standing], 98))) if standing.any() else 30.0
    level = (np.clip((height - band) / (top - band), 0, 1) * 255).astype(np.uint8)
    image = np.ascontiguousarray(cv2.applyColorMap(level, cv2.COLORMAP_TURBO)[..., ::-1])
    image[valid & (np.abs(height) <= band)] = (215, 220, 226)
    image[valid & (height < -band)] = (70, 74, 80)
    image[~valid] = 0
    return image


def save_frames(folder: Path, colors: list[bytes], depths: list[np.ndarray], color_ext: str = "jpg") -> None:
    ''' Every raw frame of a view, as captured: color_NNN.<ext> (the camera's own
    encoding) and depth_NNN.png (16-bit millimeters). '''
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for index, color in enumerate(colors):
        (folder / f"color_{index:03d}.{color_ext}").write_bytes(color)
    for index, depth in enumerate(depths):
        o3d.io.write_image(str(folder / f"depth_{index:03d}.png"), o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.uint16))))


def load_view(folder: Path, frames: int | None = None) -> View:
    ''' A saved view. With `frames` (a full scan's raw frames, in folder/frames/), the depth
    is rebuilt as the median of the first `frames` of them, exactly as a capture of that
    many frames would have made it, instead of the saved median of all of them. '''
    folder = Path(folder)
    record = json.loads((folder / "view.json").read_text())
    color_path = next(folder.glob("color.*"))
    color = np.asarray(o3d.io.read_image(str(color_path)))[..., :3]
    if frames is None:
        depth = np.asarray(o3d.io.read_image(str(folder / "depth.png"))).astype(np.uint16)
    else:
        paths = sorted((folder / "frames").glob("depth_*.png"))
        if len(paths) < frames:
            raise ValueError(f"{folder} has {len(paths)} raw depth frames, fewer than {frames} "
                             "(raw frames are only saved by scan.py --full-scan)")
        depth = median_depth([np.asarray(o3d.io.read_image(str(p))).astype(np.uint16) for p in paths[:frames]])
    meta = {k: v for k, v in record.items() if k not in ("T_world_cam", "intrinsics")}
    if frames is not None:
        meta["frames_used"] = frames
    return View(color, depth, np.array(record["T_world_cam"]), Intrinsics(**record["intrinsics"]), meta, folder.name)


def load_views(folder: Path, frames: int | None = None) -> list[View]:
    return [load_view(path.parent, frames) for path in sorted(Path(folder).glob("*/view.json"))]
