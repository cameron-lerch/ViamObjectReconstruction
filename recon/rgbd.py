"""RGB-D frames: decoding, denoising, back-projection and the on-disk view format.

Each captured view is a folder holding color.jpg, depth.png (uint16, mm) and
view.json (camera pose in the world, intrinsics, joints, timestamps).
"""

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage

from .transforms import transform_points

_DEPTH_HEADER = b"DEPTHMAP"


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

    def to_o3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

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
    """Decode Viam's image/vnd.viam.dep format: header, int64 width and height, uint16 pixels (all big-endian)."""
    if not data.startswith(_DEPTH_HEADER):
        raise ValueError("Not a Viam depth map")
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    return np.frombuffer(data, ">u2", offset=24, count=width * height).reshape(height, width).astype(np.uint16)


def median_depth(frames: list[np.ndarray], min_valid_fraction: float = 0.5) -> np.ndarray:
    """Per-pixel median over valid readings; pixels valid in too few frames become 0."""
    stack = np.stack(frames).astype(np.float32)
    valid = stack > 0
    stack[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixels with no reading in any frame
        result = np.nanmedian(stack, axis=0)
    result[valid.mean(axis=0) < min_valid_fraction] = 0
    return np.nan_to_num(result).round().astype(np.uint16)


def edge_mask(depth: np.ndarray, relative_jump: float = 0.02, min_jump: float = 5.0, grow: int = 0) -> np.ndarray:
    """Pixels next to a depth discontinuity, widened by `grow` pixels.

    RealSense "flying pixels" sit right at the jump, floating between the near and
    far surface. A few pixels further in, stereo matching still blends the two
    ("edge fattening"), pushing an object's outline outward; `grow` drops those too.
    """
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
    """World points, their colors (0-1) and their (row, col) pixels for every valid depth
    reading away from depth edges (and `edge_grow` pixels beyond them)."""
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
    """Save a view; `color` is either encoded image bytes (with `color_ext`) or an RGB array."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(color, bytes):
        (folder / f"color.{color_ext}").write_bytes(color)
    else:
        o3d.io.write_image(str(folder / "color.jpg"), o3d.geometry.Image(np.ascontiguousarray(color)), quality=95)
    o3d.io.write_image(str(folder / "depth.png"), o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.uint16))))
    record = {"T_world_cam": T_world_cam.tolist(), "intrinsics": asdict(intrinsics), **(meta or {})}
    (folder / "view.json").write_text(json.dumps(record, indent=2))
    return folder


def load_view(folder: Path) -> View:
    folder = Path(folder)
    record = json.loads((folder / "view.json").read_text())
    color_path = next(folder.glob("color.*"))
    color = np.asarray(o3d.io.read_image(str(color_path)))[..., :3]
    depth = np.asarray(o3d.io.read_image(str(folder / "depth.png"))).astype(np.uint16)
    meta = {k: v for k, v in record.items() if k not in ("T_world_cam", "intrinsics")}
    return View(color, depth, np.array(record["T_world_cam"]), Intrinsics(**record["intrinsics"]), meta, folder.name)


def load_views(folder: Path) -> list[View]:
    return [load_view(path.parent) for path in sorted(Path(folder).glob("*/view.json"))]
