''' Synthetic RGB-D views of known objects on a table, for testing without the robot.

Views get RealSense-like flaws: depth noise growing with distance squared,
dropouts, flying pixels at depth edges, and a fixed hand-eye calibration error
so that the reported camera poses are slightly wrong, as on the real arm.
'''

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .rgbd import Intrinsics, View, median_depth
from .transforms import make_transform, rotation_z

TABLE_Z = -8.0
INTRINSICS = Intrinsics(1280, 720, 908.2, 908.2, 646.6, 373.8)


@dataclass
class SceneObject:
    mesh: o3d.geometry.TriangleMesh
    color: tuple[float, float, float]


@dataclass
class SyntheticScene:
    objects: list[SceneObject] = field(default_factory=list)
    table_color: tuple[float, float, float] = (0.75, 0.75, 0.72)

    def __post_init__(self):
        table = o3d.geometry.TriangleMesh.create_box(1400, 1400, 20).translate((-300, -850, TABLE_Z - 20))
        self._meshes = [table] + [o.mesh for o in self.objects]
        self._colors = np.array([self.table_color] + [o.color for o in self.objects])
        self._scene = o3d.t.geometry.RaycastingScene()
        for mesh in self._meshes:
            self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def render(self, T_world_cam: np.ndarray, rng: np.random.Generator, intrinsics: Intrinsics = INTRINSICS,
               noise_at_300mm: float = 1.2, dropout: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(intrinsics.matrix), o3d.core.Tensor(np.linalg.inv(T_world_cam)),
            intrinsics.width, intrinsics.height)
        hits = self._scene.cast_rays(rays)
        rays = rays.numpy()
        t_hit = hits["t_hit"].numpy()
        geometry = hits["geometry_ids"].numpy()
        normals = hits["primitive_normals"].numpy()
        hit = np.isfinite(t_hit)

        points = rays[..., :3] + rays[..., 3:] * np.where(hit, t_hit, 0)[..., None]
        depth = (points - T_world_cam[:3, 3]) @ T_world_cam[:3, 2]
        depth[~hit] = 0
        depth += rng.normal(0, 1, depth.shape) * noise_at_300mm * (depth / 300.0) ** 2
        depth = _add_flying_pixels(depth, rng)
        depth[rng.random(depth.shape) < dropout] = 0

        directions = rays[..., 3:] / np.linalg.norm(rays[..., 3:], axis=-1, keepdims=True)
        shade = 0.35 + 0.65 * np.abs((normals * directions).sum(-1))
        base = self._colors[np.where(hit, geometry, 0)]
        color = np.clip(base * shade[..., None] * 255 + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        return color, np.clip(depth, 0, 65535).round().astype(np.uint16)


def _add_flying_pixels(depth: np.ndarray, rng: np.random.Generator, jump: float = 15.0) -> np.ndarray:
    ''' At depth edges, some pixels land somewhere between the near and far surface. '''
    padded = np.pad(depth, 1, mode="edge")
    neighbors = np.stack([padded[1:-1, :-2], padded[1:-1, 2:], padded[:-2, 1:-1], padded[2:, 1:-1]])
    far = neighbors.max(axis=0)
    near = np.where(neighbors > 0, neighbors, np.inf).min(axis=0)
    near = np.where(np.isfinite(near), near, far)
    edge = (far - near > jump) & (depth > 0) & (rng.random(depth.shape) < 0.5)
    mix = rng.random(depth.shape)
    return np.where(edge, near + mix * (far - near), depth)


def cylinder(x: float, y: float, radius: float, height: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=80, split=4)
    return mesh.translate((x, y, TABLE_Z + height / 2))


def box(x: float, y: float, size: tuple[float, float, float], yaw_deg: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(*size).translate(-np.array(size) / 2)
    mesh.rotate(rotation_z(yaw_deg), center=(0, 0, 0))
    return mesh.translate((x, y, TABLE_Z + size[2] / 2))


def demo_scene() -> SyntheticScene:
    return SyntheticScene([
        SceneObject(cylinder(430, -170, 25, 90), (0.1, 0.6, 0.25)),
        SceneObject(box(340, -40, (70, 40, 55), 30), (0.8, 0.35, 0.1)),
    ])


def hand_eye_error(translation_mm: float = 2.5, rotation_deg: float = 0.8, seed: int = 7) -> np.ndarray:
    ''' A fixed camera-mount error: true camera pose = reported pose @ error. '''
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    offset = rng.normal(size=3)
    offset *= translation_mm / np.linalg.norm(offset)
    return make_transform(Rotation.from_rotvec(np.radians(rotation_deg) * axis).as_matrix(), offset)


def render_views(scene: SyntheticScene, reported_poses: list[np.ndarray], seed: int = 0,
                 error: np.ndarray | None = None, frames: int = 1) -> list[View]:
    ''' Render each pose (with the hand-eye error applied) and pair it with the reported pose. '''
    rng = np.random.default_rng(seed)
    error = hand_eye_error() if error is None else error
    views = []
    for index, reported in enumerate(reported_poses):
        renders = [scene.render(reported @ error, rng) for _ in range(frames)]
        depth = median_depth([d for _, d in renders]) if frames > 1 else renders[0][1]
        views.append(View(renders[0][0], depth, reported, INTRINSICS, {"synthetic": True}, f"view_{index:03d}"))
    return views
