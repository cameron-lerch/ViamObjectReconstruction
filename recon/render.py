''' Render meshes and point clouds to images offscreen (numpy z-buffer, no window or GPU).

Used for figures saved with each scan and in the README. Open the real files in
Blender to inspect them interactively.
'''

from pathlib import Path

import numpy as np
import open3d as o3d

from .transforms import look_at

BACKGROUND = np.array([255, 255, 255], dtype=float)
BASE_COLOR = np.array([0.72, 0.74, 0.78])


def render(mesh: o3d.geometry.TriangleMesh, size: int = 480, elevation_deg: float = 25.0,
           azimuth_deg: float = -60.0, center: np.ndarray | None = None, radius: float | None = None,
           use_colors: bool = False, wire_boundary: bool = True) -> np.ndarray:
    ''' Shaded image (size x size x 3, uint8) of a mesh seen from the given direction. '''
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    image = np.tile(BACKGROUND, (size, size, 1))
    if len(triangles) == 0:
        return image.astype(np.uint8)
    center = vertices.mean(axis=0) if center is None else center
    radius = np.linalg.norm(vertices - center, axis=1).max() if radius is None else radius
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    camera = look_at(center + direction * radius * 4, center)
    local = (vertices - camera[:3, 3]) @ camera[:3, :3]  # camera frame: x right, y down, z forward
    scale = size / (2.3 * radius) * 4 * radius  # orthographic-like: focal scaled to the distance
    px = local[:, 0] / local[:, 2] * scale + size / 2
    py = local[:, 1] / local[:, 2] * scale + size / 2

    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.4, -0.3, 0.85])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(normals @ light)
    if use_colors and mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)[triangles].mean(axis=1)
    else:
        colors = np.tile(BASE_COLOR, (len(triangles), 1))

    depth = np.full((size, size), np.inf)
    for k, tri in enumerate(triangles):
        xs, ys, zs = px[tri], py[tri], local[tri, 2]
        x0, x1 = max(int(xs.min()), 0), min(int(np.ceil(xs.max())), size - 1)
        y0, y1 = max(int(ys.min()), 0), min(int(np.ceil(ys.max())), size - 1)
        if x0 > x1 or y0 > y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        region = depth[y0 : y1 + 1, x0 : x1 + 1]
        nearer = inside & (z < region)
        region[nearer] = z[nearer]
        image[y0 : y1 + 1, x0 : x1 + 1][nearer] = colors[k] * shade[k] * 255

    if wire_boundary:
        # Mark open edges (holes) in red so gaps are easy to spot.
        directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        for a_index, b_index in directed[counts[inverse.ravel()] == 1]:
            for t in np.linspace(0, 1, 12):
                x = int(px[a_index] * (1 - t) + px[b_index] * t)
                y = int(py[a_index] * (1 - t) + py[b_index] * t)
                if 0 <= x < size and 0 <= y < size:
                    image[y, x] = (210, 40, 40)
    return image.astype(np.uint8)


def render_points(points: np.ndarray, size: int = 480, elevation_deg: float = 25.0, azimuth_deg: float = -60.0,
                  center: np.ndarray | None = None, radius: float | None = None, dot: int = 2,
                  colors: np.ndarray | None = None) -> np.ndarray:
    ''' Raw points as small dots, darker when farther, so a noisy cloud stays legible
    (in their own colors, 0-1, if given). '''
    image = np.tile(BACKGROUND, (size, size, 1))
    if len(points) == 0:
        return image.astype(np.uint8)
    center = points.mean(axis=0) if center is None else center
    radius = np.linalg.norm(points - center, axis=1).max() if radius is None else radius
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    camera = look_at(center + direction * radius * 4, center)
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    scale = size / (2.3 * radius) * 4 * radius
    px = (local[:, 0] / local[:, 2] * scale + size / 2).astype(int)
    py = (local[:, 1] / local[:, 2] * scale + size / 2).astype(int)
    depth = local[:, 2]
    shade = 0.25 + 0.6 * (depth - depth.min()) / max(np.ptp(depth), 1e-9)
    for k in np.argsort(-depth):  # far first, near drawn over it
        x, y = px[k], py[k]
        if 0 <= x < size - dot and 0 <= y < size - dot:
            if colors is not None:
                image[y : y + dot, x : x + dot] = colors[k] * 255 * (1 - 0.5 * shade[k])
            else:
                image[y : y + dot, x : x + dot] = np.array([0.35, 0.45, 0.6]) * 255 * (1 - shade[k]) + 255 * shade[k] * 0.35
    return image.astype(np.uint8)


def contact_sheet(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    height = max(i.shape[0] for i in images)
    sheet = np.full((height, sum(i.shape[1] for i in images) + gap * (len(images) - 1), 3), 255, np.uint8)
    x = 0
    for image in images:
        sheet[: image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return sheet


def save_png(image: np.ndarray, path: Path) -> None:
    o3d.io.write_image(str(path), o3d.geometry.Image(np.ascontiguousarray(image)))
