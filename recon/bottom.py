"""Build a closed (watertight) mesh with a flat bottom on the table.

The fused surface is open: the camera never sees the underside, fusion stops a
little above the table, and steep or hidden walls leave holes. Objects here
rest flat on the table, which gives two priors:

- the bottom is flat: the object's footprint (its outline seen from above),
  lying on the table plane;
- where a wall was never seen, it most likely drops straight down from that
  outline. Wall points are added only where no observed surface is nearby, so
  measurements always win and the prior fills only the gaps.

Screened Poisson reconstruction fits one closed surface through the observed
points (with their normals) plus these prior points, and anything that ends up
below the table is flattened onto it.
"""

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

from .scene import TablePlane

GRID = 1.0  # footprint grid spacing (mm)
WALL_GAP = 3.0  # prior wall points need no observed point within this distance (mm)
TOP_SEARCH = 6.0  # the wall's local height comes from observed points within this distance (mm)


def _table_axes(table: TablePlane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = table.normal / np.linalg.norm(table.normal)
    u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return normal, u, np.cross(normal, u)


def footprint(flat: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    """Outline seen from above, as a smoothed occupancy grid, and the grid's origin.

    Edge noise pads the outline, so it is shrunk by `shrink` mm.
    """
    low = flat.min(axis=0) - 4 * GRID
    cells = ((flat - low) / GRID).astype(int)
    grid = np.zeros(cells.max(axis=0) + 5, dtype=bool)
    grid[cells[:, 0], cells[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=2)
    grid = ndimage.binary_fill_holes(grid)
    grid = ndimage.binary_opening(grid)
    if shrink > 0:
        grid = ndimage.binary_erosion(grid, iterations=max(1, round(shrink / GRID)))
    return ndimage.gaussian_filter(grid.astype(float), 1.0), low


def prior_points(points: np.ndarray, table: TablePlane, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    """Flat-bottom points and (where unobserved) straight-wall points, with outward normals."""
    normal, u, v = _table_axes(table)
    flat = np.stack([points @ u, points @ v], axis=1)
    occupancy, low = footprint(flat, shrink)
    on_plane = -table.offset * normal

    def to_3d(xy: np.ndarray, height: np.ndarray) -> np.ndarray:
        return on_plane + xy[:, :1] * u + xy[:, 1:] * v + np.outer(height, normal)

    inside = np.argwhere(occupancy >= 0.5) * GRID + low + GRID / 2
    bottom = to_3d(inside, np.zeros(len(inside)))
    bottom_normals = np.tile(-normal, (len(bottom), 1))

    # The outline: cells straddling the 0.5 level, moved onto it along the gradient.
    gradient = np.stack(np.gradient(occupancy, GRID), axis=-1)
    edge = (occupancy > 0.15) & (occupancy < 0.85)
    cells = np.argwhere(edge)
    slope = gradient[edge]
    strength = np.linalg.norm(slope, axis=1)
    keep = strength > 1e-6
    cells, slope, strength = cells[keep], slope[keep], strength[keep]
    outward = -slope / strength[:, None]
    offset = (occupancy[edge][keep] - 0.5) / strength
    outline = cells * GRID + low + GRID / 2 + outward * offset[:, None]

    heights = table.height(points)
    near_top = cKDTree(flat).query_ball_point(outline, TOP_SEARCH)
    tops = np.array([heights[n].max() if n else heights.max() for n in near_top])
    wall, wall_normals = [], []
    for xy, direction, top in zip(outline, outward, tops):
        levels = np.arange(GRID / 2, top, GRID)
        wall.append(to_3d(np.tile(xy, (len(levels), 1)), levels))
        wall_normals.append(np.tile(direction[0] * u + direction[1] * v, (len(levels), 1)))
    if wall:
        wall, wall_normals = np.vstack(wall), np.vstack(wall_normals)
        gaps, _ = cKDTree(points).query(wall)
        wall, wall_normals = wall[gaps > WALL_GAP], wall_normals[gaps > WALL_GAP]
    else:
        wall, wall_normals = np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack([bottom, wall]), np.vstack([bottom_normals, wall_normals])


def boundary_edge_count(mesh: o3d.geometry.TriangleMesh) -> int:
    triangles = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def enclosed_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """Volume of a closed mesh (divergence theorem); tolerates tiny self-intersections."""
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6)


def close_with_flat_bottom(cloud: o3d.geometry.PointCloud, table: TablePlane, depth: int = 9,
                           scale: float = 3.0, shrink: float = 1.0, base_band: float = 5.0) -> o3d.geometry.TriangleMesh:
    """Closed mesh through the fused surface points, with a flat bottom on the table.

    `cloud` needs outward normals (the TSDF's point cloud has them). `scale` sizes
    Poisson's working box relative to the data; too tight a box leaves the surface
    open where it meets the box. The footprint comes from points above the base
    band, so it is the object's outline rather than the flared base.
    """
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    above = table.height(points) > 0
    if above.sum() < 50:
        return o3d.geometry.TriangleMesh()
    points, normals = points[above], normals[above]
    # Where wall meets table, mixed depth pixels flare outward; the straight-wall
    # prior replaces the lowest `base_band` mm of observed surface.
    clear = table.height(points) > base_band
    extra, extra_normals = prior_points(points[clear], table, shrink)
    points, normals = points[clear], normals[clear]
    samples = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([points, extra])))
    samples.normals = o3d.utility.Vector3dVector(np.vstack([normals, extra_normals]))
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(samples, depth=depth, scale=scale, linear_fit=True)
    # Poisson can add small detached bubbles; keep the object.
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(counts)))
    mesh.remove_unreferenced_vertices()
    # Flatten whatever dips below the table onto it.
    vertices = np.asarray(mesh.vertices)
    heights = table.height(vertices)
    below = heights < 0
    vertices[below] -= np.outer(heights[below], table.normal / np.linalg.norm(table.normal))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh
