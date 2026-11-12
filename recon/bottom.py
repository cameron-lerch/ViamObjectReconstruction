''' Build a closed (watertight) mesh with a flat bottom on the table.

The fused surface is open: the camera never sees the underside, fusion stops a
little above the table, and steep or hidden walls leave holes. Objects here
rest flat on the table, which gives two priors:

- the bottom is flat: the object's footprint (its outline seen from above),
  lying on the table plane;
- where a wall was never seen, it most likely drops straight down from the
  lowest place the surface was seen. On a box that is the outline itself; on a
  rounded object (a pumpkin) the surface curves back in below its widest point,
  and the wall drops from where it was last seen, tucked in underneath, instead
  of from the widest outline. Wall points are added only where no observed
  surface is nearby, so measurements always win and the prior fills only the gaps.

Screened Poisson reconstruction fits one closed surface through the observed
points (with their normals) plus these prior points, and anything that ends up
below the table is flattened onto it.

Holes are filled only where nothing was seen through them: where some view's depth
landed on the table inside the outline (the middle of a roll of tape, gaps in a
coiled cable), the outline keeps the hole, so it gets walls and no bottom.
'''

import logging

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

from .scene import TablePlane

log = logging.getLogger(__name__)

GRID = 1.0  # footprint grid spacing (mm)
WALL_GAP = 3.0  # prior wall points need no observed point within this distance (mm)
TOP_SEARCH = 6.0  # the wall's local height comes from observed points within this distance (mm)
# Tucking the wall in under a rounded object (judgment, checked on the 2026-10-28 pumpkin):
CORRIDOR = 3.0  # observed points within this distance (mm) of the inward line from an outline point
MAX_TUCK = 60.0  # how far inside the outline (mm) the lowest seen surface may lie
FACING = 0.3  # a point counts only if its normal faces outward along the line (cos, about 73deg)
LOWEST_SHARE = 0.05  # the lowest point is the 5th percentile by height, so one stray reading can't set it
SMOOTH = 6.0  # tuck and height are medians over outline points within this distance (mm)
# Only the first surface met going inward counts: a gap this long (mm) along the line
# ends it. Inside an open cup, the line otherwise reaches the far side's inner wall
# (also facing that way), and tucking the wall 58mm in cut away the whole flat bottom.
SURFACE_GAP = 10.0
# Table scraps: points this low (mm) that lie outside the outline of everything higher,
# by more than SCRAP_MARGIN (mm). Judgment, from the second 2026-10-28 pumpkin scan.
# For a low object the limit drops to SCRAP_SHARE of its height: a cable lying flat
# (strands about 8mm tall, 24mm where loops cross) is mostly under 15mm, and lost 64%
# of its points as "scraps" on 2026-11-11.
SCRAP_HEIGHT = 15.0
SCRAP_SHARE = 0.2
SCRAP_MARGIN = 3.0
# Closing keeps every piece with at least this share of the largest piece's triangles
# (a cable's plug end can close as its own piece); Poisson's small bubbles are dropped.
KEEP_PIECE = 0.05
# A closed mesh reaching further than this (mm) past the points it was fit to has
# ballooned (Poisson does on thin slivers, such as 2 views of a pumpkin's top).
BALLOON_MARGIN = 20.0
# Seeing the table through a hole (judgment, from the 2026-11-11 tape roll): depth pixels
# within this distance (mm) of the table count as table, and a hole in the outline stays
# open when the table was seen in at least this many of its cells.
SEEN_TABLE_BAND = 2.0
MIN_SEEN_CELLS = 5


def _table_axes(table: TablePlane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = table.normal / np.linalg.norm(table.normal)
    # Any vector not parallel to the normal seeds the cross product; avoid a near-parallel one.
    u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return normal, u, np.cross(normal, u)


def footprint(flat: np.ndarray, shrink: float, seen: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    ''' Outline seen from above, as a smoothed occupancy grid, and the grid's origin.

    Edge noise pads the outline, so it is shrunk by `shrink` mm. Holes in it are filled,
    except those where the table was `seen` (table points, in the same 2D coordinates).
    '''
    # A few cells of padding so closing and blurring never touch the grid's border.
    low = flat.min(axis=0) - 4 * GRID
    cells = ((flat - low) / GRID).astype(int)
    grid = np.zeros(cells.max(axis=0) + 5, dtype=bool)
    grid[cells[:, 0], cells[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=2)
    filled = ndimage.binary_fill_holes(grid)
    if seen is not None and len(seen):
        holes, count = ndimage.label(filled & ~grid)
        at = np.floor((seen - low) / GRID).astype(int)
        inside = np.all((at >= 0) & (at < grid.shape), axis=1)
        table = np.zeros(grid.shape, dtype=bool)
        table[at[inside, 0], at[inside, 1]] = True
        hits = ndimage.sum(table, holes, index=np.arange(1, count + 1))
        open_holes = np.flatnonzero(hits >= MIN_SEEN_CELLS) + 1
        if len(open_holes):
            log.info("  %d hole(s) left open: the table was seen through them", len(open_holes))
        filled &= ~np.isin(holes, open_holes)
    grid = filled
    grid = ndimage.binary_opening(grid)
    if shrink > 0:
        grid = ndimage.binary_erosion(grid, iterations=max(1, round(shrink / GRID)))
    return ndimage.gaussian_filter(grid.astype(float), 1.0), low


def lowest_seen(outline: np.ndarray, outward: np.ndarray, flat: np.ndarray, heights: np.ndarray,
                flat_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ''' For each outline point: how far inward (mm) the lowest outward-facing observed
    surface lies, and its height (NaN where nothing was seen along that line).

    The search looks inward along the line from the outline point, among points whose
    normal faces outward along it, so the top (facing up) and the opposite wall
    (facing the other way) don't count.
    '''
    tuck = np.full(len(outline), np.nan)
    height = np.full(len(outline), np.nan)
    tree = cKDTree(flat)
    for i, (xy, direction) in enumerate(zip(outline, outward)):
        near = np.array(tree.query_ball_point(xy, MAX_TUCK + CORRIDOR), dtype=int)
        if len(near) == 0:
            continue
        offset = flat[near] - xy
        along = -(offset @ direction)  # how far inward
        across = np.abs(offset @ np.array([-direction[1], direction[0]]))
        facing = flat_normals[near] @ direction
        on_line = (along > -CORRIDOR) & (along < MAX_TUCK) & (across < CORRIDOR) & (facing > FACING)
        hit, depth = near[on_line], along[on_line]
        if len(hit) == 0:
            continue
        order = np.argsort(depth)
        gaps = np.flatnonzero(np.diff(depth[order]) > SURFACE_GAP)
        if len(gaps):
            hit = hit[order[:gaps[0] + 1]]
        order = hit[np.argsort(heights[hit])]
        lowest = order[int(LOWEST_SHARE * (len(order) - 1))]
        tuck[i] = max(0.0, -((flat[lowest] - xy) @ direction))
        height[i] = heights[lowest]
    # Smooth along the outline (medians of neighbors), so the wall's foot doesn't zigzag.
    seen = np.isfinite(tuck)
    if seen.any():
        neighbors = cKDTree(outline[seen]).query_ball_point(outline, SMOOTH)
        tuck_seen, height_seen = tuck[seen], height[seen]
        for i, n in enumerate(neighbors):
            if n and seen[i]:
                tuck[i], height[i] = np.median(tuck_seen[n]), np.median(height_seen[n])
    return tuck, height


def prior_points(points: np.ndarray, table: TablePlane, shrink: float,
                 normals: np.ndarray | None = None,
                 seen_table: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    ''' Flat-bottom points and (where unobserved) straight-wall points, with outward normals.

    With `normals`, each wall drops from the lowest place the surface was seen,
    and the flat bottom shrinks to match (see the module docstring).
    '''
    normal, u, v = _table_axes(table)
    flat = np.stack([points @ u, points @ v], axis=1)
    seen = None if seen_table is None else np.stack([seen_table @ u, seen_table @ v], axis=1)
    occupancy, low = footprint(flat, shrink, seen)
    on_plane = -table.offset * normal

    def to_3d(xy: np.ndarray, height: np.ndarray) -> np.ndarray:
        return on_plane + xy[:, :1] * u + xy[:, 1:] * v + np.outer(height, normal)

    # The outline: cells straddling the 0.5 level, moved onto it along the gradient.
    gradient = np.stack(np.gradient(occupancy, GRID), axis=-1)
    # Cells partly inside the blurred outline straddle its 0.5 level (15-85% is a band
    # a couple of cells wide).
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
    feet = outline
    if normals is not None and len(outline):
        tuck, seen_height = lowest_seen(outline, outward, flat, heights, np.stack([normals @ u, normals @ v], axis=1))
        tucked = np.isfinite(tuck)
        feet = outline - np.where(tucked, tuck, 0.0)[:, None] * outward
        # From the foot, the wall only needs to reach the lowest seen surface there.
        tops = np.where(tucked, np.minimum(tops, seen_height + TOP_SEARCH), tops)
    else:
        tuck = np.zeros(len(outline))

    # The flat bottom: the footprint, minus what lies outside the walls' feet.
    inside_cells = np.argwhere(occupancy >= 0.5) * GRID + low + GRID / 2
    tucks = np.nan_to_num(tuck)
    if len(outline) and (tucks > GRID).any():
        cut = np.zeros(len(inside_cells), dtype=bool)
        tree = cKDTree(inside_cells)
        moved = tucks > GRID
        for hits in tree.query_ball_point(outline[moved], tucks[moved]):
            cut[hits] = True
        inside_cells = inside_cells[~cut]
    bottom = to_3d(inside_cells, np.zeros(len(inside_cells)))
    bottom_normals = np.tile(-normal, (len(bottom), 1))

    wall, wall_normals = [], []
    for xy, direction, top in zip(feet, outward, tops):
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


def table_scraps(points: np.ndarray, table: TablePlane) -> np.ndarray:
    ''' Points near the table that lie outside the outline of the points above them.

    An object's outline seen from above contains all of its surface, even what is
    tucked in underneath, so a low point outside it is a scrap of table (from the
    object's shadow, or flared depth pixels) that fusion kept. Poisson would try to
    wrap a surface around it, and sometimes balloons instead.
    '''
    heights = table.height(points)
    if len(heights) == 0:
        return np.zeros(0, dtype=bool)
    low = heights < min(SCRAP_HEIGHT, SCRAP_SHARE * np.percentile(heights, 98))
    if low.all() or not low.any():
        return np.zeros(len(points), dtype=bool)
    _, u, v = _table_axes(table)
    flat = np.stack([points @ u, points @ v], axis=1)
    occupancy, origin = footprint(flat[~low], 0.0)
    inside = ndimage.binary_dilation(occupancy >= 0.5, iterations=max(1, round(SCRAP_MARGIN / GRID)))
    cells = np.floor((flat[low] - origin) / GRID).astype(int)
    valid = np.all((cells >= 0) & (cells < inside.shape), axis=1)
    kept = np.zeros(low.sum(), dtype=bool)
    kept[valid] = inside[cells[valid, 0], cells[valid, 1]]
    scraps = np.zeros(len(points), dtype=bool)
    scraps[np.flatnonzero(low)[~kept]] = True
    return scraps


def table_seen(views, poses: list[np.ndarray], table: TablePlane, center: np.ndarray,
               radius: float) -> np.ndarray:
    ''' Points where some view's depth landed on the table, within `radius` mm of `center`
    (across the table): what the cameras saw around and through the object. '''
    found = []
    for view, pose in zip(views, poses):
        k = view.intrinsics
        rows, cols = np.nonzero(view.depth)
        z = view.depth[rows, cols].astype(float)
        camera = np.stack([(cols - k.cx) * z / k.fx, (rows - k.cy) * z / k.fy, z], axis=1)
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        near = (np.abs(table.height(world)) < SEEN_TABLE_BAND) & (
            np.linalg.norm(world[:, :2] - center[:2], axis=1) < radius)
        found.append(world[near])
    return np.vstack(found) if found else np.zeros((0, 3))


def boundary_edge_count(mesh: o3d.geometry.TriangleMesh) -> int:
    triangles = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def enclosed_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    ''' Volume of a closed mesh (divergence theorem); tolerates tiny self-intersections. '''
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6)


def close_with_flat_bottom(cloud: o3d.geometry.PointCloud, table: TablePlane, depth: int = 9,
                           scale: float = 3.0, shrink: float = 1.0, base_band: float = 5.0,
                           priors: bool = True, seen_table: np.ndarray | None = None) -> o3d.geometry.TriangleMesh:
    ''' Closed mesh through the fused surface points, with a flat bottom on the table.

    `cloud` needs outward normals (the TSDF's point cloud has them). `scale` sizes
    Poisson's working box relative to the data; too tight a box leaves the surface
    open where it meets the box. The footprint comes from points above the base
    band, so it is the object's outline rather than the flared base. `seen_table`
    (from `table_seen`) keeps holes the cameras saw the table through open.
    '''
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    above = table.height(points) > 0
    if above.sum() < 50:  # too little surface to close
        return o3d.geometry.TriangleMesh()
    points, normals = points[above], normals[above]
    keep = ~table_scraps(points, table)
    points, normals = points[keep], normals[keep]
    # Where wall meets table, mixed depth pixels flare outward; the straight-wall
    # prior replaces the lowest `base_band` mm of observed surface.
    clear = table.height(points) > base_band
    if priors:
        extra, extra_normals = prior_points(points[clear], table, shrink, normals[clear], seen_table)
    else:  # plain screened Poisson through the measured surface alone
        extra, extra_normals = np.zeros((0, 3)), np.zeros((0, 3))
    points, normals = points[clear], normals[clear]
    samples = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([points, extra])))
    samples.normals = o3d.utility.Vector3dVector(np.vstack([normals, extra_normals]))
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(samples, depth=depth, scale=scale, linear_fit=True)
    # Poisson can add small detached bubbles; keep the object's pieces.
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[np.asarray(labels)] < KEEP_PIECE * counts.max())
    mesh.remove_unreferenced_vertices()
    # Flatten whatever dips below the table onto it.
    vertices = np.asarray(mesh.vertices)
    heights = table.height(vertices)
    below = heights < 0
    vertices[below] -= np.outer(heights[below], table.normal / np.linalg.norm(table.normal))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # Checked after flattening, so Poisson's usual bulge below the bottom doesn't count.
    fitted = np.asarray(samples.points)
    if len(vertices) == 0 or (np.any(vertices.min(axis=0) < fitted.min(axis=0) - BALLOON_MARGIN)
                              or np.any(vertices.max(axis=0) > fitted.max(axis=0) + BALLOON_MARGIN)):
        log.warning("  closing failed (the surface ballooned past the data); too little was seen to close it")
        return o3d.geometry.TriangleMesh()
    mesh.compute_vertex_normals()
    return mesh


def close_as_height_field(cloud: o3d.geometry.PointCloud, table: TablePlane, cell: float = 1.5,
                          seen_table: np.ndarray | None = None) -> o3d.geometry.TriangleMesh:
    ''' Closed mesh of a low object as solid from the table up to its measured top.

    For things lying flat (a coiled cable): the underside is never seen, so filling
    straight down is as good a guess as any, and unlike Poisson it can't balloon or
    bridge the gaps between strands. The top is the highest fused point in each
    `cell`-mm square; small gaps in the sampling are closed, while gaps where the
    table was seen (`seen_table`, from `table_seen`) stay open. Built directly as
    columns over the grid (top, walls, bottom), so it is closed by construction.
    '''
    points = np.asarray(cloud.points)
    heights = table.height(points)
    keep = heights > 0
    if keep.sum() < 50:
        return o3d.geometry.TriangleMesh()
    points, heights = points[keep], heights[keep]
    normal, u, v = _table_axes(table)
    flat = np.stack([points @ u, points @ v], axis=1)
    low = flat.min(axis=0) - 3 * cell
    at = ((flat - low) / cell).astype(int)
    shape = tuple(at.max(axis=0) + 4)
    top = np.full(shape, -np.inf)
    np.maximum.at(top, (at[:, 0], at[:, 1]), heights)
    occupied = np.isfinite(top)
    # Close sampling gaps (a cell or two), then drop specks of noise.
    filled = ndimage.binary_closing(occupied, iterations=1)
    if seen_table is not None and len(seen_table):
        seen = ((np.stack([seen_table @ u, seen_table @ v], axis=1) - low) / cell).astype(int)
        inside = np.all((seen >= 0) & (seen < shape), axis=1)
        table_cells = np.zeros(shape, dtype=bool)
        table_cells[seen[inside, 0], seen[inside, 1]] = True
        filled &= ~(table_cells & ~occupied)  # a gap the camera saw table through stays a gap
    filled = ndimage.binary_opening(filled)
    pieces, count = ndimage.label(filled)
    if count == 0:
        return o3d.geometry.TriangleMesh()
    sizes = ndimage.sum(filled, pieces, index=np.arange(1, count + 1))
    filled = np.isin(pieces, np.flatnonzero(sizes >= KEEP_PIECE * sizes.max()) + 1)
    # Heights: empty cells inside take their neighbors' highest, then a 3x3 median smooths.
    grid = np.where(occupied, top, 0.0)
    for _ in range(3):
        grid = np.where(filled & ~np.isfinite(top) & (grid == 0), ndimage.grey_dilation(grid, size=3), grid)
    grid = np.where(filled, ndimage.median_filter(grid, size=3), 0.0)
    grid = np.maximum(grid, cell)  # every column has some height

    # Vertices at grid corners, top and bottom; a corner's top is its highest filled cell.
    corners = np.zeros((shape[0] + 1, shape[1] + 1))
    for di in (0, 1):
        for dj in (0, 1):
            padded = np.zeros_like(corners)
            padded[di:di + shape[0], dj:dj + shape[1]] = np.where(filled, grid, 0.0)
            corners = np.maximum(corners, padded)
    index = -np.ones(corners.shape, dtype=int)
    used = np.zeros(corners.shape, dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            used[di:di + shape[0], dj:dj + shape[1]] |= filled
    index[used] = np.arange(used.sum())
    ci, cj = np.nonzero(used)
    xy = low + np.stack([ci, cj], axis=1) * cell
    on_plane = -table.offset * normal
    base = on_plane + xy[:, :1] * u + xy[:, 1:] * v
    n = len(base)
    vertices = np.vstack([base + np.outer(corners[ci, cj], normal), base])  # tops, then bottoms
    i, j = np.nonzero(filled)
    a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
    # u x v is the table normal, so (a, b, c) counterclockwise in (u, v) faces up.
    triangles = [np.stack([a, b, c], 1), np.stack([a, c, d], 1),            # top, facing up
                 np.stack([a + n, c + n, b + n], 1), np.stack([a + n, d + n, c + n], 1)]  # bottom, facing down
    # Walls along every edge between a filled cell and an empty one, facing out.
    padded = np.pad(filled, 1)
    for di, dj, p0, p1 in ((1, 0, (1, 0), (1, 1)), (-1, 0, (0, 1), (0, 0)), (0, 1, (1, 1), (0, 1)), (0, -1, (0, 0), (1, 0))):
        out = filled & ~padded[1 + di:1 + di + shape[0], 1 + dj:1 + dj + shape[1]]
        oi, oj = np.nonzero(out)
        s, e = index[oi + p0[0], oj + p0[1]], index[oi + p1[0], oj + p1[1]]
        triangles += [np.stack([s, s + n, e], 1), np.stack([e, s + n, e + n], 1)]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(np.vstack(triangles)))
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    return mesh
