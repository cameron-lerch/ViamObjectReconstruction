''' Closed mesh of a cable (or anything long, thin and lying on the table) as a tube
around its centerline.

A voxel fusion band about as wide as a cable is thick blurs neighboring strands
together, and filling down from the measured top leaves flat-topped ribbons. Methods
built for thin structures (CurveFusion, Vid2Curve, cable tracking in robotics) model
the curve instead: find the centerline, then put a tube of the right radius around it.

Here, from above (the cable rests on the table, so its path is a 2D curve with a height):

1. the top-down height map of the fused points (`bottom.top_down_heights`), keeping
   gaps where the table was seen through;
2. its centerline, by thinning the covered area to one cell wide (skeletonization),
   with short side branches (blur at the edges) pruned;
3. a radius: the cable's own, from the height of single strands lying on the table
   (or measured with calipers), and larger only where the object is clearly thicker
   (a plug: half its width there, at most half its height); smoothed along the line;
4. the centerline's height: the measured top minus the radius, never below the radius
   (resting on the table);
5. a closed surface where the distance to the nearest centerline point equals that
   point's radius (marching cubes on that distance field), cut at the table.

Where strands cross, the 2D centerline has one junction, so the strand underneath is
lifted over it instead of passing below: a known limit of working from above.
'''

import logging

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize

from .bottom import _table_axes, top_down_heights
from .scene import TablePlane

log = logging.getLogger(__name__)

PRUNE = 12.0  # side branches shorter than this (mm) are blur at the cable's edge or a plug's corners, not cable
SMOOTH = 5.0  # radius and height are medians over centerline points within this distance (mm)
NEAREST = 12  # centerline points considered for each voxel of the distance field


def _neighbors(skeleton: np.ndarray) -> np.ndarray:
    count = ndimage.convolve(skeleton.astype(int), np.ones((3, 3), dtype=int), mode="constant") - 1
    return np.where(skeleton, count, 0)


def prune(skeleton: np.ndarray, steps: int) -> np.ndarray:
    ''' Remove side branches up to `steps` cells long, keeping the real ends' length.

    Trimming every end `steps` times removes short branches entirely and shortens the
    long ones; growing back from the remaining ends (not from junctions) restores them.
    '''
    trimmed = skeleton.copy()
    for _ in range(steps):
        trimmed &= ~(_neighbors(trimmed) == 1)
    if not trimmed.any():
        return skeleton  # all of it was short: keep it as it was
    ends = trimmed & (_neighbors(trimmed) <= 1)
    grown = trimmed.copy()
    front = ends
    for _ in range(steps):
        front = ndimage.binary_dilation(front, structure=np.ones((3, 3))) & skeleton & ~grown
        if not front.any():
            break
        grown |= front
    return grown


def close_as_tube(cloud: o3d.geometry.PointCloud, table: TablePlane, cell: float = 1.0,
                  seen_table: np.ndarray | None = None, radius: float = 0.0) -> o3d.geometry.TriangleMesh:
    ''' Closed mesh of a cable lying on the table: a tube of varying radius around its
    centerline (see the module docstring). `radius` (mm) fixes the tube's radius, for a
    cable measured with calipers; 0 finds it from the data at each point. '''
    found = top_down_heights(cloud, table, cell, seen_table)
    if found is None:
        return o3d.geometry.TriangleMesh()
    top, covered, low = found
    skeleton = prune(skeletonize(covered), max(1, round(PRUNE / cell)))
    rows, cols = np.nonzero(skeleton)
    if len(rows) < 5:
        return o3d.geometry.TriangleMesh()
    width = ndimage.distance_transform_edt(covered) * cell  # distance to the edge, from above
    tops = top[rows, cols]
    # The cable's own radius: half the height of single strands lying on the table (the
    # lowest fifth of the centerline's tops and anything up to twice that; crossings and
    # plugs are taller), unless measured. Edge pixels are filtered and trimmed, so a
    # strand looks narrower from above than it is, and its height is the better measure.
    if radius <= 0:
        single = tops[tops <= 2 * np.percentile(tops, 20)]
        radius = 0.5 * float(np.median(single))
    # Thicker parts (a plug) keep their own size: half their width, at most half their height.
    radii = np.maximum(radius, np.minimum(width[rows, cols], tops / 2))
    centers = (np.column_stack([rows, cols]) + 0.5) * cell  # cell centers
    # Medians along the line, so one noisy cell doesn't pinch or bulge the tube.
    tree = cKDTree(centers)
    near = tree.query_ball_point(centers, SMOOTH)
    radii = np.array([np.median(radii[n]) for n in near])
    tops = np.array([np.median(tops[n]) for n in near])
    radii = np.maximum(radii, cell)
    heights = np.maximum(radii, tops - radii)
    axis = np.column_stack([centers, heights])
    log.info("  tube: %d centerline points, cable radius %.1fmm (thicker parts up to %.1fmm)",
             len(axis), radius, radii.max())

    # Distance field on a grid (table axes u, v, height), negative inside the tube.
    pad = radii.max() + 2 * cell
    lo = axis.min(axis=0) - pad
    lo[2] = -cell  # one layer below the table, so the bottom closes
    hi = axis.max(axis=0) + pad
    shape = np.ceil((hi - lo) / cell).astype(int) + 1
    grid = np.stack(np.meshgrid(*[lo[k] + cell * np.arange(shape[k]) for k in range(3)], indexing="ij"), axis=-1)
    voxels = grid.reshape(-1, 3)
    tree3 = cKDTree(axis)
    distance, index = tree3.query(voxels, k=min(NEAREST, len(axis)))
    field = (distance - radii[index]).min(axis=1)
    field = np.maximum(field, -voxels[:, 2])  # nothing below the table
    field = field.reshape(shape)
    field[[0, -1], :, :] = field[:, [0, -1], :] = field[:, :, [0, -1]] = cell  # closed at the box's edges
    vertices, faces, _, _ = marching_cubes(field, level=0.0, spacing=(cell, cell, cell))
    vertices += lo
    normal, u, v = _table_axes(table)
    on_plane = -table.offset * normal
    world = on_plane + (vertices[:, :1] + low[0]) * u + (vertices[:, 1:2] + low[1]) * v + vertices[:, 2:] * normal
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world), o3d.utility.Vector3iVector(faces))
    # Outward normals: flip the winding if the enclosed volume comes out negative.
    a, b, c = (world[faces[:, k]] for k in range(3))
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        mesh.triangles = o3d.utility.Vector3iVector(faces[:, ::-1])
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    return mesh
