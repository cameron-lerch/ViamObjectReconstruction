''' Objects lying flat on the table (a coiled cable, 2026-11-11): the closing must not treat
them as table scraps, and the height-field closing must give a closed mesh that keeps
gaps open. Guards the cable scan whose watertight mesh kept only 34 x 128mm of the coil. '''

import numpy as np
import open3d as o3d

from recon.bottom import boundary_edge_count, close_as_height_field, enclosed_volume, table_scraps
from recon.scene import TablePlane

TABLE = TablePlane(np.array([0.0, 0.0, 1.0]), 0.0)


def strip(x0, x1, y0, y1, height, spacing=0.5):
    g = np.mgrid[x0:x1:spacing, y0:y1:spacing].reshape(2, -1).T
    return np.column_stack([g, np.full(len(g), height)])


def test_a_low_strand_is_not_a_table_scrap():
    # Strands 8mm tall, crossing to 20mm in one spot: before, every strand point under
    # 15mm outside the crossing's outline counted as table.
    points = np.vstack([strip(0, 150, 0, 8, 8.0), strip(70, 80, 0, 8, 20.0)])
    assert table_scraps(points, TABLE).mean() < 0.05


def test_height_field_is_closed_and_keeps_a_gap():
    ring = np.vstack([strip(0, 30, 0, 4, 8.0), strip(0, 30, 11, 15, 8.0),
                      strip(0, 4, 0, 15, 8.0), strip(26, 30, 0, 15, 8.0)])
    mesh = close_as_height_field(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ring)), TABLE)
    assert boundary_edge_count(mesh) == 0 and mesh.is_edge_manifold()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() > 0  # normals face out
    # The 22 x 7mm gap in the middle stays open: well under the full 30 x 15 x 8mm block.
    assert enclosed_volume(mesh) < 0.8 * 30 * 15 * 8


def test_tube_follows_a_cable_with_its_radius():
    # A cable 8mm thick lying in a U on the table, sampled on its top half: the tube
    # closing gives one closed, round tube of about 4mm radius along it.
    rng = np.random.default_rng(0)
    s = np.linspace(0, 1, 4000)
    path = np.where(s[:, None] < 0.5, np.column_stack([s * 200, np.zeros_like(s)]),
                    np.column_stack([100 + 40 * np.sin((s - 0.5) * 2 * np.pi), 40 - 40 * np.cos((s - 0.5) * 2 * np.pi)]))
    angle = rng.uniform(-np.pi / 2, np.pi / 2, len(s))
    side = np.column_stack([-np.gradient(path[:, 1]), np.gradient(path[:, 0])])
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    points = np.column_stack([path + side * 4.0 * np.sin(angle)[:, None], 4.0 + 4.0 * np.cos(angle)])
    from recon.tube import close_as_tube
    mesh = close_as_tube(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)), TABLE)
    assert boundary_edge_count(mesh) == 0
    labels, counts, _ = mesh.cluster_connected_triangles()
    assert len(counts) == 1  # one piece
    top = np.asarray(mesh.vertices)[:, 2].max()
    assert 6.5 < top < 10.0  # about the cable's 8mm, not a slab or a sliver
