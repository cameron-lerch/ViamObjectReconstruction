''' Flat-bottom closing and the 1..n view progression, on synthetic data. '''

import numpy as np
import open3d as o3d
import pytest

from recon import synthetic
from recon.bottom import boundary_edge_count, close_with_flat_bottom, enclosed_volume
from recon.config import Config
from recon.progression import progression, spread_order
from recon.scene import DetectedObject, TablePlane
from recon.views import generate_candidates, order_for_travel

TABLE = TablePlane(np.array([0.0, 0.0, 1.0]), 0.0)


def surface_cloud(mesh: o3d.geometry.TriangleMesh, keep) -> o3d.geometry.PointCloud:
    cloud = mesh.sample_points_uniformly(40000, use_triangle_normal=True)
    points = np.asarray(cloud.points)
    return cloud.select_by_index(np.flatnonzero(keep(points)))


def test_open_cylinder_is_closed_flat_on_the_table():
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(15, 60, 80, 6).translate((0, 0, 30))
    # What fusion sees: the walls from 3mm up and the top, never the bottom.
    cloud = surface_cloud(cylinder, lambda p: p[:, 2] > 3)
    mesh = close_with_flat_bottom(cloud, TABLE, shrink=0.0)
    assert boundary_edge_count(mesh) == 0
    assert np.asarray(mesh.vertices)[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert enclosed_volume(mesh) == pytest.approx(np.pi * 15**2 * 60, rel=0.05)


def test_unseen_wall_is_filled_straight_down_from_the_outline():
    box = o3d.geometry.TriangleMesh.create_box(40, 30, 50).translate((-20, -15, 0))
    # The +x wall and the bottom were never seen.
    cloud = surface_cloud(box, lambda p: (p[:, 2] > 3) & (p[:, 0] < 19.5))
    mesh = close_with_flat_bottom(cloud, TABLE, shrink=0.0)
    assert boundary_edge_count(mesh) == 0
    assert enclosed_volume(mesh) == pytest.approx(40 * 30 * 50, rel=0.08)
    assert np.asarray(mesh.vertices)[:, 0].max() == pytest.approx(20, abs=2.5)


def test_wall_tucks_in_under_a_rounded_object():
    # A ball resting on the table, seen down to 15mm above it: its surface curves back
    # in below the widest point, so the closed mesh must not hang a wall from there.
    ball = o3d.geometry.TriangleMesh.create_sphere(40, 60).translate((0, 0, 40))
    ball.compute_vertex_normals()
    cloud = surface_cloud(ball, lambda p: p[:, 2] > 15)
    mesh = close_with_flat_bottom(cloud, TABLE, shrink=0.0)
    vertices = np.asarray(mesh.vertices)
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    assert boundary_edge_count(mesh) == 0
    # At 15mm up the ball is 31mm across in radius; the old straight wall was 40mm.
    assert radius[(vertices[:, 2] > 5) & (vertices[:, 2] < 12)].max() < 36
    assert radius[vertices[:, 2] > 35].max() == pytest.approx(40, abs=2)


def test_progression_improves_with_more_views(tmp_path):
    config = Config()
    config.views.azimuth_step_deg = 60
    scene = synthetic.SyntheticScene([synthetic.SceneObject(synthetic.cylinder(430, -170, 25, 90), (0.1, 0.6, 0.25))])
    obj = DetectedObject(0, np.array([431.0, -168.0, 37.0]), np.array([52.0, 51.0, 92.0]), 0.0, 1000)
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), -synthetic.TABLE_Z)
    views = synthetic.render_views(scene, [c.pose() for c in order_for_travel(generate_candidates(obj, config.views))], seed=4)

    order = spread_order(views, obj.center)
    assert sorted(order) == list(range(len(views)))
    assert views[order[0]].camera_position[2] == max(v.camera_position[2] for v in views)

    rows = progression(views, obj, plane, config, tmp_path, counts=[1, 3, len(views)], truth=scene.objects[0].mesh)
    raw, one, three, everything = rows
    assert raw["views"] == "raw" and everything["views"] == len(views)
    # 1 and 3 views see slivers that Poisson can't always close (it balloons, and the
    # closing reports nothing); when they do close, all views must beat them.
    assert one["observed_triangles"] > 0 and three["observed_triangles"] > 0
    for few in (one, three):
        if few.get("closed_truth_median_mm") is not None:
            assert everything["closed_truth_median_mm"] < few["closed_truth_median_mm"]
        assert everything["measured_share"] > few.get("measured_share", 0)
    assert everything["roughness_mm"] < 0.5 * raw["roughness_mm"]
    assert everything["completeness"] == pytest.approx(1.0, abs=0.02) and three["completeness"] < 0.7
    # The bottom (~11% of this cylinder's surface) is never seen, so ~89% is the ceiling.
    assert everything["measured_share"] > 0.8
    assert everything["volume_cm3"] == pytest.approx(np.pi * 2.5**2 * 9.0, rel=0.03)
    for k in (1, 3, len(views)):
        assert (tmp_path / f"views_{k:02d}" / "mesh.obj").exists()
        assert (tmp_path / f"views_{k:02d}" / "observed.obj").exists()
    assert (tmp_path / "views_01" / "raw_points.ply").exists()
    assert (tmp_path / "summary.csv").read_text().startswith("views,")
