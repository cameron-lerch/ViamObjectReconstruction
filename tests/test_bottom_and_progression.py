"""Flat-bottom closing and the 1..n view progression, on synthetic data."""

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
    # The best single-view mesh: everything one view saw, closed with the priors.
    assert one["observed_triangles"] > 0 and one["volume_cm3"] is not None
    assert everything["closed_truth_median_mm"] < one["closed_truth_median_mm"]
    assert one["measured_share"] < three["measured_share"] < everything["measured_share"]
    assert everything["roughness_mm"] < 0.5 * raw["roughness_mm"]
    assert everything["closed_truth_median_mm"] < three["closed_truth_median_mm"]
    assert everything["completeness"] == pytest.approx(1.0, abs=0.02) and three["completeness"] < 0.7
    # The bottom (~11% of this cylinder's surface) is never seen, so ~89% is the ceiling.
    assert everything["measured_share"] > 0.8 > three["measured_share"]
    assert everything["volume_cm3"] == pytest.approx(np.pi * 2.5**2 * 9.0, rel=0.03)
    for k in (1, 3, len(views)):
        assert (tmp_path / f"views_{k:02d}" / "mesh.obj").exists()
        assert (tmp_path / f"views_{k:02d}" / "observed.obj").exists()
    assert (tmp_path / "views_01" / "raw_points.ply").exists()
    assert (tmp_path / "summary.csv").read_text().startswith("views,")
