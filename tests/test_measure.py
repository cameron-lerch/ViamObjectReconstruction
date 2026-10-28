"""Matching a mesh to the caliper ground truth by shape and size."""

import open3d as o3d

from recon.measure import identify

TRUTH = {
    "green_cylinder": {"shape": "cylinder", "diameter": 29.53, "height": 59.93},
    "yellow_block": {"shape": "box", "width": 29.26, "length": 29.48, "height": 60.05},
    "red_block": {"shape": "box", "width": 30.15, "length": 29.18, "height": 29.44},
}


def test_a_block_matches_the_block_of_its_size():
    mesh = o3d.geometry.TriangleMesh.create_box(31.0, 32.0, 60.5)  # a bit fat, like a real scan
    assert identify(mesh, TRUTH)[0] == "yellow_block"
    assert identify(o3d.geometry.TriangleMesh.create_box(31.0, 32.0, 30.0), TRUTH)[0] == "red_block"


def test_a_cylinder_matches_the_cylinder_not_a_block_of_the_same_size():
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=16.0, height=60.0, resolution=60)
    assert identify(mesh, TRUTH)[0] == "green_cylinder"


def test_something_unlike_any_entry_is_skipped():
    pumpkin = o3d.geometry.TriangleMesh.create_sphere(radius=110.0, resolution=40)
    assert identify(pumpkin, TRUTH)[0] is None
