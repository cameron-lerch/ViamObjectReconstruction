''' End to end on synthetic data: noisy views with a camera-mount error should
fuse into a mesh much closer to the true shape than any single view. '''

import numpy as np

from recon import synthetic
from recon.config import Config
from recon.reconstruct import object_mask, reconstruct, surface_distances
from recon.rgbd import back_project
from recon.scene import DetectedObject, TablePlane
from recon.views import generate_candidates, order_for_travel


def test_fused_mesh_beats_single_views():
    config = Config()
    config.views.azimuth_step_deg = 45
    scene = synthetic.SyntheticScene([synthetic.SceneObject(synthetic.cylinder(430, -170, 25, 90), (0.1, 0.6, 0.25))])
    truth = scene.objects[0].mesh
    obj = DetectedObject(0, np.array([431.0, -168.0, 37.0]), np.array([52.0, 51.0, 92.0]), 0.0, 1000)
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), -synthetic.TABLE_Z)
    poses = [c.pose() for c in order_for_travel(generate_candidates(obj, config.views))]
    views = synthetic.render_views(scene, poses, seed=3, frames=3)

    result = reconstruct(views, obj, plane, config)

    fused = surface_distances(truth, np.asarray(result.mesh.vertices))
    points, _, _ = back_project(views[3], 4, config.camera.min_depth, config.camera.max_depth)
    single = surface_distances(truth, points[object_mask(points, obj, plane, config.reconstruct)])
    assert np.median(fused) < 0.5
    assert np.median(fused) < 0.5 * np.median(single)
    assert not any(v["rejected"] for v in result.report["registration"]["views"])


def test_a_view_that_misses_the_object_is_left_out():
    # Guards the bowl scan of 2026-11-03: twice the arm ended up looking away from the object.
    # Fusion raised ("No block is touched") and alignment raised ("No normals") instead of
    # leaving those views out, so the scan's final mesh failed.
    from recon.reconstruct import integrate, new_grid, views_seeing
    config = Config()
    scene = synthetic.SyntheticScene([synthetic.SceneObject(synthetic.cylinder(430, -170, 25, 90), (0.1, 0.6, 0.25))])
    obj = DetectedObject(0, np.array([431.0, -168.0, 37.0]), np.array([52.0, 51.0, 92.0]), 0.0, 1000)
    elsewhere = DetectedObject(0, np.array([0.0, 400.0, 37.0]), np.array([52.0, 51.0, 92.0]), 0.0, 1000)
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), -synthetic.TABLE_Z)
    # The camera is aimed 700mm away from the object, so the object's box is out of its view.
    view = synthetic.render_views(scene, [generate_candidates(elsewhere, config.views)[0].pose()], seed=1)[0]
    grid = new_grid(config)
    integrate(grid, view, view.T_world_cam, obj, plane, config)  # must not raise
    assert grid.hashmap().size() == 0  # nothing was added
    assert views_seeing([view], obj, config) == []  # left out before alignment too
