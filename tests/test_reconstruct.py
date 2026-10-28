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
