''' Mount calibration should recover a known camera-mount error from a synthetic scan. '''

import numpy as np

from recon import synthetic
from recon.calibrate import apply_mount, calibrate_mount
from recon.config import Config
from recon.scene import DetectedObject
from recon.transforms import rotation_angle_deg
from recon.views import generate_candidates, order_for_travel


def test_recovers_camera_mount_error():
    config = Config()
    config.views.azimuth_step_deg = 45
    scene = synthetic.SyntheticScene([synthetic.SceneObject(synthetic.box(430, -170, (60, 40, 50), 25), (0.8, 0.4, 0.1))])
    obj = DetectedObject(0, np.array([430.0, -170.0, 17.0]), np.array([60.0, 40.0, 50.0]), 25.0, 1000)
    error = synthetic.hand_eye_error(translation_mm=12.0, rotation_deg=2.0)
    poses = [c.pose() for c in order_for_travel(generate_candidates(obj, config.views))]
    views = synthetic.render_views(scene, poses, seed=5, error=error)

    # Views report poses for an identity mount; the true mount is `error`.
    calibration = calibrate_mount(views, np.eye(4), obj, config)

    residual = np.linalg.inv(error) @ calibration.mount
    assert np.linalg.norm(residual[:3, 3]) < 1.5
    assert rotation_angle_deg(residual[:3, :3]) < 0.3
    assert calibration.rms_after_mm < calibration.rms_before_mm
    fixed = apply_mount(views, np.eye(4), calibration.mount)
    np.testing.assert_allclose(fixed[0].T_world_cam, poses[0] @ calibration.mount)
