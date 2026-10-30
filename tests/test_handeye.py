''' Hand-eye methods recover a known camera mount from simulated board views. '''

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recon.board import BoardSpec, BoardView
from recon.handeye import METHODS, board_pose, consistency, difference, leave_one_out, solve
from recon.transforms import look_at, make_transform

K = np.array([[917.0, 0.0, 640.0], [0.0, 916.0, 360.0], [0.0, 0.0, 1.0]])
MOUNT = make_transform(Rotation.from_euler("xyz", [1.0, -2.0, 88.0], degrees=True).as_matrix(), [68.0, -35.0, 24.0])
BOARD = make_transform(np.eye(3), [450.0, -350.0, 0.0])


def simulated_views(count=12, noise_px=0.2, seed=1):
    rng = np.random.default_rng(seed)
    corners = BoardSpec().board().getChessboardCorners().astype(float)
    center = BOARD[:3, 3] + [90.0, 67.5, 0.0]
    ends, found = [], []
    for k in range(count):
        azimuth, elevation = np.radians(360 * k / count), np.radians(rng.uniform(60, 88))
        eye = center + rng.uniform(300, 450) * np.array([np.cos(elevation) * np.cos(azimuth),
                                                         np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        camera = look_at(eye, center, rng.uniform(-60, 60))
        ends.append(camera @ np.linalg.inv(MOUNT))
        T_cam_board = np.linalg.inv(camera) @ BOARD
        points = corners @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
        pixels = points[:, :2] / points[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2] + rng.normal(0, noise_px, (len(points), 2))
        # The board's pose from its own pixels (solvePnP's job), as detection would give it.
        import cv2
        _, rvec, tvec = cv2.solvePnP(corners, pixels, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        pnp = make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel())
        found.append(BoardView(np.arange(len(corners)), corners, pixels, pnp, noise_px))
    return ends, found


START = MOUNT @ make_transform(Rotation.from_euler("z", 2.0, degrees=True).as_matrix(), [5.0, -3.0, 2.0])


@pytest.mark.parametrize("method", METHODS)
def test_every_method_recovers_the_mount_exactly_without_noise(method):
    ends, found = simulated_views(noise_px=0.0)
    mm, deg = difference(MOUNT, solve(method, ends, found, K, START))
    assert mm < 0.01 and deg < 0.01


def test_with_pixel_noise_fitting_the_pixels_is_most_accurate():
    # 0.2px of corner noise (a good real detection): ours 0.03mm, the closed-form
    # methods 0.3-1.8mm (Tsai, what Viam's module uses, 0.5mm).
    ends, found = simulated_views(noise_px=0.2)
    errors = {m: difference(MOUNT, solve(m, ends, found, K, START))[0] for m in METHODS}
    assert errors["ours"] < 0.1
    assert all(errors["ours"] < errors[m] < 2.5 for m in METHODS if m != "ours")


def test_scores_are_small_for_the_true_mount():
    ends, found = simulated_views()
    spread, turn = consistency(ends, found, MOUNT)
    assert spread < 1.0 and turn < 0.2
    assert np.allclose(board_pose(ends, found, K, MOUNT)[:3, 3], BOARD[:3, 3], atol=1.0)
    assert np.median(leave_one_out("ours", ends, found, K, MOUNT)) < 1.0
