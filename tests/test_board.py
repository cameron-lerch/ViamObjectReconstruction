"""ChArUco board: detection on a rendered board, and hand-eye recovery of a known camera mount."""

import cv2
import numpy as np

from recon.board import BoardSpec, BoardView, detect, hand_eye
from recon.rgbd import Intrinsics, View
from recon.transforms import look_at, make_transform, rotation_angle_deg
from scipy.spatial.transform import Rotation

SPEC = BoardSpec()
INTRINSICS = Intrinsics(1280, 720, 908.2, 908.2, 646.6, 373.8)


def test_detects_a_rendered_board_and_its_distance():
    image = SPEC.board().generateImage((1200, 900), marginSize=40)
    # A camera 400mm straight above the board: 15mm squares become 15 * f / 400 pixels.
    # generateImage fits the board to the image's tighter side.
    square_px = min((1200 - 80) / SPEC.squares_x, (900 - 80) / SPEC.squares_y)
    scale = INTRINSICS.fx / 400.0 * SPEC.square_mm / square_px
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((720, 1280), 255, np.uint8)
    y, x = (720 - small.shape[0]) // 2, (1280 - small.shape[1]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    view = View(np.dstack([canvas] * 3), np.zeros((720, 1280), np.uint16), np.eye(4), INTRINSICS)
    found = detect(view, SPEC)
    assert found is not None and len(found.ids) > 50
    assert found.reprojection_px < 0.5
    assert abs(found.T_cam_board[2, 3] - 400.0) < 4.0


def test_hand_eye_recovers_a_known_mount():
    rng = np.random.default_rng(0)
    true_mount = make_transform(Rotation.from_euler("xyz", [1.5, -2.0, 89.0], degrees=True).as_matrix(), [70.0, -29.5, 24.4])
    board_pose = make_transform(Rotation.from_euler("z", 20, degrees=True).as_matrix(), [300.0, -450.0, -8.0])
    center = (board_pose @ np.append(SPEC.center, 1.0))[:3]
    corners = SPEC.board().getChessboardCorners().astype(float)
    K = INTRINSICS.matrix
    ends, found = [], []
    for k in range(16):
        azimuth, elevation, distance = 360 * k / 16, 60 + 25 * (k % 3) / 2, 300 + 40 * (k % 4)
        a, e = np.radians(azimuth), np.radians(elevation)
        eye = center + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        camera = look_at(eye, center, roll_deg=rng.uniform(-60, 60))
        ends.append(camera @ np.linalg.inv(true_mount))
        T_cam_board = np.linalg.inv(camera) @ board_pose
        points = corners @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
        pixels = points[:, :2] / points[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2] + rng.normal(0, 0.3, (len(points), 2))
        ok, rvec, tvec = cv2.solvePnP(corners, pixels, K, None)
        found.append(BoardView(np.arange(len(corners)), corners, pixels, make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel()), 0.3))
    # Start from a mount that is 20mm / 3deg off, like the one configured on armfarm14.
    wrong = true_mount @ make_transform(Rotation.from_euler("xyz", [2, 1, 1], degrees=True).as_matrix(), [12, -10, 13])
    mount, _, diagnostics = hand_eye(ends, found, K, wrong)
    assert np.linalg.norm(mount[:3, 3] - true_mount[:3, 3]) < 1.0
    assert rotation_angle_deg(mount[:3, :3].T @ true_mount[:3, :3]) < 0.1
    assert diagnostics["reprojection_rms_px_after"] < 0.6
