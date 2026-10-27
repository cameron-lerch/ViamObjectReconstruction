"""Calibration with a ChArUco board (a checkerboard with ArUco markers in its white squares).

The board's printed geometry is known exactly, so from the color image alone
each view gives the camera's pose relative to the board, in millimeters. That
is independent of the depth sensor, which makes the board useful for two
checks the scan data can't do on its own:

- hand-eye calibration: the camera mount (camera pose on the arm's end) that
  makes the board stay put in the world across many arm poses;
- depth accuracy: what the depth sensor reads at each board corner versus where
  the image says the corner is, fit as an offset plus a scale.

Default board: calib.io 9 x 12, 15mm checkers, 11mm markers, ArUco DICT_5X5
(12 squares across, 9 down; marker ids up to 53, so the 100-marker dictionary).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .rgbd import View
from .transforms import make_transform


@dataclass
class BoardSpec:
    squares_x: int = 12
    squares_y: int = 9
    square_mm: float = 15.0
    marker_mm: float = 11.0
    dictionary: str = "DICT_5X5_100"

    def board(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_mm, self.marker_mm, dictionary)

    @property
    def center(self) -> np.ndarray:
        """Board center in the board's own frame (its origin is a corner)."""
        return np.array([self.squares_x * self.square_mm / 2, self.squares_y * self.square_mm / 2, 0.0])


@dataclass
class BoardView:
    ids: np.ndarray
    board_points: np.ndarray  # corners in the board frame (mm)
    pixels: np.ndarray  # where they were found in the image
    T_cam_board: np.ndarray
    reprojection_px: float


def detect(view: View, spec: BoardSpec, min_corners: int = 6) -> BoardView | None:
    """Find the board's chessboard corners and the board's pose relative to the camera."""
    board = spec.board()
    gray = cv2.cvtColor(view.color, cv2.COLOR_RGB2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    ids = ids.ravel()
    points = board.getChessboardCorners()[ids].astype(float)
    pixels = corners.reshape(-1, 2).astype(float)
    ok, rvec, tvec = cv2.solvePnP(points, pixels, view.intrinsics.matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(points, rvec, tvec, view.intrinsics.matrix, None)
    error = float(np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1).mean())
    return BoardView(ids, points, pixels, make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel()), error)


def depth_residuals(view: View, found: BoardView, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """For each corner: the distance the image implies, and depth reading minus it (mm)."""
    camera_points = found.board_points @ found.T_cam_board[:3, :3].T + found.T_cam_board[:3, 3]
    expected, residuals = [], []
    for (u, v), point in zip(found.pixels, camera_points):
        patch = view.depth[int(v) - window : int(v) + window + 1, int(u) - window : int(u) + window + 1].astype(float)
        patch = patch[patch > 0]
        if len(patch) >= window * window:
            expected.append(point[2])
            residuals.append(np.median(patch) - point[2])
    return np.array(expected), np.array(residuals)


def depth_model(views: list[View], found: list[BoardView]) -> dict:
    """Fit depth error = offset + scale x distance over every corner of every view."""
    pairs = [depth_residuals(v, f) for v, f in zip(views, found)]
    expected = np.concatenate([p[0] for p in pairs])
    residuals = np.concatenate([p[1] for p in pairs])
    if len(expected) < 10:
        return {}
    slope, intercept = np.polyfit(expected, residuals, 1)
    fitted = intercept + slope * expected
    return {"corners": int(len(expected)), "median_error_mm": float(np.median(residuals)),
            "offset_mm": float(intercept), "scale_percent": float(slope * 100),
            "fit_rms_mm": float(np.sqrt(np.mean((residuals - fitted) ** 2))),
            "distance_range_mm": [float(expected.min()), float(expected.max())]}


def locate_flat(view: View, found: BoardView, spec: BoardSpec) -> np.ndarray:
    """The board's world pose, assuming it lies flat: heading from the image, height
    and position from the depth at the detected corners.

    A partial view pins the board's tilt down poorly, and extending a slightly
    tilted fit to the far side of the board misplaces it by centimeters.
    """
    R = (view.T_world_cam @ found.T_cam_board)[:3, :3]
    up = np.array([0.0, 0.0, 1.0]) * np.sign(R[2, 2] or 1.0)
    x_axis = R[:, 0] - (R[:, 0] @ up) * up
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(up, x_axis)
    rotation = np.column_stack([x_axis, y_axis, up])
    rays = (found.pixels - view.intrinsics.matrix[:2, 2]) / [view.intrinsics.fx, view.intrinsics.fy]
    origins = []
    for (u, v), ray, corner in zip(found.pixels, rays, found.board_points):
        patch = view.depth[int(v) - 3 : int(v) + 4, int(u) - 3 : int(u) + 4].astype(float)
        patch = patch[patch > 0]
        if len(patch) < 9:
            continue
        world = view.T_world_cam[:3, :3] @ (np.append(ray, 1.0) * np.median(patch)) + view.T_world_cam[:3, 3]
        origins.append(world - rotation @ corner)
    if not origins:
        return view.T_world_cam @ found.T_cam_board
    return make_transform(rotation, np.median(origins, axis=0))


def _transform(params: np.ndarray) -> np.ndarray:
    return make_transform(Rotation.from_rotvec(params[:3]).as_matrix(), params[3:])


def _params(transform: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]])


def hand_eye(arm_ends: list[np.ndarray], found: list[BoardView], intrinsics: np.ndarray,
             initial_mount: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Camera mount and board pose from views of a still board at many arm poses.

    `arm_ends` are the arm's end poses in the world. Starting from
    `initial_mount` (the configured or scan-calibrated mount, within a few cm and
    degrees), it minimizes the corners' reprojection error over the mount (6
    numbers) and the board's world pose (6) together. Returns (mount,
    T_world_board, diagnostics).
    """
    mount = initial_mount
    boards = [end @ mount @ f.T_cam_board for end, f in zip(arm_ends, found)]
    T_world_board = boards[0].copy()
    T_world_board[:3, 3] = np.mean([b[:3, 3] for b in boards], axis=0)

    def residuals(x: np.ndarray) -> np.ndarray:
        mount_x, board_x = _transform(x[:6]), _transform(x[6:])
        out = []
        for end, f in zip(arm_ends, found):
            T_cam_board = np.linalg.inv(end @ mount_x) @ board_x
            camera_points = f.board_points @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
            pixels = camera_points[:, :2] / camera_points[:, 2:] * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2]
            out.append((pixels - f.pixels).ravel())
        return np.concatenate(out)

    start = np.concatenate([_params(mount), _params(T_world_board)])
    before = float(np.sqrt(np.mean(residuals(start) ** 2)))
    result = least_squares(residuals, start, loss="huber", f_scale=1.0)
    after = float(np.sqrt(np.mean(residuals(result.x) ** 2)))
    return _transform(result.x[:6]), _transform(result.x[6:]), {
        "views": len(found), "corners": int(sum(len(f.ids) for f in found)),
        "reprojection_rms_px_before": before, "reprojection_rms_px_after": after}
