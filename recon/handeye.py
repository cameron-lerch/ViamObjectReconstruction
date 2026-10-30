''' Hand-eye calibration methods compared on the same saved board views.

"ours" is recon/board.py's solver: least squares on the board corners' pixels, over the
camera mount and the board's pose together, starting from the mount in use. The others
are OpenCV's closed-form solvers of AX = XB (cv2.calibrateHandEye), run through Viam's own
camera-calibration solver (its python/solve.py, unchanged; the module uses "tsai") on the
same inputs it would get: arm end poses, and the board's pose in each camera image from
solvePnP. Viam's solver needs OpenCV 4 (OpenCV 5, which our board detection uses, has no
calibrateHandEye), so it runs in a separate environment, .venv-opencv4:

    python3 -m venv .venv-opencv4
    .venv-opencv4/bin/pip install opencv-contrib-python-headless==4.10.0.84 numpy

Each method's mount is scored by:
- leave-one-out reprojection: solve from all views but one, then predict where the left-
  out view sees the board's corners (pixels). It tests prediction, not fit, so a method
  can't grade itself;
- Viam's consistency residuals: how far the board's position (mm) and rotation (deg),
  computed through the mount from each view, wander across the views;
- how far the mount is from ours (mm, deg).
'''

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .board import BoardSpec, BoardView, detect, hand_eye
from .rgbd import load_view
from .transforms import make_transform

ROOT = Path(__file__).resolve().parent.parent
# Viam's solver, from the camera-calibration repository cloned beside this one.
VIAM_SOLVE = ROOT.parent / "ViamSourceCode" / "camera-calibration" / "python" / "solve.py"
OPENCV4_PYTHON = ROOT / ".venv-opencv4" / "bin" / "python"
OPENCV_METHODS = ["tsai", "park", "horaud", "andreff", "daniilidis"]
METHODS = ["ours", *OPENCV_METHODS]


def load_board_views(folder: Path, spec: BoardSpec | None = None):
    ''' Arm end poses, board detections, intrinsics and the mount in use, from a saved
    board run (data/board-*/board/view_*). Views where the board wasn't found are skipped. '''
    spec = spec or BoardSpec()
    ends, found, mounts, K = [], [], [], None
    for record in sorted(Path(folder).glob("view_*/view.json")):
        view = load_view(record.parent)
        detection = detect(view, spec)
        if detection is None:
            continue
        mount = np.array(view.meta["camera_mount"])
        ends.append(view.T_world_cam @ np.linalg.inv(mount))
        found.append(detection)
        mounts.append(mount)
        K = view.intrinsics.matrix
    if len(found) < 5:
        raise ValueError(f"{folder}: only {len(found)} views see the board")
    return ends, found, K, mounts[0]


def _pose(T: np.ndarray) -> dict:
    ''' A pose in Viam solve.py's format: translation (mm) and rotation vector (Rodrigues). '''
    return {"translation": T[:3, 3].tolist(), "rvec": Rotation.from_matrix(T[:3, :3]).as_rotvec().tolist()}


def viam_solve(problems: list[tuple[str, list[np.ndarray], list[BoardView]]]) -> list[np.ndarray]:
    ''' Mounts from Viam's solver for (method, arm end poses, board views) problems, in one batch. '''
    if not OPENCV4_PYTHON.exists() or not VIAM_SOLVE.exists():
        raise RuntimeError(f"Viam's solver needs {VIAM_SOLVE} and the OpenCV 4 environment {OPENCV4_PYTHON.parent.parent} "
                           "(see recon/handeye.py)")
    batch = [{"method": method,
              "stations": [{"T_be": _pose(end), "T_cw": _pose(f.T_cam_board), "reprojection_error_px": f.reprojection_px}
                           for end, f in zip(ends, found)]}
             for method, ends, found in problems]
    with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
        json.dump(batch, file)
        file.flush()
        output = subprocess.run([str(OPENCV4_PYTHON), str(ROOT / "tools" / "viam_solve_batch.py"), str(VIAM_SOLVE), file.name],
                                capture_output=True, text=True, check=True).stdout
    mounts = []
    for result in json.loads(output):
        mount = result["camera_in_parent_mm"]
        mounts.append(make_transform(Rotation.from_rotvec(mount["rvec"]).as_matrix(), np.array(mount["translation"])))
    return mounts


def solve(method: str, ends: list[np.ndarray], found: list[BoardView], K: np.ndarray,
          initial: np.ndarray) -> np.ndarray:
    ''' The camera mount (camera pose on the arm's end) by one method. '''
    if method == "ours":
        mount, _, _ = hand_eye(ends, found, K, initial)
        return mount
    return viam_solve([(method, ends, found)])[0]


def _project(T_cam_board: np.ndarray, found: BoardView, K: np.ndarray) -> np.ndarray:
    camera = found.board_points @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
    return camera[:, :2] / camera[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]


def board_pose(ends, found, K, mount) -> np.ndarray:
    ''' The board's world pose that best explains the views' corners, for a fixed mount. '''
    guesses = [end @ mount @ f.T_cam_board for end, f in zip(ends, found)]
    start = np.concatenate([Rotation.from_matrix(guesses[0][:3, :3]).as_rotvec(),
                            np.mean([g[:3, 3] for g in guesses], axis=0)])

    def residuals(x):
        board = make_transform(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:])
        return np.concatenate([(_project(np.linalg.inv(end @ mount) @ board, f, K) - f.pixels).ravel()
                               for end, f in zip(ends, found)])

    x = least_squares(residuals, start, loss="huber", f_scale=1.0).x
    return make_transform(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:])


def reprojection_px(ends, found, K, mount, board) -> np.ndarray:
    ''' RMS pixel error of each view's corners, predicted through the mount and board pose. '''
    return np.array([np.sqrt(np.mean(np.sum((_project(np.linalg.inv(end @ mount) @ board, f, K) - f.pixels) ** 2, axis=1)))
                     for end, f in zip(ends, found)])


def consistency(ends, found, mount) -> tuple[float, float]:
    ''' Viam's residuals: the board's position through each view, mean distance from their
    mean (mm); its rotation, mean angle from the first view's (deg). '''
    boards = [end @ mount @ f.T_cam_board for end, f in zip(ends, found)]
    positions = np.array([b[:3, 3] for b in boards])
    spread = float(np.mean(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
    angles = [np.degrees(np.linalg.norm(Rotation.from_matrix(boards[0][:3, :3].T @ b[:3, :3]).as_rotvec()))
              for b in boards[1:]]
    return spread, float(np.mean(angles))


def _folds(n: int) -> list[list[int]]:
    return [[i for i in range(n) if i != k] for k in range(n)]


def leave_one_out(method, ends, found, K, initial, mounts: list[np.ndarray] | None = None) -> np.ndarray:
    ''' Each view's RMS pixel error when it was left out of the solve (`mounts`: the
    mounts already solved without each view, if any). '''
    errors = []
    for k, others in enumerate(_folds(len(found))):
        e, f = [ends[i] for i in others], [found[i] for i in others]
        mount = mounts[k] if mounts is not None else solve(method, e, f, K, initial)
        board = board_pose(e, f, K, mount)
        errors.append(reprojection_px([ends[k]], [found[k]], K, mount, board)[0])
    return np.array(errors)


def difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ''' How far apart two mounts are: (mm, deg). '''
    change = np.linalg.inv(a) @ b
    return float(np.linalg.norm(change[:3, 3])), float(np.degrees(np.linalg.norm(Rotation.from_matrix(change[:3, :3]).as_rotvec())))


def compare_methods(folder: Path, methods: list[str] | None = None) -> tuple[list[dict], dict[str, np.ndarray]]:
    ''' Every method on one board run: a row of scores each, and the mounts. '''
    ends, found, K, initial = load_board_views(folder)
    methods = methods or METHODS
    folds = _folds(len(found))
    # Viam's methods: the full solve and every leave-one-out solve, in one batch.
    viam = [m for m in methods if m != "ours"]
    problems = [(m, [ends[i] for i in fold], [found[i] for i in fold]) for m in viam for fold in [list(range(len(found))), *folds]]
    solved = iter(viam_solve(problems)) if problems else iter(())
    mounts, held_out_mounts = {}, {}
    for method in methods:
        if method == "ours":
            mounts[method] = solve(method, ends, found, K, initial)
            held_out_mounts[method] = None
        else:
            mounts[method] = next(solved)
            held_out_mounts[method] = [next(solved) for _ in folds]
    reference = mounts.get("ours")
    rows = []
    for method, mount in mounts.items():
        board = board_pose(ends, found, K, mount)
        fit = reprojection_px(ends, found, K, mount, board)
        held_out = leave_one_out(method, ends, found, K, initial, held_out_mounts[method])
        spread, turn = consistency(ends, found, mount)
        row = {"method": method, "views": len(found), "fit_px": float(np.sqrt(np.mean(fit ** 2))),
               "held_out_px_median": float(np.median(held_out)), "held_out_px_max": float(held_out.max()),
               "board_spread_mm": spread, "board_turn_deg": turn,
               "mount_x": float(mount[0, 3]), "mount_y": float(mount[1, 3]), "mount_z": float(mount[2, 3])}
        if reference is not None:
            row["from_ours_mm"], row["from_ours_deg"] = difference(reference, mount)
        rows.append(row)
    return rows, mounts
