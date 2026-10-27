"""The scanning workflow: survey the table, plan each object's views, move it if
needed, capture stop-and-go views, and reconstruct."""

import asyncio
import json
import logging
import time

from pathlib import Path

import numpy as np

from .board import BoardSpec, depth_model, detect, hand_eye, locate_flat
from .calibrate import MountCalibration, apply_mount, calibrate_mount
from .config import Config
from .planner import ScanPlan, plan_scan
from .reconstruct import reconstruct, save_reconstruction
from .relocate import PickAndPlace, choose_station, execute_pick_and_place, plan_pick_and_place
from .rgbd import back_project, load_view, load_views, save_view
from .robot import Robot
from .scene import DetectedObject, TablePlane, detect_objects, fit_table_plane, merge_views
from .session import Session
from .transforms import look_at

log = logging.getLogger(__name__)

CALIBRATION_FILE = "mount_calibration.json"
MAX_BOARD_TILT_DEG = 3.0


async def capture_view(robot: Robot, folder, extra: dict | None = None, frames: int | None = None, depth: bool = True):
    color, color_ext, depth, T_world_cam, meta = await robot.capture(frames, depth)
    save_view(folder, color, depth, T_world_cam, await robot.intrinsics(), {**meta, **(extra or {})}, color_ext)
    return folder


async def move_camera_looking_down(robot: Robot, x: float, y: float, z: float, world_state=None) -> bool:
    camera = robot.config.machine.camera
    roll, pose, reason = await robot.plan_best_roll(camera, lambda r: look_at([x, y, z], [x, y, z - 100], r), world_state)
    if pose is None:
        log.warning("Cannot reach overhead view at (%.0f, %.0f, %.0f): %s", x, y, z, reason)
        return False
    await robot.move(camera, pose, world_state)
    return True


def scene_points(folders, config: Config) -> np.ndarray:
    clouds = [back_project(load_view(f), 2, config.camera.min_depth, config.camera.max_depth)[0] for f in folders]
    return np.concatenate(clouds)


def survey_grid(config: Config) -> dict[tuple[int, int], np.ndarray]:
    """Overhead camera positions over the search region, keyed by (column, row), skipping
    the arm's base. With `[workspace] survey_positions` set, those are used instead."""
    workspace = config.workspace
    if workspace.survey_positions:
        return {(k, 0): np.array(p, dtype=float) for k, p in enumerate(workspace.survey_positions)}
    cells = {}
    (x0, x1), (y0, y1) = workspace.x_range, workspace.y_range
    nx = max(1, int(np.ceil((x1 - x0) / workspace.survey_spacing)))
    ny = max(1, int(np.ceil((y1 - y0) / workspace.survey_spacing)))
    for i in range(nx):
        for j in range(ny):
            x, y = x0 + (i + 0.5) * (x1 - x0) / nx, y0 + (j + 0.5) * (y1 - y0) / ny
            if np.hypot(x, y) >= workspace.min_base_distance + 50:
                cells[(i, j)] = np.array([x, y, workspace.survey_height])
    return cells


def table_continues(points: np.ndarray, plane: TablePlane, here: np.ndarray, toward: np.ndarray,
                    min_share: float = 0.3) -> bool:
    """Whether the part of this view facing a neighboring position still shows table.

    Looks at the strip of points beyond a quarter of the way to the neighbor:
    floor, walls or nothing at all (out of depth range) don't count as table.
    """
    direction = (toward - here)[:2]
    distance = np.linalg.norm(direction)
    along = (points[:, :2] - here[:2]) @ (direction / distance)
    strip = points[along > 0.25 * distance]
    if len(strip) < 500:
        return False
    on_table = np.abs(plane.height(strip)) < 10
    return bool(on_table.mean() >= min_share)


async def survey_table(robot: Robot, session: Session, config: Config) -> list:
    """Overhead views that follow the table: start over it, and only go on to a
    neighboring grid position if this view's side facing it still shows table."""
    cells = survey_grid(config)
    reachable = {}
    for key, (x, y, z) in cells.items():
        _, pose, reason = await robot.plan_best_roll(config.machine.camera, lambda r, x=x, y=y, z=z: look_at([x, y, z], [x, y, z - 100], r))
        if pose is not None:
            reachable[key] = cells[key]
        else:
            log.info("  survey position (%.0f, %.0f) unreachable: %s", x, y, reason)
    if not reachable:
        raise RuntimeError("No survey position is reachable")
    here = (await robot.checked_camera_pose())[:3, 3]
    frontier = {min(reachable, key=lambda k: np.linalg.norm(reachable[k][:2] - here[:2]))}
    visited, folders, plane = set(), [], None
    while frontier:
        key = min(frontier, key=lambda k: np.linalg.norm(reachable[k][:2] - here[:2]))
        frontier.discard(key)
        visited.add(key)
        x, y, z = reachable[key]
        if not await move_camera_looking_down(robot, x, y, z):
            continue
        folder = await capture_view(robot, session.survey_dir / f"view_{len(folders):03d}",
                                    {"frames": config.capture.survey_frames, "grid": list(key)}, config.capture.survey_frames)
        folders.append(folder)
        here = reachable[key]
        points = scene_points([folder], config)
        if plane is None:
            try:
                plane = fit_table_plane(points, config.workspace)
            except RuntimeError:
                raise RuntimeError("The first survey view shows no table; start with the camera over the table")
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (key[0] + di, key[1] + dj)
            if neighbor in reachable and neighbor not in visited and neighbor not in frontier:
                if table_continues(points, plane, reachable[key], reachable[neighbor]):
                    frontier.add(neighbor)
                else:
                    visited.add(neighbor)  # off the table (or nothing to see) in that direction
                    log.info("  not surveying (%.0f, %.0f): no table in that direction", *reachable[neighbor][:2])
    log.info("Surveyed %d of %d grid positions (the rest are off the table or unreachable)", len(folders), len(cells))
    return folders


async def survey(robot: Robot, session: Session, config: Config, execute: bool) -> tuple[list[DetectedObject], TablePlane]:
    """Look straight down across the table (following it from where the camera is,
    or, with motion off, only from where the camera is now), then find the table, the
    objects on it and the calibration board. With motion on, finish with a head-on
    view of the board."""
    if execute:
        folders = await survey_table(robot, session, config)
    else:
        log.info("Motion is off: surveying from the camera's current pose only")
        folders = [await capture_view(robot, session.survey_dir / "view_000", frames=config.capture.survey_frames)]
    if not folders:
        raise RuntimeError("No survey views captured")

    # Detect in each view on its own: with camera calibration error, the views
    # disagree by several millimeters, and merging their clouds smears objects.
    detections, planes = [], []
    for folder in folders:
        view = load_view(folder)
        points, colors, _ = back_project(view, 2, config.camera.min_depth, config.camera.max_depth)
        view_plane = fit_table_plane(points, config.workspace)
        detections.append((view.camera_position[:2], detect_objects(points, view_plane, config.workspace, colors)))
        planes.append(view_plane)
    objects = merge_views(detections)
    # Keep the table as seen by the camera closest to the objects.
    middle = np.mean([obj.center[:2] for obj in objects], axis=0) if objects else detections[0][0]
    plane = planes[int(np.argmin([np.linalg.norm(camera_xy - middle) for camera_xy, _ in detections]))]
    session.save_scene(objects, plane)
    robot.set_measured_table(plane)
    session.update(intrinsics=(await robot.intrinsics()).__dict__, config=config.to_dict())
    log.info("Table at z=%.1fmm (tilt %.2fdeg); found %d object(s)", plane.z_at(*middle),
             np.degrees(np.arccos(plane.normal[2])), len(objects))
    for obj in objects:
        log.info("  %s", obj.describe())

    board = find_board(folders, BoardSpec())
    if board is None:
        log.info("No calibration board in the survey views")
    else:
        T_world_board, found, tilt = board
        center = (T_world_board @ np.append(BoardSpec().center, 1.0))[:3]
        log.info("Calibration board: center (%.0f, %.0f, %.0f), %d corners seen, %.1fdeg from level",
                 *center, len(found.ids), tilt)
        record = {"pose": T_world_board.tolist(), "center": center.tolist(), "survey_tilt_deg": tilt}
        if tilt > MAX_BOARD_TILT_DEG:
            log.warning("  the board should look level: the camera mount (%s) is probably wrong",
                        config.machine.camera_mount_file or "machine config")
        elif execute:
            record |= await view_board_head_on(robot, session, config, center)
        session.update(board=record)
    return objects, plane


def find_board(folders, spec: BoardSpec):
    """The best view of the board among `folders`: (board pose lying flat, detection, tilt)."""
    best = None
    for folder in folders:
        view = load_view(folder)
        found = detect(view, spec)
        if found is not None and (best is None or len(found.ids) > len(best[1].ids)):
            best = (view, found)
    if best is None:
        return None
    view, found = best
    normal = (view.T_world_cam @ found.T_cam_board)[:3, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    return locate_flat(view, found, spec), found, tilt


async def view_board_head_on(robot: Robot, session: Session, config: Config, center: np.ndarray) -> dict:
    """Look straight down at the board's center so all of it is in view, and check it."""
    spec = BoardSpec()
    if not await move_camera_looking_down(robot, center[0], center[1], center[2] + config.board.view_height):
        return {}
    folder = await capture_view(robot, session.root / "board" / "head_on")
    view = load_view(folder)
    found = detect(view, spec)
    if found is None:
        log.warning("  board not found in the head-on view")
        return {}
    normal = (view.T_world_cam @ found.T_cam_board)[:3, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    depth = depth_model([view], [found])
    log.info("  head-on view: %d of %d corners, %.2fpx, %.1fdeg from level; depth vs. board: median %+.2fmm",
             len(found.ids), (spec.squares_x - 1) * (spec.squares_y - 1), found.reprojection_px, tilt,
             depth.get("median_error_mm", float("nan")))
    T_world_board = locate_flat(view, found, spec)
    return {"pose": T_world_board.tolist(), "center": (T_world_board @ np.append(spec.center, 1.0))[:3].tolist(),
            "head_on": {"corners": int(len(found.ids)), "reprojection_px": found.reprojection_px,
                        "tilt_deg": tilt, "depth": depth}}


async def locate_near(robot: Robot, x: float, y: float, plane: TablePlane, session: Session, object_id: int,
                      config: Config) -> DetectedObject:
    """Look straight down at (x, y) and return the object found closest to it."""
    height = plane.z_at(x, y) + config.views.standoff + 200
    if not await move_camera_looking_down(robot, x, y, height):
        raise RuntimeError("Cannot look at the scan station")
    folder = await capture_view(robot, session.object_dir(object_id) / "located")
    found = detect_objects(scene_points([folder], config), plane, config.workspace)
    if not found:
        raise RuntimeError(f"No object found near ({x:.0f}, {y:.0f})")
    nearest = min(found, key=lambda o: np.hypot(o.center[0] - x, o.center[1] - y))
    nearest.id = object_id
    log.info("  relocated %s", nearest.describe())
    return nearest


async def confirm(prompt: str) -> bool:
    answer = await asyncio.to_thread(input, f"{prompt} [Enter = yes, n = skip] ")
    return answer.strip().lower() not in ("n", "no", "q")


async def plan_object(robot: Robot, obj: DetectedObject, others: list[DetectedObject],
                      config: Config) -> tuple[ScanPlan, PickAndPlace | None, ScanPlan | None]:
    """Plan views in place; if coverage is too low, plan a move to a better station."""
    plan = await plan_scan(robot, obj, others, config)
    log.info("Object %d in place: %s", obj.id, plan.summary())
    if plan.coverage >= config.views.min_coverage or not config.relocate.enabled:
        return plan, None, None
    log.info("  coverage below %.0f%%; looking for a scan station", config.views.min_coverage * 100)
    station, station_plan = await choose_station(robot, obj, others, config)
    if station is None or station_plan.coverage <= plan.coverage + 0.05:
        log.info("  no station improves coverage; scanning in place")
        return plan, None, None
    job = await plan_pick_and_place(robot, obj, station, others, config)
    if job is None:
        log.info("  no reachable grasp (object %.0fmm wide, gripper max %.0fmm); scanning in place",
                 obj.size[1], config.relocate.max_grasp_width)
        return plan, None, None
    log.info("  will move it to (%.0f, %.0f): %s", station.center[0], station.center[1], station_plan.summary())
    return plan, job, station_plan


def save_plan(session: Session, obj: DetectedObject, plan: ScanPlan, job: PickAndPlace | None,
              station_plan: ScanPlan | None) -> None:
    session.write_json(session.object_dir(obj.id) / "plan.json", {
        "in_place": plan.to_dict(),
        "relocation": None if job is None else {"station": job.destination.center.tolist(),
                                                "plan": station_plan.to_dict()},
    })


async def scan_object(robot: Robot, session: Session, obj: DetectedObject, others: list[DetectedObject],
                      plane: TablePlane, config: Config, execute: bool, step: bool,
                      max_views: int | None = None) -> None:
    plan, job, station_plan = await plan_object(robot, obj, others, config)
    save_plan(session, obj, plan, job, station_plan)
    if not execute:
        return
    if step and not await confirm(f"Scan object {obj.id}?"):
        return

    target = obj
    if job is not None:
        if step and not await confirm(f"Move object {obj.id} to the scan station?"):
            job = None
        else:
            await execute_pick_and_place(robot, job, others, config)
            target = await locate_near(robot, *job.destination.center[:2], plane, session, obj.id, config)
            plan = await plan_scan(robot, target, others, config)
            log.info("  at station: %s", plan.summary())
    session.write_json(session.object_dir(obj.id) / "scanned_object.json", target.to_dict())

    world_state = robot.world_state([target, *others], config.views.object_margin)
    # Views run top first, then ring by ring downward, so a limited run keeps to the high views.
    for index, view in enumerate(plan.views[:max_views]):
        try:
            await robot.move(config.machine.camera, view.T_world_cam, world_state)
        except Exception as error:  # the planner can still fail on a path it approved from elsewhere
            log.warning("  view %d (%s) skipped: %s", index, view.candidate.describe(), error)
            continue
        await capture_view(robot, session.views_dir(obj.id) / f"view_{index:03d}", {
            "azimuth_deg": view.candidate.azimuth_deg, "elevation_deg": view.candidate.elevation_deg,
            "roll_deg": view.roll_deg, "commanded_T_world_cam": view.T_world_cam.tolist()})
        log.info("  captured view %d/%d (%s)", index + 1, len(plan.views[:max_views]), view.candidate.describe())

    if job is not None and config.relocate.return_after_scan:
        back = await plan_pick_and_place(robot, target, obj, others, config)
        if back is None:
            log.warning("  cannot plan the move back; object %d stays at the station", obj.id)
        else:
            await execute_pick_and_place(robot, back, others, config)

    reconstruct_object(session, obj.id, plane, config)


def scanned_object(session: Session, object_id: int) -> DetectedObject:
    """The object as it was when scanned (it may have been moved to a station)."""
    objects, _ = session.load_scene()
    scanned = session.object_dir(object_id) / "scanned_object.json"
    return (DetectedObject.from_dict(json.loads(scanned.read_text())) if scanned.exists()
            else next(o for o in objects if o.id == object_id))


def calibrate_session(session: Session, object_id: int, config: Config) -> MountCalibration:
    """Estimate the camera mount from one object's views and save it with the session."""
    views = load_views(session.views_dir(object_id))
    mounts = {json.dumps(v.meta["camera_mount"]) for v in views if "camera_mount" in v.meta}
    if len(mounts) > 1:
        raise ValueError("Views were captured with different camera mounts")
    if not mounts:
        raise ValueError("Views do not record the camera mount they were captured with")
    original = np.array(json.loads(mounts.pop()))
    calibration = calibrate_mount(views, original, scanned_object(session, object_id), config)
    session.write_json(session.root / CALIBRATION_FILE, {
        "from_object": object_id, "original": original.tolist(), "calibrated": calibration.mount.tolist(),
        "rms_before_mm": calibration.rms_before_mm, "rms_after_mm": calibration.rms_after_mm})
    return calibration


def apply_session_calibration(session: Session, views: list) -> list:
    """Recompute camera poses with the session's calibrated mount, when there is one."""
    path = session.root / CALIBRATION_FILE
    if not path.exists():
        return views
    calibration = json.loads(path.read_text())
    original, calibrated = np.array(calibration["original"]), np.array(calibration["calibrated"])
    for view in views:
        if "camera_mount" in view.meta and not np.allclose(view.meta["camera_mount"], original):
            raise ValueError(f"{view.name} was captured with a different mount than the calibration assumes")
    log.info("  using the calibrated camera mount from %s", path.name)
    return apply_mount(views, original, calibrated)


def reconstruct_object(session: Session, object_id: int, plane: TablePlane, config: Config, refine: bool = True):
    obj = scanned_object(session, object_id)
    views = apply_session_calibration(session, load_views(session.views_dir(object_id)))
    log.info("Reconstructing object %d from %d views", object_id, len(views))
    result = reconstruct(views, obj, plane, config, refine)
    save_reconstruction(result, session.recon_dir(object_id))
    deviation = result.report["median_single_view_deviation_mm"]
    log.info("  mesh: %d triangles; single views deviate %.2fmm (median) from it -> %s",
             result.report["mesh"]["triangles"], deviation or float("nan"), session.recon_dir(object_id))
    return result


async def search_for_board(robot: Robot, session: Session, config: Config, spec: BoardSpec,
                           search: bool) -> np.ndarray | None:
    """Where the board is, roughly, from color-only views (about 0.15s each): first from
    where the camera is, then (with `search`) from the survey grid, nearest first."""
    async def look(name: str):
        folder = await capture_view(robot, session.root / "board" / name, frames=1, depth=False)
        view = load_view(folder)
        found = detect(view, spec)
        if found is None:
            return None
        center = (locate_flat(view, found, spec) @ np.append(spec.center, 1.0))[:3]
        log.info("Board seen (%d corners) from %s; roughly centered at (%.0f, %.0f, %.0f)", len(found.ids), name, *center)
        return center

    center = await look("search_000")
    if center is not None or not search:
        return center
    here = (await robot.checked_camera_pose())[:3, 3]
    for index, (x, y, z) in enumerate(sorted(survey_grid(config).values(), key=lambda p: np.linalg.norm(p[:2] - here[:2])), 1):
        if await move_camera_looking_down(robot, x, y, z):
            center = await look(f"search_{index:03d}")
            if center is not None:
                return center
    return None


async def board_calibration(robot: Robot, session: Session, config: Config, execute: bool,
                            spec: BoardSpec | None = None, known_center: np.ndarray | None = None) -> dict | None:
    """Hand-eye calibration and a depth check with a ChArUco board lying on the table.

    Locates the board (from the survey's `known_center`, or from where the camera is
    now), plans views of it from several distances and directions (plan-only), and
    with `execute` captures them and solves for the camera mount and the depth
    sensor's error.
    """
    spec = spec or BoardSpec()
    if known_center is None:
        known_center = await search_for_board(robot, session, config, spec, execute)
        if known_center is None:
            raise RuntimeError("The board isn't in view; add --execute to search for it (color-only, quick), or run a survey")
    if not execute:
        center = known_center
        log.info("Planning around the board centered at (%.0f, %.0f, %.0f)", *center)
    else:
        # Head-on, with the whole board in view, to pin down where it is.
        await move_camera_looking_down(robot, known_center[0], known_center[1], known_center[2] + config.board.view_height)
        located = await capture_view(robot, session.root / "board" / "located", {"frames": 3}, 3)
        view = load_view(located)
        found = detect(view, spec)
        if found is None:
            raise RuntimeError("The board isn't in the head-on view")
        # The mat lies on the table, so it must look level. If it doesn't, the camera
        # mount we're using is wrong, and every planned view would miss the board.
        normal = (view.T_world_cam @ found.T_cam_board)[:3, 2]
        tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
        if tilt > MAX_BOARD_TILT_DEG:
            raise RuntimeError(f"The board looks {tilt:.1f}deg from level through the camera mount in use; "
                               "fix the camera mount (scan.py calibrate --save-mount) before calibrating")
        T_world_board = locate_flat(view, found, spec)
        center = (T_world_board @ np.append(spec.center, 1.0))[:3]
        log.info("Board head-on: %d corners, %.2fpx, %.1fdeg from level; center (%.0f, %.0f, %.0f)",
                 len(found.ids), found.reprojection_px, tilt, *center)

    camera = config.machine.camera
    poses = []
    for distance in (300.0, 380.0, 460.0):
        for elevation in (62.0, 75.0, 88.0):
            for azimuth in range(0, 360, 60):
                a, e = np.radians(azimuth + elevation), np.radians(elevation)  # stagger azimuths per ring
                eye = center + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                poses.append(lambda roll, eye=eye: look_at(eye, center, roll))
    reachable = []
    for pose_for_roll in poses:
        _, pose, _ = await robot.plan_best_roll(camera, pose_for_roll)
        if pose is not None:
            reachable.append(pose)
    # Spread the chosen views out: farthest-first over camera position and viewing direction.
    chosen = [reachable[0]] if reachable else []
    while reachable and len(chosen) < min(config.board.views, len(reachable)):
        score = [min(np.linalg.norm(p[:3, 3] - c[:3, 3]) / 100 + (1 - p[:3, 2] @ c[:3, 2]) * 5 for c in chosen)
                 for p in reachable]
        chosen.append(reachable[int(np.argmax(score))])
    log.info("%d of %d board views are reachable; using %d", len(reachable), len(poses), len(chosen))
    if not execute:
        log.info("Planning only. Add --execute to capture the board views.")
        return None

    mount = await robot.camera_mount()
    views, found_views, ends = [], [], []
    for index, pose in enumerate(chosen):
        await robot.move(camera, pose)
        folder = await capture_view(robot, session.root / "board" / f"view_{index:03d}",
                                    {"frames": config.board.frames}, config.board.frames)
        board_view = load_view(folder)
        found = detect(board_view, spec)
        if found is None:
            log.warning("  view %d: board not found; skipped", index)
            continue
        views.append(board_view)
        found_views.append(found)
        ends.append(board_view.T_world_cam @ np.linalg.inv(mount))
        log.info("  view %d/%d: %d corners, %.2fpx", index + 1, len(chosen), len(found.ids), found.reprojection_px)
    if len(found_views) < 5:
        raise RuntimeError(f"Only {len(found_views)} usable board views")
    calibrated, T_world_board, diagnostics = hand_eye(ends, found_views, views[0].intrinsics.matrix, mount)
    result = {"original_mount": mount.tolist(), "board_mount": calibrated.tolist(), "board_pose": T_world_board.tolist(),
              "hand_eye": diagnostics, "depth": depth_model(views, found_views)}
    session.write_json(session.root / "board" / "board_calibration.json", result)
    return result


def save_mount(config: Config, mount: np.ndarray, source: str) -> Path:
    """Make `mount` the camera mount used for every move and capture."""
    if not config.machine.camera_mount_file:
        raise ValueError("Set [machine] camera_mount_file in config.toml first")
    path = config.resolve(config.machine.camera_mount_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mount": mount.tolist(), "source": source, "saved": time.strftime("%Y-%m-%d %H:%M"),
                                "note": "Camera pose relative to the arm's end (mm). Used for every move and capture."},
                               indent=2) + "\n")
    return path
