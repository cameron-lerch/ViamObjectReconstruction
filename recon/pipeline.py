"""The scanning workflow: survey the table, plan each object's views, move it if
needed, capture stop-and-go views, and reconstruct."""

import asyncio
import json
import logging

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


async def capture_view(robot: Robot, folder, extra: dict | None = None):
    color, color_ext, depth, T_world_cam, meta = await robot.capture()
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


async def survey(robot: Robot, session: Session, config: Config, execute: bool) -> tuple[list[DetectedObject], TablePlane]:
    """Capture overhead views (from the survey positions, or from where the camera is
    now when motion is off), then find the table and the objects on it."""
    folders = []
    if execute:
        for index, (x, y, z) in enumerate(config.workspace.survey_positions):
            if await move_camera_looking_down(robot, x, y, z):
                folders.append(await capture_view(robot, session.survey_dir / f"view_{index:03d}"))
    else:
        log.info("Motion is off: surveying from the camera's current pose only")
        folders.append(await capture_view(robot, session.survey_dir / "view_000"))
    if not folders:
        raise RuntimeError("No survey views captured")

    # Detect in each view on its own: with camera calibration error, the views
    # disagree by several millimeters, and merging their clouds smears objects.
    detections, planes = [], []
    for folder in folders:
        points = scene_points([folder], config)
        view_plane = fit_table_plane(points, config.workspace)
        camera_xy = load_view(folder).camera_position[:2]
        detections.append((camera_xy, detect_objects(points, view_plane, config.workspace)))
        planes.append(view_plane)
    objects = merge_views(detections)
    # Keep the table as seen by the camera closest to the objects.
    middle = np.mean([obj.center[:2] for obj in objects], axis=0) if objects else detections[0][0]
    plane = planes[int(np.argmin([np.linalg.norm(camera_xy - middle) for camera_xy, _ in detections]))]
    session.save_scene(objects, plane)
    robot.set_measured_table(plane)
    session.update(intrinsics=(await robot.intrinsics()).__dict__, config=config.to_dict())
    log.info("Table at z=%.1fmm (tilt %.2fdeg); found %d object(s)", plane.z_at(*config.workspace.survey_positions[0][:2]),
             np.degrees(np.arccos(plane.normal[2])), len(objects))
    for obj in objects:
        log.info("  %s", obj.describe())
    return objects, plane


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


async def board_calibration(robot: Robot, session: Session, config: Config, execute: bool,
                            spec: BoardSpec | None = None) -> dict | None:
    """Hand-eye calibration and a depth check with a ChArUco board lying on the table.

    Finds the board from where the camera is now, plans views of it from several
    distances and directions (plan-only), and with `execute` captures them and
    solves for the camera mount and the depth sensor's error.
    """
    spec = spec or BoardSpec()
    located = await capture_view(robot, session.root / "board" / "located", {"frames": 3})
    view = load_view(located)
    found = detect(view, spec)
    if found is None:
        raise RuntimeError("The board isn't in view; move the camera so part of it is visible and try again")
    # The mat lies on the table, so it must look level. If it doesn't, the camera's
    # frame in the machine config is wrong, and every planned view would miss it.
    normal = (view.T_world_cam @ found.T_cam_board)[:3, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    if tilt > MAX_BOARD_TILT_DEG:
        raise RuntimeError(f"The board looks {tilt:.1f}deg from level through the configured camera frame; "
                           "fix the camera frame in the machine config (see scan.py calibrate) before calibrating")
    T_world_board = locate_flat(view, found, spec)
    center = (T_world_board @ np.append(spec.center, 1.0))[:3]
    log.info("Board found (%d corners, %.2fpx, %.1fdeg from level); its center is at (%.0f, %.0f, %.0f)",
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
        folder = await capture_view(robot, session.root / "board" / f"view_{index:03d}", {"frames": 3})
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
