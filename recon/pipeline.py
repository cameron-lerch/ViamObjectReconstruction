"""The scanning workflow: survey the table, plan each object's views, move it if
needed, capture stop-and-go views, and reconstruct."""

import asyncio
import json
import logging

import numpy as np

from .config import Config
from .planner import ScanPlan, plan_scan
from .reconstruct import reconstruct, save_reconstruction
from .relocate import PickAndPlace, choose_station, execute_pick_and_place, plan_pick_and_place
from .rgbd import back_project, load_view, load_views, save_view
from .robot import Robot
from .scene import DetectedObject, TablePlane, detect_objects, fit_table_plane
from .session import Session
from .transforms import look_at

log = logging.getLogger(__name__)


async def capture_view(robot: Robot, folder, extra: dict | None = None):
    color, color_ext, depth, T_world_cam, meta = await robot.capture()
    save_view(folder, color, depth, T_world_cam, await robot.intrinsics(), {**meta, **(extra or {})}, color_ext)
    return folder


async def move_camera_looking_down(robot: Robot, x: float, y: float, z: float, world_state=None) -> bool:
    reason = ""
    for roll in robot.config.views.rolls_deg:
        pose = look_at([x, y, z], [x, y, z - 100], roll)
        trajectory, reason = await robot.plan(robot.config.machine.camera, pose, world_state)
        if trajectory is not None:
            await robot.move(robot.config.machine.camera, pose, world_state)
            return True
    log.warning("Cannot reach overhead view at (%.0f, %.0f, %.0f): %s", x, y, z, reason)
    return False


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

    points = scene_points(folders, config)
    plane = fit_table_plane(points, config.workspace)
    objects = detect_objects(points, plane, config.workspace)
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
                      plane: TablePlane, config: Config, execute: bool, step: bool) -> None:
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
    for index, view in enumerate(plan.views):
        try:
            await robot.move(config.machine.camera, view.T_world_cam, world_state)
        except Exception as error:  # the planner can still fail on a path it approved from elsewhere
            log.warning("  view %d (%s) skipped: %s", index, view.candidate.describe(), error)
            continue
        await capture_view(robot, session.views_dir(obj.id) / f"view_{index:03d}", {
            "azimuth_deg": view.candidate.azimuth_deg, "elevation_deg": view.candidate.elevation_deg,
            "roll_deg": view.roll_deg, "commanded_T_world_cam": view.T_world_cam.tolist()})
        log.info("  captured view %d/%d (%s)", index + 1, len(plan.views), view.candidate.describe())

    if job is not None and config.relocate.return_after_scan:
        back = await plan_pick_and_place(robot, target, obj, others, config)
        if back is None:
            log.warning("  cannot plan the move back; object %d stays at the station", obj.id)
        else:
            await execute_pick_and_place(robot, back, others, config)

    reconstruct_object(session, obj.id, plane, config)


def reconstruct_object(session: Session, object_id: int, plane: TablePlane, config: Config, refine: bool = True):
    objects, _ = session.load_scene()
    scanned = session.object_dir(object_id) / "scanned_object.json"
    obj = (DetectedObject.from_dict(json.loads(scanned.read_text())) if scanned.exists()
           else next(o for o in objects if o.id == object_id))
    views = load_views(session.views_dir(object_id))
    log.info("Reconstructing object %d from %d views", object_id, len(views))
    result = reconstruct(views, obj, plane, config, refine)
    save_reconstruction(result, session.recon_dir(object_id))
    deviation = result.report["median_single_view_deviation_mm"]
    log.info("  mesh: %d triangles; single views deviate %.2fmm (median) from it -> %s",
             result.report["mesh"]["triangles"], deviation or float("nan"), session.recon_dir(object_id))
    return result
