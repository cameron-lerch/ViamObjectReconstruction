''' The scanning workflow: survey the table, plan each object's views, move it if
needed, capture stop-and-go views, and reconstruct. '''

import asyncio
import json
import logging
import time
from dataclasses import replace

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .board import BoardSpec, depth_model, detect, hand_eye, locate_flat
from .calibrate import MountCalibration, apply_mount, calibrate_mount
from .config import Config
from .figures import doubling_counts, facing_the_arm, progression_counts, progression_figure, progression_video
from .planner import ScanPlan, plan_scan, spread_views
from .progression import progression
from .reconstruct import local_table_plane, reconstruct, save_reconstruction, views_seeing
from .relocate import PickAndPlace, choose_station, execute_pick_and_place, plan_pick_and_place
from .rgbd import back_project, load_view, load_views, save_frames, save_view
from .robot import MissedTarget, MoveDeclined, Robot, RobotUnresponsive, ask_operator
from .scene import (DetectedObject, TablePlane, clutter_around, detect_objects, fit_table_plane, join_connected,
                    merge_views, views_agree)
from .session import Session
from .survey_map import survey_map
from .transforms import look_at, rotation_angle_deg

log = logging.getLogger(__name__)

CALIBRATION_FILE = "mount_calibration.json"
MAX_BOARD_TILT_DEG = 3.0


async def capture_view(robot: Robot, folder, extra: dict | None = None, frames: int | None = None, depth: bool = True,
                       keep_frames: bool = False):
    ''' Capture and save a view; with `keep_frames`, every raw frame too, in folder/frames/
    (color_NNN.jpg and depth_NNN.png, 16-bit millimeters), beside the median depth.png. '''
    color, color_ext, depth, T_world_cam, meta = await robot.capture(frames, depth, keep=keep_frames)
    save_view(folder, color, depth, T_world_cam, await robot.intrinsics(), {**meta, **(extra or {})}, color_ext)
    if keep_frames and robot.last_frames:
        await asyncio.to_thread(save_frames, Path(folder) / "frames", *robot.last_frames)  # ~1.7s for 50
    if robot.live:
        robot.live.captured(folder, robot.table_plane)
    return folder


async def move_camera_looking_down(robot: Robot, x: float, y: float, z: float, world_state=None) -> bool:
    camera = robot.config.machine.camera
    roll, pose, reason = await robot.plan_best_roll(camera, lambda r: look_at([x, y, z], [x, y, z - 100], r), world_state)
    if pose is None:
        log.warning("Cannot reach overhead view at (%.0f, %.0f, %.0f): %s", x, y, z, reason)
        return False
    try:
        await robot.move(camera, pose, world_state)
    except (MoveDeclined, RobotUnresponsive):  # Stop, or the robot stopped answering: end the run
        raise
    except Exception as error:  # missed the target, or the move's own plan failed: skip this position
        log.warning("Overhead view at (%.0f, %.0f, %.0f) skipped: %s", x, y, z, why_skipped(error))
        return False
    return True


def scene_points(folders, config: Config) -> np.ndarray:
    clouds = [back_project(load_view(f), 2, config.camera.min_depth, config.camera.max_depth)[0] for f in folders]
    return np.concatenate(clouds)


async def go_home(robot: Robot, session: Session, config: Config) -> None:
    ''' Park the camera straight above the calibration mat at survey height (a pose
    like every survey view), so each run ends in the same, known-safe place. '''
    board = session.read().get("board")
    if not board:
        log.info("No calibration mat in this session's survey; leaving the arm where it is")
        return
    x, y, _ = board["center"]
    log.info("Returning home: camera above the calibration mat at (%.0f, %.0f)", x, y)
    await move_camera_looking_down(robot, x, y, config.workspace.survey_height, robot.world_state([], 0.0))


def table_continues(points: np.ndarray, plane: TablePlane, here: np.ndarray, toward: np.ndarray,
                    min_share: float = 0.3) -> bool:
    ''' Whether the part of this view facing a neighboring position still shows table.

    Looks at the strip of points beyond a quarter of the way to the neighbor:
    floor, walls or nothing at all (out of depth range) don't count as table. The
    table must also be seen at least halfway to the neighbor: past a table's edge
    there may be no points at all, so a share of points alone can't tell.
    '''
    direction = (toward - here)[:2]
    distance = np.linalg.norm(direction)
    along = (points[:, :2] - here[:2]) @ (direction / distance)
    strip = points[along > 0.25 * distance]
    # Thresholds from README section 5.1 (judgment, checked on armfarm14): at least 500
    # points to judge by, and table means within 10mm of the first view's plane.
    if len(strip) < 500:
        return False
    on_table = np.abs(plane.height(strip)) < 10
    if not on_table.any() or np.percentile(along[along > 0.25 * distance][on_table], 99) < 0.5 * distance:
        return False
    return bool(on_table.mean() >= min_share)


class NeedsTable(RuntimeError):
    ''' The survey can't start: the camera isn't over the table. Jog the arm there and run again. '''


MIN_LOOK_DOWN = 0.5  # the camera must point at least 30deg below horizontal to start a survey
TABLE_BAND_MM = 10.0  # points this close to a view's table plane count as table, for the table area


def survey_lattice_neighbors(key: tuple[int, int]) -> list[tuple[int, int]]:
    return [(key[0] + di, key[1] + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]


async def survey_table(robot: Robot, session: Session, config: Config) -> list:
    ''' Overhead views that find the table from where the camera is, then follow it outward.

    1. Look before moving: the first picture is taken from wherever the camera is (it must
       point down at the table; otherwise NeedsTable asks to jog the arm over it), and the
       table is found in it. It's saved as survey/first_look, apart from the survey views.
    2. Small moves next: the camera only turns to look straight down and rises (or lowers)
       to `survey_height` above that table, over the same spot.
    3. Then outward on a lattice `survey_spacing` apart, starting there: the camera goes on
       to a neighboring position only if its last picture still shows table on that side, so
       it stops at the table's edges; positions the planner can't reach (out of reach, or
       blocked by Viam's obstacles, like walls) are skipped. At most `survey_max_views`.
    With `[workspace] survey_positions` set, the camera looks down from those instead.
    '''
    workspace = config.workspace
    folders = []

    async def look(extra: dict) -> Path:
        folder = await capture_view(robot, session.survey_dir / f"view_{len(folders):03d}",
                                    {"frames": config.capture.survey_frames, **extra}, config.capture.survey_frames)
        folders.append(folder)
        return folder

    if workspace.survey_positions:
        for k, (x, y, z) in enumerate(workspace.survey_positions):
            if await move_camera_looking_down(robot, x, y, z):
                await look({"position": k})
        return folders

    start = await robot.checked_camera_pose()
    if start[2, 2] > -MIN_LOOK_DOWN:
        raise NeedsTable("The camera isn't looking down at the table (it points %.0fdeg below horizontal). Jog the arm "
                         "so the camera is over the table, looking down, and run again" % np.degrees(np.arcsin(max(-1.0, -start[2, 2]))))
    # Only to find the table: objects are found in the straight-down views that follow, as
    # an angled view from close up fits the table less reliably.
    first = await capture_view(robot, session.survey_dir / "first_look", {"frames": config.capture.survey_frames},
                               config.capture.survey_frames)
    try:
        plane = fit_table_plane(scene_points([first], config), workspace)
    except RuntimeError as error:
        raise NeedsTable("The camera doesn't see the table from where it is. Jog the arm so the camera is over the "
                         "table, looking down, and run again") from error

    x, y, z = start[:3, 3]
    anchor, step, height = np.array([x, y]), workspace.survey_spacing, workspace.survey_height

    def position(key: tuple[int, int]) -> np.ndarray:
        xy = anchor + step * np.array(key, dtype=float)
        return np.array([*xy, plane.z_at(*xy) + height])

    log.info("Table found from where the camera is; looking straight down from %.0fmm above it, then following "
             "the table outward in %.0fmm steps", height, step)
    if not await move_camera_looking_down(robot, *position((0, 0))):
        # Survey height out of reach here: survey from the camera's own height instead.
        height = z - plane.z_at(x, y)
        log.info("  survey height unreachable here; surveying from %.0fmm above the table instead", height)
        if not await move_camera_looking_down(robot, x, y, z):
            raise NeedsTable("The arm can't look straight down from where the camera is. Jog it over the table and run again")
    here = (0, 0)
    points = scene_points([await look({"grid": [0, 0]})], config)
    visited, frontier = {here}, set()

    def extend(key, points):
        for neighbor in survey_lattice_neighbors(key):
            if neighbor in visited or neighbor in frontier:
                continue
            p = position(neighbor)
            if np.hypot(*p[:2]) < workspace.base_clearance:
                visited.add(neighbor)  # right over the arm's base
            elif table_continues(points, plane, position(key), p):
                frontier.add(neighbor)
            else:
                visited.add(neighbor)  # off the table (or nothing to see) in that direction
                log.info("  not surveying (%.0f, %.0f): no table in that direction", *p[:2])

    extend(here, points)
    while frontier and len(folders) < workspace.survey_max_views:
        key = min(frontier, key=lambda k: np.linalg.norm(position(k)[:2] - position(here)[:2]))
        frontier.discard(key)
        visited.add(key)
        if not await move_camera_looking_down(robot, *position(key)):
            continue
        here = key
        points = scene_points([await look({"grid": list(key)})], config)
        extend(key, points)
    if frontier:
        log.warning("  stopped after %d survey views (survey_max_views); %d more positions showed table", len(folders), len(frontier))
    log.info("Surveyed the table from %d positions (it ends, or the arm can't reach, in every other direction)", len(folders))
    return folders


def table_area(clouds: list[tuple[np.ndarray, np.ndarray]]) -> list[list[float]] | None:
    ''' The table area the survey saw, [[x0, x1], [y0, y1]]: where its views' points lie on
    their table plane (0.5th to 99.5th percentile, so a few stray points don't stretch it). '''
    table = [points[np.abs(heights) < TABLE_BAND_MM, :2] for points, heights in clouds]
    table = np.concatenate([t for t in table if len(t)]) if any(len(t) for t in table) else np.empty((0, 2))
    if len(table) < 1000:
        return None
    low, high = np.percentile(table, 0.5, axis=0), np.percentile(table, 99.5, axis=0)
    return [[round(float(low[0]), 1), round(float(high[0]), 1)], [round(float(low[1]), 1), round(float(high[1]), 1)]]


def use_table_area(config: Config, region: list[list[float]] | None) -> None:
    ''' Search the table area the survey saw, unless config.toml sets one. '''
    workspace = config.workspace
    if region and not (workspace.x_range and workspace.y_range):
        workspace.x_range, workspace.y_range = list(region[0]), list(region[1])


async def survey(robot: Robot, session: Session, config: Config, execute: bool) -> tuple[list[DetectedObject], TablePlane]:
    ''' Look straight down across the table (following it from where the camera is,
    or, with motion off, only from where the camera is now), then find the table, the
    objects on it and the calibration board. With motion on, finish with a head-on
    view of the board. '''
    if execute:
        folders = await survey_table(robot, session, config)
    else:
        log.info("Motion is off: surveying from the camera's current pose only")
        folders = [await capture_view(robot, session.survey_dir / "view_000", frames=config.capture.survey_frames)]
    if not folders:
        raise RuntimeError("No survey views captured")

    # Detect in each view on its own: with camera calibration error, the views
    # disagree by several millimeters, and merging their clouds smears objects.
    detections, skipped, planes, clouds = [], [], [], []
    for folder in folders:
        view = load_view(folder)
        points, colors, pixels = back_project(view, 2, config.camera.min_depth, config.camera.max_depth)
        view_plane = fit_table_plane(points, config.workspace)
        others: list[DetectedObject] = []
        found = detect_objects(points, view_plane, config.workspace, colors, others)
        # Pieces of one object split by depth (a cable's plug and its flat coil) join in color.
        found, others = join_connected(found, others, points, pixels, view_plane.height(points), view.color, config.workspace)
        detections.append((view.camera_position[:2], found))
        skipped.append((view.camera_position[:2], others))
        planes.append(view_plane)
        clouds.append((points, view_plane.height(points)))
    region = table_area(clouds)
    use_table_area(config, region)
    if region:
        log.info("Table area seen: x %.0f to %.0f, y %.0f to %.0f", *region[0], *region[1])
    # A patch of bad depth (on bare table near an image's edge, say) can look like a
    # small object in one view. Keep an object only if at least half of the views that
    # saw its spot saw something standing there (on armfarm14, every such phantom was
    # tall in 1 of 3-5 views; real objects in nearly all).
    objects = []
    for obj in merge_views(detections):
        seen, standing = views_agree(obj, clouds)
        if 2 * standing >= seen:
            objects.append(obj)
        else:
            log.info("  skipped something at (%.0f, %.0f), %.0f x %.0f x %.0fmm: only %d of the %d survey views that "
                     "saw its spot saw it standing (a patch of bad depth)", *obj.center[:2], *obj.size, standing, seen)
    objects = [DetectedObject(index, o.center, o.size, o.yaw_deg, o.n_points, o.color) for index, o in enumerate(objects)]
    clutter = clutter_around(objects, merge_views(skipped))
    # Keep the table as seen by the camera closest to the objects.
    middle = np.mean([obj.center[:2] for obj in objects], axis=0) if objects else detections[0][0]
    plane = planes[int(np.argmin([np.linalg.norm(camera_xy - middle) for camera_xy, _ in detections]))]
    session.save_scene(objects, plane, clutter, region)
    robot.set_scene(plane, clutter)
    try:  # a picture for people: it must never stop the survey
        picture = survey_map(session, objects, clutter, config, plane.z_at(*middle))
        log.info("Map of the table: %s", picture)
        if robot.live:
            robot.live.show_map(picture)
    except Exception as error:
        log.warning("Couldn't draw the survey map: %s", error)
    session.update(intrinsics=(await robot.intrinsics()).__dict__, config=config.to_dict())
    log.info("Table at z=%.1fmm (tilt %.2fdeg); found %d object(s)", plane.z_at(*middle),
             np.degrees(np.arccos(plane.normal[2])), len(objects))
    for obj in objects:
        log.info("  %s", obj.describe())
    if clutter:
        log.info("%d other thing(s) on the table, kept as obstacles:", len(clutter))
        for thing in clutter:
            log.info("  (%.0f, %.0f), %.0f x %.0f x %.0fmm", *thing.center[:2], *thing.size)

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
    ''' The best view of the board among `folders`: (board pose lying flat, detection, tilt). '''
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


HOME_SLIDE_MM = 25.0  # how far each try moves toward the arm when the mat's center is out of reach
HOME_SLIDE_MAX_MM = 250.0


async def view_board_head_on(robot: Robot, session: Session, config: Config, center: np.ndarray) -> dict:
    ''' Look straight down at the board's center so all of it is in view, and check it. If
    the arm can't reach over the center, look from as close to it as it can, sliding toward
    the arm's base. Where the camera ends up becomes the home position (save_home). '''
    spec = BoardSpec()
    toward_base = -center[:2] / np.linalg.norm(center[:2])
    height = center[2] + config.board.view_height
    for slide in np.arange(0.0, HOME_SLIDE_MAX_MM + 1, HOME_SLIDE_MM):
        x, y = center[:2] + slide * toward_base
        _, pose, _ = await robot.plan_best_roll(config.machine.camera, lambda r, x=x, y=y: look_at([x, y, height], [x, y, height - 100], r))
        if pose is not None:
            break
    else:
        log.warning("  can't look down at the board from within %.0fmm of its center; no head-on view", HOME_SLIDE_MAX_MM)
        return {}
    if slide:
        log.info("  the board's center is out of reach; looking down from %.0fmm closer to the arm", slide)
    if not await move_camera_looking_down(robot, x, y, height):
        return {}
    # 3 frames, not the scan's frames_per_view (50 in a full scan): the board's pose and
    # depth check need no more, and each frame costs about 2.6s over the cloud connection.
    folder = await capture_view(robot, session.root / "board" / "head_on", frames=3)
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
    save_home(config, np.array([x, y, height]), center, float(slide))
    return {"pose": T_world_board.tolist(), "center": (T_world_board @ np.append(spec.center, 1.0))[:3].tolist(),
            "head_on": {"corners": int(len(found.ids)), "reprojection_px": found.reprojection_px,
                        "tilt_deg": tilt, "depth": depth}}


async def locate_near(robot: Robot, x: float, y: float, plane: TablePlane, session: Session, object_id: int,
                      config: Config) -> DetectedObject:
    ''' Look straight down at (x, y) and return the object found closest to it. '''
    height = plane.z_at(x, y) + config.views.standoff + 200  # 200mm higher: sees the whole station area
    if not await move_camera_looking_down(robot, x, y, height):
        raise RuntimeError("Cannot look at the scan station")
    folder = await capture_view(robot, session.object_dir(object_id) / "located", frames=3)
    found = detect_objects(scene_points([folder], config), plane, config.workspace)
    if not found:
        raise RuntimeError(f"No object found near ({x:.0f}, {y:.0f})")
    nearest = min(found, key=lambda o: np.hypot(o.center[0] - x, o.center[1] - y))
    nearest.id = object_id
    log.info("  relocated %s", nearest.describe())
    return nearest


async def confirm(prompt: str) -> bool:
    answer = await asyncio.to_thread(ask_operator, f"{prompt} [Enter = yes, n = skip] ")
    return answer.strip().lower() not in ("n", "no", "q")


async def plan_object(robot: Robot, obj: DetectedObject, others: list[DetectedObject],
                      config: Config) -> tuple[ScanPlan, PickAndPlace | None, ScanPlan | None]:
    ''' Plan views in place; if coverage is too low, plan a move to a better station. '''
    plan = await plan_scan(robot, obj, others, config)
    log.info("Object %d in place: %s", obj.id, plan.summary())
    if plan.coverage >= config.views.min_coverage or not config.relocate.enabled:
        return plan, None, None
    log.info("  coverage below %.0f%%; looking for a scan station", config.views.min_coverage * 100)
    station, station_plan = await choose_station(robot, obj, others, config)
    if station is None or station_plan.coverage <= plan.coverage + 0.05:  # worth a move only for 5+ points more
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


MAX_RESUME_SHIFT_MM = 3.0  # how far an object may have moved before a resumed scan refuses


class ObjectMoved(RuntimeError):
    ''' A resumed scan found the object (or the table) moved since its first views. '''


def captured_views(session: Session, object_id: int) -> list[tuple[int, Path, dict]]:
    ''' (index, folder, view.json) of each complete view of an object, in order. '''
    views = []
    for record in sorted(session.views_dir(object_id).glob("view_*/view.json")):
        views.append((int(record.parent.name.split("_")[1]), record.parent, json.loads(record.read_text())))
    return views


def saved_layout(session: Session, object_id: int, config: Config) -> tuple[Config, dict[float, float]]:
    ''' The ring layout an object's scan started with, so that a resume plans the same views:
    base_in_gap from the session's config (sessions from before it saved none: off), and
    the rings the planner turned to get around the arm's base (plan.json). '''
    saved = session.read().get("config", {}).get("views", {})
    config = replace(config, views=replace(config.views, base_in_gap=saved.get("base_in_gap", False)))
    path = session.object_dir(object_id) / "plan.json"
    record = (json.loads(path.read_text()).get("in_place") or {}) if path.exists() else {}
    return config, {float(elevation): turn for elevation, turn in record.get("ring_rotations_deg", {}).items()}


def why_skipped(error: Exception) -> str:
    ''' A move's failure in a few words, naming who refused it: Viam's motion planner (before
    the arm moves) or the arm's own controller (while it moves). '''
    message = str(error)
    if "Self Collision" in message:
        return "the arm's controller (xArm) predicted a self-collision and refused the move"
    if "zero IK solutions" in message or "too far" in message:
        return "unreachable according to Viam's motion planner (no joint angles reach it)"
    if "violation between" in message:
        between = message.split("violation between ", 1)[1].split(" geometries")[0]
        return f"Viam's motion planner found a collision between {between}"
    if isinstance(error, MissedTarget):
        return str(error)
    return message[:160]


def view_key(azimuth_deg: float, elevation_deg: float) -> tuple[int, int]:
    ''' Which planned view this is: its angle around the object and its ring. '''
    return round(float(azimuth_deg)) % 360, round(float(elevation_deg))


def missing_views(planned: list, done: list[tuple[int, Path, dict]]) -> list:
    ''' The planned views not yet captured (matched by angle around the object and ring). '''
    taken = {view_key(r["azimuth_deg"], r["elevation_deg"]) for _, _, r in done if "azimuth_deg" in r}
    return [v for v in planned if view_key(v.candidate.azimuth_deg, v.candidate.elevation_deg) not in taken]


def object_position(view, obj: DetectedObject, plane: TablePlane, config: Config) -> tuple[np.ndarray, float] | None:
    ''' The object's center (x, y) and top height (95th percentile above the table) in one view. '''
    points, _, _ = back_project(view, 2, config.camera.min_depth, config.camera.max_depth)
    keep = obj.contains(points, 40.0) & (plane.height(points) > 5.0)
    if keep.sum() < 50:
        return None
    return points[keep, :2].mean(axis=0), float(np.percentile(plane.height(points[keep]), 95))


async def check_unmoved(robot: Robot, session: Session, obj: DetectedObject, others: list[DetectedObject],
                        plane: TablePlane, config: Config) -> None:
    ''' Before resuming an object: look again from its first captured view (the top view if
    there is one) and compare where the object is. Raises ObjectMoved beyond 3mm. '''
    views = captured_views(session, obj.id)
    index, folder, record = next(((i, f, r) for i, f, r in views if r.get("elevation_deg") == 90.0), views[0])
    old = load_view(folder)
    pose = np.array(record.get("commanded_T_world_cam", record["T_world_cam"]))
    await robot.move(config.machine.camera, pose, robot.world_state([obj, *others], config.views.object_margin))
    new = load_view(await capture_view(robot, session.object_dir(obj.id) / "resume_check", frames=3))
    before, after = object_position(old, obj, plane, config), object_position(new, obj, plane, config)
    if before is None or after is None:
        raise ObjectMoved(f"Object {obj.id} isn't where view {index} saw it; start a fresh scan")
    shift = float(np.linalg.norm(after[0] - before[0]))
    rise = after[1] - before[1]
    log.info("  check against view %d: the object moved %.1fmm sideways, its top %+.1fmm", index, shift, rise)
    if shift > MAX_RESUME_SHIFT_MM or abs(rise) > MAX_RESUME_SHIFT_MM:
        raise ObjectMoved(f"Object {obj.id} moved {shift:.1f}mm sideways and {rise:+.1f}mm in height since its first "
                          f"views (limit {MAX_RESUME_SHIFT_MM:.0f}mm); start a fresh scan")


async def scan_object(robot: Robot, session: Session, obj: DetectedObject, others: list[DetectedObject],
                      plane: TablePlane, config: Config, execute: bool, step: bool,
                      max_views: int | None = None, resume: bool = False) -> None:
    ''' Plan and capture one object's views. With `resume`, continue an interrupted scan in
    the same session: views already captured are skipped and the new ones numbered after them. '''
    done = captured_views(session, obj.id) if resume else []
    if done:
        frames = {record.get("frames") for _, _, record in done}
        if frames != {config.capture.frames_per_view}:
            raise RuntimeError(f"Object {obj.id}'s views so far used {sorted(frames)} frames per view, this run "
                               f"{config.capture.frames_per_view}; resume with the same options (--full-scan or not)")
        target = scanned_object(session, obj.id)
        config, rotations = saved_layout(session, obj.id, config)
        plan = await plan_scan(robot, target, others, config, rotations)
        job = station_plan = None
    else:
        plan, job, station_plan = await plan_object(robot, obj, others, config)
        save_plan(session, obj, plan, job, station_plan)
    if not execute:
        return
    if step and not await confirm(f"Scan object {obj.id}?"):
        return

    target = scanned_object(session, obj.id) if done else obj
    if job is not None:
        if step and not await confirm(f"Move object {obj.id} to the scan station?"):
            job = None
        else:
            await execute_pick_and_place(robot, job, others, config)
            target = await locate_near(robot, *job.destination.center[:2], plane, session, obj.id, config)
            plan = await plan_scan(robot, target, others, config)
            log.info("  at station: %s", plan.summary())
    if not done:
        session.write_json(session.object_dir(obj.id) / "scanned_object.json", target.to_dict())

    world_state = robot.world_state([target, *others], config.views.object_margin)
    # With max_views, a subset spread over all directions (not just the first, highest views).
    views = spread_views(plan.views, max_views)
    if max_views is not None:
        log.info("  capturing %d of %d planned views, spread over all directions", len(views), len(plan.views))
    first = 0
    if done:
        views = missing_views(views, done)
        first = max(index for index, _, _ in done) + 1
        log.info("  resuming: %d view(s) already captured, %d still to capture", len(done), len(views))
    if robot.live:
        robot.live.start_object(session.root.name, target, plane, len(done) + len(views))
        for count, (_, folder, _) in enumerate(done):  # the preview starts where the scan left off
            robot.live.view_captured(folder, count)
    for number, view in enumerate(views):
        index = first + number
        try:
            await robot.move(config.machine.camera, view.T_world_cam, world_state)
        except MoveDeclined:  # End Scan, STOP, or a declined preview: end here, not "skip this view"
            raise
        except RobotUnresponsive:  # the robot stopped answering: end the run (scan.py says how to resume)
            raise
        except Exception as error:  # the planner can still fail on a path it approved from elsewhere
            log.warning("  view %d/%d (%s) skipped: %s", len(done) + number + 1, len(done) + len(views),
                        view.candidate.describe(), why_skipped(error))
            continue
        await capture_view(robot, session.views_dir(obj.id) / f"view_{index:03d}", {
            "azimuth_deg": view.candidate.azimuth_deg, "elevation_deg": view.candidate.elevation_deg,
            "roll_deg": view.roll_deg, "commanded_T_world_cam": view.T_world_cam.tolist()},
            keep_frames=config.capture.keep_frames)
        log.info("  view %d/%d (%s) captured", len(done) + number + 1, len(done) + len(views), view.candidate.describe())
        if robot.live:
            robot.live.view_captured(session.views_dir(obj.id) / f"view_{index:03d}", len(done) + number)
    if robot.live:
        robot.live.end_object()

    if job is not None and config.relocate.return_after_scan:
        back = await plan_pick_and_place(robot, target, obj, others, config)
        if back is None:
            log.warning("  cannot plan the move back; object %d stays at the station", obj.id)
        else:
            await execute_pick_and_place(robot, back, others, config)


def scanned_object(session: Session, object_id: int) -> DetectedObject:
    ''' The object as it was when scanned (it may have been moved to a station). '''
    objects, _ = session.load_scene()
    scanned = session.object_dir(object_id) / "scanned_object.json"
    return (DetectedObject.from_dict(json.loads(scanned.read_text())) if scanned.exists()
            else next(o for o in objects if o.id == object_id))


def calibrate_session(session: Session, object_id: int, config: Config) -> MountCalibration:
    ''' Estimate the camera mount from one object's views and save it with the session. '''
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
    ''' Recompute camera poses with the session's calibrated mount, when there is one. '''
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


def reconstruct_object(session: Session, object_id: int, plane: TablePlane, config: Config, refine: bool | None = None):
    obj = scanned_object(session, object_id)
    views = apply_session_calibration(session, load_views(session.views_dir(object_id)))
    log.info("Reconstructing object %d from %d views", object_id, len(views))
    result = reconstruct(views, obj, plane, config, refine)
    save_reconstruction(result, session.recon_dir(object_id))
    deviation = result.report["median_single_view_deviation_mm"]
    log.info("  mesh: %d triangles; single views deviate %.2fmm (median) from it -> %s",
             result.report["mesh"]["triangles"], deviation or float("nan"), session.recon_dir(object_id))
    return result


def progression_snapshot(session: Session, object_id: int, plane: TablePlane, config: Config,
                         refine: bool | None = None) -> Path:
    ''' Meshes from 1, 2, 4, ... all views (spread out) and a figure of them, saved with
    the object's results: object_NN/progression.svg (and .png). '''
    obj = scanned_object(session, object_id)
    views = apply_session_calibration(session, load_views(session.views_dir(object_id)))
    folder = session.recon_dir(object_id) / "progression"
    counts = sorted(set(doubling_counts(len(views))) | set(progression_counts(len(views))))
    progression(views, obj, plane, config, folder, counts, refine=refine)
    svg = session.object_dir(object_id) / "progression.svg"
    named = f" ({obj.color_name})" if obj.color_name else ""
    progression_figure(folder, svg, f"Object {object_id}{named}: reconstruction from 1 to {len(views)} views, seen from the arm's side",
                       progression_counts(len(views)), azimuth=facing_the_arm(obj.center))
    log.info("  progression figure: %s", svg.with_suffix(".png") if svg.with_suffix(".png").exists() else svg)
    return svg


def progression_video_snapshot(session: Session, object_id: int, plane: TablePlane, config: Config) -> Path:
    ''' The scan building up, view by view, saved with the object's results as
    object_NN/progression_vid.mp4 (and .gif), from the final reconstruction's aligned poses. '''
    folder = session.recon_dir(object_id)
    report = json.loads((folder / "report.json").read_text())
    poses = [np.array(pose) for pose in json.loads((folder / "refined_poses.json").read_text())]
    # The views the reconstruction used, in the same order as its poses.
    views = views_seeing(apply_session_calibration(session, load_views(session.views_dir(object_id))),
                         scanned_object(session, object_id), config)
    if len(views) != len(poses):
        raise RuntimeError(f"{len(views)} views but {len(poses)} saved poses; rebuild the mesh first (scan.py reconstruct)")
    obj = DetectedObject.from_dict(report["object"])
    planes = [local_table_plane(view, pose, obj, plane, config) for view, pose in zip(views, poses)]
    path = progression_video(views, poses, planes, obj, config, session.object_dir(object_id) / "progression_vid.mp4",
                             "Object scan, view by view", azimuth=facing_the_arm(obj.center))
    log.info("  progression video: %s", path)
    return path


def finish_object(session: Session, object_id: int, plane: TablePlane, config: Config, refine: bool | None = None) -> None:
    ''' Everything offline after an object's views are captured: its mesh, then the
    progression figure and video. '''
    reconstruct_object(session, object_id, plane, config, refine)
    try:  # a figure for people: it must never stop the run
        progression_snapshot(session, object_id, plane, config, refine)
    except Exception as error:
        log.warning("  couldn't make the progression figure: %s", error)
    try:
        progression_video_snapshot(session, object_id, plane, config)
    except Exception as error:
        log.warning("  couldn't make the progression video: %s", error)


async def search_for_board(robot: Robot, session: Session, config: Config, spec: BoardSpec,
                           search: bool) -> np.ndarray | None:
    ''' Where the board is, roughly, from color-only views (about 0.15s each): first from
    where the camera is, then (with `search`) from overhead positions around it, nearest first. '''
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
    workspace = config.workspace
    # Overhead positions around where the camera is (up to 3 survey steps away), nearest first.
    steps = np.array([(i, j) for i in range(-3, 4) for j in range(-3, 4)], dtype=float) * workspace.survey_spacing
    spots = [here[:2] + s for s in sorted(steps, key=np.linalg.norm) if np.hypot(*(here[:2] + s)) >= workspace.base_clearance]
    for index, (x, y, z) in enumerate((np.array([*xy, workspace.survey_height]) for xy in spots), 1):
        if await move_camera_looking_down(robot, x, y, z):
            center = await look(f"search_{index:03d}")
            if center is not None:
                return center
    return None


async def board_calibration(robot: Robot, session: Session, config: Config, execute: bool,
                            spec: BoardSpec | None = None, known_center: np.ndarray | None = None) -> dict | None:
    ''' Hand-eye calibration and a depth check with a ChArUco board lying on the table.

    Locates the board (from the survey's `known_center`, or from where the camera is
    now), plans views of it from several distances and directions (plan-only), and
    with `execute` captures them and solves for the camera mount and the depth
    sensor's error.
    '''
    spec = spec or BoardSpec()
    found_from_here = known_center is None
    if found_from_here:
        known_center = await search_for_board(robot, session, config, spec, execute)
        if known_center is None:
            raise RuntimeError("The board isn't in view; add --execute to search for it (color-only, quick), or run a survey")
    if execute and found_from_here:
        # No saved record of this arm's board: its camera mount isn't trusted yet (a new arm's
        # can be off by tens of degrees, which puts the board hundreds of mm from where it
        # is). So first a rough calibration in place, from small turns of the camera that keep
        # the board in view, before moving anywhere; then the board's center through it.
        # The search may have caught the board at the edge of a picture: center it first, so
        # the small turns keep all of it in view. The mount from the Viam app's camera frame
        # is close enough to aim with (it put the board about 25mm off on armfarm12).
        if not await move_camera_looking_down(robot, known_center[0], known_center[1],
                                              known_center[2] + config.board.view_height):
            log.info("  can't look straight down at the board's rough center; turning from where the camera is")
        log.info("Rough calibration first, turning the camera a little in place over the board")
        start = await capture_view(robot, session.root / "board" / "start", {"frames": 3}, 3)
        view = load_view(start)
        found = detect(view, spec)
        if found is None:
            raise RuntimeError("The board isn't in view over its rough center; is the camera frame in the Viam app "
                               "roughly where the camera really is?")
        log.info("  %d of %d corners in view", len(found.ids), (spec.squares_x - 1) * (spec.squares_y - 1))
        mount = await rough_mount(robot, session, config, spec, view, found)
        view = replace(view, T_world_cam=view.T_world_cam @ np.linalg.inv(await robot.camera_mount()) @ mount)
        robot.use_camera_mount(mount)
        known_center = (locate_flat(view, found, spec) @ np.append(spec.center, 1.0))[:3]
        log.info("  board centered at (%.0f, %.0f, %.0f) through the rough mount, %.1fdeg from level",
                 *known_center, board_tilt(view, found))
    if not execute:
        center = known_center
        log.info("Planning around the board centered at (%.0f, %.0f, %.0f)", *center)
    else:
        # Head-on, with the whole board in view, to pin down where it is.
        await move_camera_looking_down(robot, known_center[0], known_center[1], known_center[2] + config.board.view_height)
        located = await capture_view(robot, session.root / "board" / "located", {"frames": 3}, 3)
        view = load_view(located)
        found = detect(view, spec)
        if found is None and not found_from_here:
            # Not where this arm last saw it: the mat was moved. Look for it as on a first run.
            log.info("The board isn't where this arm last saw it; looking for it from here and nearby")
            known_center = await search_for_board(robot, session, config, spec, True)
            if known_center is not None:
                await move_camera_looking_down(robot, known_center[0], known_center[1], known_center[2] + config.board.view_height)
                view = load_view(await capture_view(robot, session.root / "board" / "located", {"frames": 3}, 3))
                found = detect(view, spec)
        if found is None:
            raise RuntimeError("The board isn't in the head-on view")
        # The mat lies on the table, so it must look level. If it doesn't, the camera mount
        # we're using is wrong (as on a new arm, before its first calibration), and the
        # planned views could miss the board: first solve a rough mount from a few small
        # turns of the camera, which keep the board in view.
        tilt = board_tilt(view, found)
        if tilt > MAX_BOARD_TILT_DEG:
            log.info("The board looks %.1fdeg from level through the camera mount in use (expected on a new arm); "
                     "turning the camera a little in place to estimate a rough mount first", tilt)
            mount = await rough_mount(robot, session, config, spec, view, found)
            view = replace(view, T_world_cam=view.T_world_cam @ np.linalg.inv(await robot.camera_mount()) @ mount)
            robot.use_camera_mount(mount)
            tilt = board_tilt(view, found)
            if tilt > MAX_BOARD_TILT_DEG:
                raise RuntimeError(f"The board still looks {tilt:.1f}deg from level after the rough calibration; "
                                   "is it flat on the table?")
        T_world_board = locate_flat(view, found, spec)
        center = (T_world_board @ np.append(spec.center, 1.0))[:3]
        log.info("Board head-on: %d corners, %.2fpx, %.1fdeg from level; center (%.0f, %.0f, %.0f)",
                 len(found.ids), found.reprojection_px, tilt, *center)

    camera = config.machine.camera
    # Candidate board views: 3 distances that keep the 180 x 135mm board well inside the
    # image, steep enough elevations that every corner is seen, every 60deg around.
    # Judgment; the hand-eye solve wants varied rotations more than many views.
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
        # 100mm between cameras counts as much as about 37deg between viewing directions
        # ((1 - cos 37deg) x 5 = 1): both matter for hand-eye calibration.
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
        try:
            await robot.move(camera, pose)
        except (MoveDeclined, RobotUnresponsive):
            raise
        except Exception as error:  # the move's own plan can fail where the plan-only check passed
            log.warning("  view %d/%d skipped: %s", index + 1, len(chosen), why_skipped(error))
            continue
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
    if len(found_views) < 5:  # hand-eye needs several distinct rotations; 5 is a floor
        raise RuntimeError(f"Only {len(found_views)} usable board views")
    calibrated, T_world_board, diagnostics = hand_eye(ends, found_views, views[0].intrinsics.matrix, mount)
    result = {"original_mount": mount.tolist(), "board_mount": calibrated.tolist(), "board_pose": T_world_board.tolist(),
              "hand_eye": diagnostics, "depth": depth_model(views, found_views)}
    session.write_json(session.root / "board" / "board_calibration.json", result)
    return result


def board_tilt(view, found) -> float:
    ''' How far from level (degrees) the board looks in `view`, through the camera mount it was taken with. '''
    normal = (view.T_world_cam @ found.T_cam_board)[:3, 2]
    return float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))


ROUGH_TURNS_DEG = (("x", 8.0), ("x", -8.0), ("y", 8.0), ("y", -8.0), ("z", 10.0), ("z", -10.0))


async def rough_mount(robot: Robot, session: Session, config: Config, spec: BoardSpec, view, found) -> np.ndarray:
    ''' A rough camera mount from the board seen while the camera only turns in place (8-10deg
    each way from `view`, so the board stays in view): enough to plan the full calibration's
    views. In simulation it recovered mounts that were 2-44deg and 22-55mm off to within
    0.1deg and 0.5mm, from these 7 views. '''
    mount = await robot.camera_mount()
    ends, founds = [view.T_world_cam @ np.linalg.inv(mount)], [found]
    for index, (axis, angle) in enumerate(ROUGH_TURNS_DEG):
        target = view.T_world_cam.copy()
        target[:3, :3] = view.T_world_cam[:3, :3] @ Rotation.from_euler(axis, angle, degrees=True).as_matrix()
        try:
            await robot.move(config.machine.camera, target)
        except Exception as error:  # an unreachable turn: use the others
            log.info("  turn %d skipped: %s", index + 1, error)
            continue
        folder = await capture_view(robot, session.root / "board" / f"rough_{index:03d}", {"frames": 3}, 3)
        turned = load_view(folder)
        seen = detect(turned, spec)
        if seen is not None:
            ends.append(turned.T_world_cam @ np.linalg.inv(mount))
            founds.append(seen)
    if len(founds) < 5:
        raise RuntimeError(f"Only {len(founds)} views of the board while turning the camera; is it in view?")
    rough, _, diagnostics = hand_eye(ends, founds, view.intrinsics.matrix, mount)
    log.info("  rough mount from %d views: %.1fmm / %.1fdeg from the one in use (%.2fpx)", len(founds),
             np.linalg.norm(rough[:3, 3] - mount[:3, 3]), rotation_angle_deg(mount[:3, :3].T @ rough[:3, :3]),
             diagnostics.get("reprojection_rms_px_after", float("nan")))
    return rough


def save_home(config: Config, camera: np.ndarray, board_center: np.ndarray, slide_mm: float) -> None:
    ''' Make `camera` (looking straight down at the board) the home position every run starts at. '''
    if not config.machine.home_file:
        return
    path = config.resolve(config.machine.home_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"camera_mm": camera.round(1).tolist(), "board_center_mm": board_center.round(1).tolist(),
                                "slid_toward_arm_mm": slide_mm, "saved": time.strftime("%Y-%m-%d %H:%M"),
                                "note": "Home position: the camera here, looking straight down at the ChArUco mat. "
                                        "Found by the survey; delete this file to find it again."}, indent=2) + "\n")
    log.info("  home position saved: camera at (%.0f, %.0f, %.0f), looking down at the mat (%s)", *camera, path.name)


def save_mount(config: Config, mount: np.ndarray, source: str) -> Path:
    ''' Make `mount` the camera mount used for every move and capture. '''
    if not config.machine.camera_mount_file:
        raise ValueError("Set [machine] camera_mount_file in config.toml first")
    path = config.resolve(config.machine.camera_mount_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mount": mount.tolist(), "source": source, "saved": time.strftime("%Y-%m-%d %H:%M"),
                                "note": "Camera pose relative to the arm's end (mm). Used for every move and capture."},
                               indent=2) + "\n")
    return path


def prune_board_runs(config: Config) -> list[str]:
    ''' Delete board calibration runs (data/board-*) that no calibration file names as its
    source: each arm keeps only the run behind its current camera mount, since a new
    calibration simply replaces the old one. A run with a file named KEEP in it stays
    (one used by analysis/compare_calibration.py). Returns the names deleted. '''
    import shutil
    used = set()
    for path in config.resolve("calibration").glob("*.json"):
        try:
            source = json.loads(path.read_text()).get("source", "")
        except (OSError, json.JSONDecodeError):
            continue
        used.update(word.strip() for word in str(source).split(",") if word.strip().startswith("board-"))
    deleted = []
    for folder in sorted(config.resolve(config.data_dir).glob("board-*")):
        if folder.is_dir() and folder.name not in used and not (folder / "KEEP").exists():
            shutil.rmtree(folder)
            deleted.append(folder.name)
    return deleted
