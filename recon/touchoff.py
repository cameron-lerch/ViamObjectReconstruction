''' Touch-off: find where the gripper's fingertips really are by touching the table.

The planner keeps the gripper off the table using the gripper's collision model.
If the real fingertips reach further than that model, the planner will happily
drive them into the table. A touch-off measures the real reach the way
machinists set up a tool: the closed gripper, pointing straight down, steps
toward bare table while the operator slides a sheet of paper under the
fingertips and stops it the moment the paper drags (about 0.1mm thick).

    reach = height of the gripper frame at contact - height of the table

The table's height comes from the survey (the camera, which the ChArUco board
showed is accurate to about half a millimeter). The result is saved to the
touch-off file; `Robot.table_slab` adds any reach beyond the model to its margin.
On armfarm14 the fingertips reach 76mm, 26mm past the model's 50mm.

At each of three spots of bare table, spread out: lift straight up until the
fingertips are 600mm above the table (clear of anything standing on it), move
over the spot, come straight down to 10mm above the table, step down 1mm at a
time until the paper drags, and go straight back up. Three contacts also show
how the table the arm touches compares with the table the camera sees.
'''

import asyncio
import json
import logging
import time
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from .config import Config
from .robot import MoveDeclined, Robot
from .scene import DetectedObject, TablePlane
from .transforms import make_transform, pose_to_matrix, rotation_z

log = logging.getLogger(__name__)

PAPER_MM = 0.1  # a sheet of printer paper
TRAVEL = 600.0  # fingertips this far above the table while moving between spots
START_GAP = 10.0  # come straight down to this far above the table (by the measured reach)...
UNMEASURED_GAP = 40.0  # ...or this far by the gripper model, if the reach was never measured (the model was 26mm short)
STEP, MAX_STEP = 1.0, 5.0  # Enter steps 1mm; typing a number steps that far (up to 5mm)
FLOOR = 10.0  # never go more than this past the expected contact (40mm when only the model is known)
CLEARANCE = 120.0  # from a spot to anything standing on the table (the open claws are 170mm wide)
VICINITY = 150.0  # ...and at most this much farther from the nearest scan target: the table that matters is near them
BOARD_CLEARANCE = 150.0  # from a spot to the calibration board's center (the board is 180 x 135mm)


async def modeled_reach(robot: Robot) -> float:
    ''' How far past the gripper frame's origin, along its pointing axis, the collision model reaches. '''
    reach = 0.0
    for geometry in await robot.gripper.get_geometries():
        box = geometry.box.dims_mm
        center = pose_to_matrix(geometry.center)
        half = np.array([box.x, box.y, box.z]) / 2
        for signs in product((-1, 1), repeat=3):
            corner = center[:3, :3] @ (half * signs) + center[:3, 3]
            reach = max(reach, float(corner[2]))
    return reach


def measured_reach(config: Config) -> float | None:
    ''' The reach saved by an earlier touch-off, if any. '''
    path = config.resolve(config.machine.touch_off_file) if config.machine.touch_off_file else None
    if path is None or not path.exists():
        return None
    return float(json.loads(path.read_text())["reach_mm"])


def pointing_down(x: float, y: float, z: float, roll: float) -> np.ndarray:
    ''' Gripper frame at (x, y, z) with its pointing axis (+z) straight down. '''
    down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    return make_transform(down @ rotation_z(roll), [x, y, z])


def bare_table(points: np.ndarray, plane: TablePlane, radius: float = 40.0, spacing: float = 5.0):
    ''' A test for "the survey saw bare table all over this disc" (so a spot isn't off
    the table's edge, or somewhere the survey never looked). The 40mm disc is wider
    than the closed fingertips; table means within 3mm of the survey's plane. '''
    table_xy = points[np.abs(plane.height(points)) < 3.0][:, :2]
    table_xy = np.unique(np.round(table_xy / spacing), axis=0) * spacing
    tree = cKDTree(table_xy)
    u, v = np.meshgrid(np.arange(-radius, radius + spacing, spacing), np.arange(-radius, radius + spacing, spacing))
    disc = np.column_stack([u.ravel(), v.ravel()])
    disc = disc[np.linalg.norm(disc, axis=1) <= radius]

    def covered(xy: np.ndarray) -> bool:
        gaps, _ = tree.query(disc + xy, distance_upper_bound=spacing)
        return np.isfinite(gaps).mean() >= 0.95  # a few missing readings are normal
    return covered


def candidate_spots(obstacles: list[DetectedObject], plane: TablePlane, config: Config, board_center=None,
                    on_table=None, clearance: float = CLEARANCE, spacing: float = 20.0,
                    targets: list[DetectedObject] | None = None, vicinity: float = VICINITY) -> list[np.ndarray]:
    ''' Bare-table spots on a grid over the search region, clear of everything the
    survey saw standing on the table (scan targets and clutter) and of the board,
    and, given `targets`, near one of them (within `vicinity` beyond the clearance).
    `on_table` (from `bare_table`) checks the survey really saw table there. '''
    workspace = config.workspace
    (x0, x1), (y0, y1) = workspace.x_range, workspace.y_range
    xs, ys = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
    spots = []
    for xy in np.column_stack([xs.ravel(), ys.ravel()]):
        if np.linalg.norm(xy) < workspace.base_clearance:
            continue
        if any(np.linalg.norm(xy - o.center[:2]) < o.footprint_radius + clearance for o in obstacles):
            continue
        if targets and min(np.linalg.norm(xy - o.center[:2]) - o.footprint_radius for o in targets) > clearance + vicinity:
            continue
        if board_center is not None and np.linalg.norm(xy - np.asarray(board_center)[:2]) < BOARD_CLEARANCE:
            continue
        if on_table is not None and not on_table(xy):
            continue
        spots.append(np.array([xy[0], xy[1], plane.z_at(*xy)]))
    return spots


def spread_order(spots: list[np.ndarray]) -> list[np.ndarray]:
    ''' Spots reordered so that each is as far as possible from those before it,
    starting with the one nearest the middle: any first three are well spread. '''
    if not spots:
        return []
    xy = np.array([s[:2] for s in spots])
    first = int(np.argmin(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
    order = [first]
    nearest = np.linalg.norm(xy - xy[first], axis=1)  # each spot's distance to the closest one already picked
    nearest[first] = -1.0
    for _ in range(len(spots) - 1):
        k = int(np.argmax(nearest))
        order.append(k)
        nearest = np.minimum(nearest, np.linalg.norm(xy - xy[k], axis=1))
        nearest[order] = -1.0
    return [spots[k] for k in order]


def travel_height(table_z: float, reach: float) -> float:
    ''' Gripper-frame height with the fingertips TRAVEL above the table. '''
    return table_z + reach + TRAVEL


def table_comparison(results: list[dict]) -> dict:
    ''' The table the arm touched vs the table the camera saw, from the contacts.

    At each spot, contact height - camera table height = reach + (camera error there).
    The reach is the same everywhere, so any slope across the spots is a tilt
    between the two tables. With exactly three spots the plane fits them exactly.
    '''
    if len(results) < 3:
        return {}
    xy = np.array([r["spot"] for r in results])
    offset = np.array([r["gripper_z"] - r["table_z"] for r in results])
    A = np.column_stack([xy, np.ones(len(xy))])
    (a, b, _), *_ = np.linalg.lstsq(A, offset, rcond=None)
    slope = float(np.hypot(a, b))
    return {"tilt_deg": float(np.degrees(np.arctan(slope))), "tilt_mm_per_100mm": 100 * slope}


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip().lower()


def step_size(answer: str) -> float | None:
    ''' Enter = the standard step; a number = that many mm (capped); anything else = None. '''
    if answer == "":
        return STEP
    try:
        return min(MAX_STEP, max(0.1, float(answer)))
    except ValueError:
        return None


def without_table(robot: Robot, objects: list[DetectedObject], margin: float):
    ''' Obstacles for the straight up and down moves, which go below the table slab's
    safety margin on purpose (everything standing on the table still counts). '''
    robot.use_table_slab = False
    try:
        return robot.world_state(objects, margin)
    finally:
        robot.use_table_slab = True


async def lift_to(robot: Robot, z: float, world_state) -> None:
    ''' Straight up from wherever the gripper is, keeping its orientation (skipped if it is already that high). '''
    gripper = robot.config.machine.gripper
    T = await robot.frame_pose(gripper)
    if T[2, 3] < z - 1.0:
        T[2, 3] = z
        await robot.move(gripper, T, world_state, linear=True)


async def touch_off(robot: Robot, objects: list[DetectedObject], plane: TablePlane, config: Config,
                    execute: bool, board_center=None, survey_points: np.ndarray | None = None) -> dict | None:
    gripper = config.machine.gripper
    model = await modeled_reach(robot)
    known = measured_reach(config)
    expected = known if known is not None else model
    gap, floor_gap = (START_GAP, FLOOR) if known is not None else (UNMEASURED_GAP, 4 * FLOOR)
    margin = config.views.object_margin
    # Spots avoid the clutter too; robot.world_state adds it to every plan by itself.
    obstacles = [*objects, *robot.clutter]
    on_table = bare_table(survey_points, plane) if survey_points is not None else None
    candidates = spread_order(candidate_spots(obstacles, plane, config, board_center, on_table, targets=objects))
    if len(candidates) < 3:
        log.warning("Fewer than 3 open spots near the objects; using open table anywhere")
        candidates = spread_order(candidate_spots(obstacles, plane, config, board_center, on_table))
    log.info("Gripper model reaches %.1fmm past the gripper frame; %s. %d bare-table spots clear of %d thing(s) on the table:",
             model, f"an earlier touch-off measured {known:.1f}mm" if known is not None else "the real reach was never measured",
             len(candidates), len(obstacles))
    plans = []
    for x, y, table_z in candidates:
        high = travel_height(table_z, expected)
        roll, pose, reason = await robot.plan_best_roll(
            gripper, lambda r, x=x, y=y: pointing_down(x, y, high, r), robot.world_state(objects, margin))
        log.info("  (%.0f, %.0f), table at z=%.1f: %s", x, y, table_z,
                 f"reachable with the fingertips {TRAVEL:.0f}mm up" if pose is not None else reason)
        if pose is not None:
            plans.append((np.array([x, y, table_z]), roll, high))
        if len(plans) == 3:
            break
    if not plans:
        raise RuntimeError(f"No touch-off spot is reachable with the fingertips {TRAVEL:.0f}mm above the table")
    if not execute:
        log.info("Planning only. Add --execute to touch off (have a sheet of paper ready).")
        return None

    log.info("Closing the gripper: the closed fingertips are what touches.")
    await robot.gripper.grab()
    results = []
    for spot, roll, high in plans:
        x, y, table_z = spot
        start = table_z + expected + gap
        straight = without_table(robot, objects, margin)
        await lift_to(robot, high, straight)
        await robot.move(gripper, pointing_down(x, y, high, roll), robot.world_state(objects, margin))
        await robot.move(gripper, pointing_down(x, y, start, roll), straight, linear=True)
        print(f"\nAt ({x:.0f}, {y:.0f}): fingertips about {gap:.0f}mm above the table.\n"
              f"Slide a sheet of paper under them. Enter = down {STEP:.0f}mm, a number = down that many mm "
              f"(up to {MAX_STEP:.0f}), t = paper drags (touching), q = stop.")
        z, floor = start, table_z + expected - floor_gap
        touched = False
        try:
            while True:
                answer = await ask(f"  fingertips ~{z - table_z - expected:+5.1f}mm above the table: ")
                if answer == "t":
                    touched = True
                    break
                step = step_size(answer)
                if step is None:
                    break
                if z - step < floor:
                    log.warning("  stopped: that would go more than %.0fmm past the expected contact", floor_gap)
                    break
                z -= step
                await robot.move(gripper, pointing_down(x, y, z, roll), straight, linear=True)
        finally:
            if touched:
                await robot.wait_until_still()
                contact = (await robot.frame_pose(gripper))[2, 3]
                reach = contact - table_z - PAPER_MM
                results.append({"spot": [float(x), float(y)], "table_z": float(table_z), "gripper_z": float(contact),
                                "reach_mm": float(reach), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                log.info("  contact: gripper frame at z=%.2f, table at z=%.2f -> fingertips reach %.1fmm (model: %.1fmm)",
                         contact, table_z, reach, model)
            await robot.move(gripper, pointing_down(x, y, high, roll), straight, linear=True)
        if not touched:
            raise MoveDeclined("touch-off stopped without contact")
    reaches = [r["reach_mm"] for r in results]
    return {"modeled_reach_mm": model, "reach_mm": float(np.median(reaches)), "reach_spread_mm": float(np.ptp(reaches)),
            "extra_reach_mm": float(np.median(reaches) - model), "table_vs_camera": table_comparison(results),
            "points": results}
