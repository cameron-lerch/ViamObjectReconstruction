"""Touch-off: find where the gripper's fingertips really are by touching the table.

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
"""

import asyncio
import logging
import time
from itertools import product

import numpy as np

from .config import Config
from .robot import MoveDeclined, Robot
from .scene import DetectedObject, TablePlane
from .transforms import make_transform, pose_to_matrix, rotation_z

log = logging.getLogger(__name__)

PAPER_MM = 0.1
START_GAP = 30.0  # modeled fingertips this far above the table before stepping
FINE_FROM = 6.0  # switch to fine steps this far above the expected contact
COARSE_STEP, FINE_STEP = 2.0, 0.5
FLOOR = 10.0  # never go more than this past where the model says contact is


async def modeled_reach(robot: Robot) -> float:
    """How far past the gripper frame's origin, along its pointing axis, the collision model reaches."""
    reach = 0.0
    for geometry in await robot.gripper.get_geometries():
        box = geometry.box.dims_mm
        center = pose_to_matrix(geometry.center)
        half = np.array([box.x, box.y, box.z]) / 2
        for signs in product((-1, 1), repeat=3):
            corner = center[:3, :3] @ (half * signs) + center[:3, 3]
            reach = max(reach, float(corner[2]))
    return reach


def pointing_down(x: float, y: float, z: float, roll: float) -> np.ndarray:
    """Gripper frame at (x, y, z) with its pointing axis (+z) straight down."""
    down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    return make_transform(down @ rotation_z(roll), [x, y, z])


def contact_spots(objects: list[DetectedObject], plane: TablePlane, config: Config, board_center=None,
                  count: int = 3, clearance: float = 80.0) -> list[np.ndarray]:
    """Bare-table spots near the objects, clear of them and of the calibration board,
    spread out (farthest-first)."""
    workspace = config.workspace
    middle = np.mean([o.center[:2] for o in objects], axis=0) if objects else np.array([400.0, -300.0])
    candidates = []
    for radius in (150.0, 220.0):
        for angle in np.radians(np.arange(0, 360, 20)):
            xy = middle + radius * np.array([np.cos(angle), np.sin(angle)])
            (x0, x1), (y0, y1) = workspace.x_range, workspace.y_range
            if not (x0 < xy[0] < x1 and y0 < xy[1] < y1) or np.linalg.norm(xy) < workspace.min_base_distance + 50:
                continue
            if any(np.linalg.norm(xy - o.center[:2]) < o.footprint_radius + clearance for o in objects):
                continue
            if board_center is not None and np.linalg.norm(xy - np.asarray(board_center)[:2]) < 150:
                continue
            candidates.append(np.array([xy[0], xy[1], plane.z_at(*xy)]))
    spots = candidates[:1]
    while len(spots) < count and len(spots) < len(candidates):
        spots.append(max(candidates, key=lambda c: min(np.linalg.norm(c[:2] - s[:2]) for s in spots)))
    return spots


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip().lower()


async def touch_off(robot: Robot, objects: list[DetectedObject], plane: TablePlane, config: Config,
                    execute: bool, board_center=None) -> dict | None:
    gripper = config.machine.gripper
    model = await modeled_reach(robot)
    spots = contact_spots(objects, plane, config, board_center)
    if not spots:
        raise RuntimeError("No bare-table spot found near the objects")
    log.info("Gripper model reaches %.1fmm past the gripper frame. Touch-off spots (table height from the survey):", model)
    plans = []
    for x, y, table_z in spots:
        roll, pose, reason = await robot.plan_best_roll(
            gripper, lambda r, x=x, y=y, z=table_z + model + START_GAP: pointing_down(x, y, z, r))
        log.info("  (%.0f, %.0f), table at z=%.1f: %s", x, y, table_z, "reachable" if pose is not None else reason)
        if pose is not None:
            plans.append((np.array([x, y, table_z]), roll))
    if not execute:
        log.info("Planning only. Add --execute to touch off (have a sheet of paper ready).")
        return None

    log.info("Closing the gripper: the closed fingertips are what touches.")
    await robot.gripper.grab()
    results = []
    for spot, roll in plans:
        x, y, table_z = spot
        start = table_z + model + START_GAP
        robot.use_table_slab = True
        await robot.move(gripper, pointing_down(x, y, start, roll), robot.world_state([], 0.0))
        print(f"\nAt ({x:.0f}, {y:.0f}): fingertips about {START_GAP:.0f}mm above the table (by the gripper model).\n"
              "Slide a sheet of paper under them. Enter = step down, t = paper drags (touching), q = stop.")
        robot.use_table_slab = False
        z, floor = start, table_z + model - FLOOR
        touched = False
        try:
            while True:
                step = FINE_STEP if z - (table_z + model) <= FINE_FROM else COARSE_STEP
                answer = await ask(f"  fingertips ~{z - table_z - model:+5.1f}mm by the model; Enter = down {step}mm, t = touching, q = stop: ")
                if answer == "t":
                    touched = True
                    break
                if answer == "q":
                    break
                if z - step < floor:
                    log.warning("  stopped: %.0fmm past where the model expects contact, and no contact", FLOOR)
                    break
                z -= step
                await robot.move(gripper, pointing_down(x, y, z, roll), None, linear=True)
        finally:
            if touched:
                await robot.wait_until_still()
                contact = (await robot.frame_pose(gripper))[2, 3]
                reach = contact - table_z - PAPER_MM
                results.append({"spot": [float(x), float(y)], "table_z": float(table_z), "gripper_z": float(contact),
                                "reach_mm": float(reach), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                log.info("  contact: gripper frame at z=%.2f, table at z=%.2f -> fingertips reach %.1fmm (model: %.1fmm)",
                         contact, table_z, reach, model)
            await robot.move(gripper, pointing_down(x, y, start, roll), None, linear=True)
            robot.use_table_slab = True
        if not touched:
            raise MoveDeclined("touch-off stopped without contact")
    reaches = [r["reach_mm"] for r in results]
    summary = {"modeled_reach_mm": model, "reach_mm": float(np.median(reaches)), "reach_spread_mm": float(np.ptp(reaches)),
               "extra_reach_mm": float(np.median(reaches) - model), "points": results}
    return summary
