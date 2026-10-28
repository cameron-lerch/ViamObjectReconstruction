''' Move an object the arm cannot scan well to a scan station, and back afterward.

Grasps are top-down, with the jaws closing across the object's narrow side.
While the object is carried it is attached to the gripper in the planner's
world state, so the planned path keeps the object clear of obstacles too.
'''

import logging
import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .planner import ScanPlan, plan_scan
from .robot import Robot
from .scene import DetectedObject
from .transforms import make_transform

log = logging.getLogger(__name__)

PLACE_CLEARANCE = 2.0  # release the object this far above the table


@dataclass
class PickAndPlace:
    obj: DetectedObject
    destination: DetectedObject
    T_grasp: np.ndarray  # gripper frame at the grasp
    T_place: np.ndarray  # gripper frame at the release

    def above(self, T: np.ndarray, height: float) -> np.ndarray:
        raised = T.copy()
        raised[2, 3] += height
        return raised


def grasp_options(obj: DetectedObject, config: Config) -> list[np.ndarray]:
    ''' Top-down gripper poses closing across the object's width; empty if it is too wide. '''
    relocate = config.relocate
    if obj.size[1] > relocate.max_grasp_width:
        return []
    depth = min(relocate.grasp_depth, obj.size[2] / 2)
    tip_z = obj.top_z - depth
    options = []
    for flip in (0.0, 180.0):
        angle = math.radians(obj.yaw_deg + 90 + relocate.jaw_axis_offset_deg + flip)
        z_axis = np.array([0.0, 0.0, -1.0])
        y_axis = np.array([math.cos(angle), math.sin(angle), 0.0])
        rotation = np.column_stack([np.cross(y_axis, z_axis), y_axis, z_axis])
        origin = [obj.center[0], obj.center[1], tip_z + relocate.fingertip_offset]
        options.append(make_transform(rotation, origin))
    return options


def station_is_free(obj: DetectedObject, x: float, y: float, others: list[DetectedObject], clearance: float) -> bool:
    return all(
        np.hypot(other.center[0] - x, other.center[1] - y) >= obj.footprint_radius + other.footprint_radius + clearance
        for other in others
    )


async def choose_station(robot: Robot, obj: DetectedObject, others: list[DetectedObject],
                         config: Config) -> tuple[DetectedObject, ScanPlan] | tuple[None, None]:
    ''' First free station whose scan plan reaches the coverage target, else the best one seen. '''
    best: tuple[DetectedObject, ScanPlan] | tuple[None, None] = (None, None)
    for x, y in config.relocate.stations:
        if not station_is_free(obj, x, y, others, config.relocate.min_clearance):
            log.info("  station (%.0f, %.0f) is occupied", x, y)
            continue
        moved = obj.moved_to(x, y)
        plan = await plan_scan(robot, moved, others, config)
        log.info("  station (%.0f, %.0f): %s", x, y, plan.summary())
        if plan.coverage >= config.views.min_coverage:
            return moved, plan
        if best[1] is None or plan.coverage > best[1].coverage:
            best = (moved, plan)
    return best


async def plan_pick_and_place(robot: Robot, obj: DetectedObject, destination: DetectedObject,
                              others: list[DetectedObject], config: Config) -> PickAndPlace | None:
    ''' Find a grasp whose approach, grasp and place poses all have plans. '''
    gripper = config.machine.gripper
    margin = config.views.object_margin
    height = config.relocate.approach_height
    for T_grasp in grasp_options(obj, config):
        T_place = destination.pose @ np.linalg.inv(obj.pose) @ T_grasp
        T_place[2, 3] += PLACE_CLEARANCE
        job = PickAndPlace(obj, destination, T_grasp, T_place)
        checks = [
            (job.above(T_grasp, height), robot.world_state([obj, *others], margin)),
            (T_grasp, robot.world_state(others, margin)),
            (job.above(T_place, height), robot.world_state(others, margin)),
            (T_place, robot.world_state(others, margin)),
        ]
        for pose, world_state in checks:
            trajectory, reason = await robot.plan(gripper, pose, world_state)
            if trajectory is None:
                log.info("  grasp option rejected: %s", reason)
                break
        else:
            return job
    return None


async def execute_pick_and_place(robot: Robot, job: PickAndPlace, others: list[DetectedObject], config: Config) -> None:
    gripper = config.machine.gripper
    margin = config.views.object_margin
    height = config.relocate.approach_height
    clear = robot.world_state(others, margin)
    carried = robot.world_state(others, margin, carried=(job.obj, np.linalg.inv(job.T_grasp) @ job.obj.pose))

    await robot.gripper.open()
    await robot.move(gripper, job.above(job.T_grasp, height), robot.world_state([job.obj, *others], margin))
    await robot.move(gripper, job.T_grasp, clear, linear=True)
    await robot.gripper.grab()
    if not await robot.gripper.is_holding_something():
        await robot.gripper.open()
        await robot.move(gripper, job.above(job.T_grasp, height), clear, linear=True)
        raise RuntimeError(f"Gripper did not hold object {job.obj.id}")
    await robot.move(gripper, job.above(job.T_grasp, height), carried, linear=True)
    await robot.move(gripper, job.above(job.T_place, height), carried)
    await robot.move(gripper, job.T_place, clear, linear=True)
    await robot.gripper.open()
    await robot.move(gripper, job.above(job.T_place, height), clear, linear=True)
