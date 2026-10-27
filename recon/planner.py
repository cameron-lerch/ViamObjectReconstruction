"""Turn candidate viewpoints into a reachable scan plan for one object."""

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .robot import Robot
from .scene import DetectedObject
from .views import ViewCandidate, aim_point, coverage, generate_candidates, order_for_travel


@dataclass
class PlannedView:
    candidate: ViewCandidate
    roll_deg: float
    T_world_cam: np.ndarray
    joints_deg: np.ndarray  # planned end joints, from planning the view on its own


@dataclass
class ScanPlan:
    obj: DetectedObject
    views: list[PlannedView] = field(default_factory=list)
    unreachable: list[tuple[ViewCandidate, str]] = field(default_factory=list)
    coverage: float = 0.0

    def summary(self) -> str:
        total = len(self.views) + len(self.unreachable)
        return f"{len(self.views)}/{total} views reachable, coverage {self.coverage:.0%}"

    def to_dict(self) -> dict:
        return {
            "object": self.obj.to_dict(),
            "coverage": self.coverage,
            "views": [{"azimuth_deg": v.candidate.azimuth_deg, "elevation_deg": v.candidate.elevation_deg,
                       "roll_deg": v.roll_deg, "joints_deg": v.joints_deg.tolist(),
                       "T_world_cam": v.T_world_cam.tolist()} for v in self.views],
            "unreachable": [{"azimuth_deg": c.azimuth_deg, "elevation_deg": c.elevation_deg, "reason": reason}
                            for c, reason in self.unreachable],
        }


async def plan_scan(robot: Robot, obj: DetectedObject, others: list[DetectedObject], config: Config) -> ScanPlan:
    """Check every candidate view with the motion planner, in travel order.

    Each view uses the camera roll whose joints are closest to the previous
    view's, so the wrist does not wind up over a ring. The scanned object and
    all other detected objects are obstacles, so plans never put the gripper
    or camera through them.
    """
    world_state = robot.world_state([obj, *others], config.views.object_margin)
    plan = ScanPlan(obj)
    previous = None
    for candidate in order_for_travel(generate_candidates(obj, config.views)):
        roll, pose, result = await robot.plan_best_roll(config.machine.camera, candidate.pose, world_state, previous)
        if pose is None:
            plan.unreachable.append((candidate, result))
            continue
        plan.views.append(PlannedView(candidate, roll, pose, result))
        previous = result
    plan.coverage = coverage(aim_point(obj, config.views), [v.candidate.eye for v in plan.views], config.views)
    return plan
