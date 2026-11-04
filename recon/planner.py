''' Turn candidate viewpoints into a reachable scan plan for one object. '''

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .robot import Robot
from .scene import DetectedObject
from .views import ViewCandidate, aim_point, coverage, generate_candidates, order_ring

log = logging.getLogger(__name__)


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
    rotations: dict[float, float] = field(default_factory=dict)  # rings turned to get around the arm's base

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
            "ring_rotations_deg": {str(elevation): turn for elevation, turn in self.rotations.items()},
        }


def blocked_by_base(reason: str) -> bool:
    ''' A planner failure that names the arm's base (e.g. "collision: arm:link4 and arm:link_base"). '''
    return reason.startswith("collision") and "base" in reason


async def plan_scan(robot: Robot, obj: DetectedObject, others: list[DetectedObject], config: Config,
                    rotations: dict[float, float] | None = None) -> ScanPlan:
    ''' Check every candidate view with the motion planner, ring by ring in travel order.

    Each view uses the camera roll whose joints are closest to the previous
    view's, so the wrist does not wind up over a ring. The scanned object and
    all other detected objects are obstacles, so plans never put the gripper
    or camera through them.

    When the arm's base blocks a view on a ring, the ring is planned again turned by half
    a step, and kept turned if that reaches more of its views. Views that fail for other
    reasons (out of reach, say) don't turn a ring. `rotations` (from a saved plan) replays
    a layout instead, so a resumed scan plans the views it started with.
    '''
    world_state = robot.world_state([obj, *others], config.views.object_margin)
    plan = ScanPlan(obj, rotations=dict(rotations or {}))
    half_step = 180 / max(3, round(360 / config.views.azimuth_step_deg))
    candidates = generate_candidates(obj, config.views, plan.rotations)
    previous, last_eye = None, None
    for index, elevation in enumerate(sorted({c.elevation_deg for c in candidates}, reverse=True)):
        ring = order_ring([c for c in candidates if c.elevation_deg == elevation], last_eye, index % 2 == 0)
        views, missed, end = await _plan_ring(robot, ring, obj, world_state, previous, config)
        if rotations is None and len(ring) > 1 and any(blocked_by_base(reason) for _, reason in missed):
            turned = {**plan.rotations, elevation: half_step}
            other = order_ring([c for c in generate_candidates(obj, config.views, turned) if c.elevation_deg == elevation],
                               last_eye, index % 2 == 0)
            other_views, other_missed, other_end = await _plan_ring(robot, other, obj, world_state, previous, config)
            if len(other_views) > len(views):
                log.info("  %.0f degree ring was rotated by %.0f degrees to avoid the arm's base and reach more views "
                         "(%d of %d instead of %d)", elevation, half_step, len(other_views), len(other), len(views))
                ring, views, missed, end = other, other_views, other_missed, other_end
                plan.rotations[elevation] = half_step
        plan.views.extend(views)
        plan.unreachable.extend(missed)
        previous, last_eye = end, ring[-1].eye
    plan.coverage = coverage(aim_point(obj, config.views), [v.candidate.eye for v in plan.views], config.views)
    return plan


async def _plan_ring(robot: Robot, ring: list[ViewCandidate], obj: DetectedObject, world_state, previous,
                     config: Config) -> tuple[list[PlannedView], list[tuple[ViewCandidate, str]], np.ndarray | None]:
    ''' Plan one ring's views in order: (reachable views, unreachable views with reasons,
    the last reachable view's end joints, or `previous` if none). '''
    views, missed = [], []
    for candidate in ring:
        if candidate.eye[2] - obj.bottom_z < config.views.min_camera_height:
            missed.append((candidate, f"camera would be under {config.views.min_camera_height:.0f}mm above the table"))
            continue
        roll, pose, result = await robot.plan_best_roll(config.machine.camera, candidate.pose, world_state, previous)
        if pose is None:
            missed.append((candidate, result))
            continue
        views.append(PlannedView(candidate, roll, pose, result))
        previous = result
    return views, missed, previous


def spread_views(views: list[PlannedView], count: int | None) -> list[PlannedView]:
    ''' `count` of the views spread as evenly as possible over the viewing directions:
    the most overhead first, then each next one farthest (in direction) from those
    already chosen. They keep their planned (travel) order. All of them without a count.

    On the real block and cylinder, 16 of 34 views spread like this already measured
    every side and the top; more views mostly reduce noise (README section 3.1).
    '''
    if count is None or count >= len(views):
        return list(views)
    directions = np.array([(v.candidate.eye - v.candidate.target) / np.linalg.norm(v.candidate.eye - v.candidate.target)
                           for v in views])
    chosen = [int(np.argmax(directions[:, 2]))]
    while len(chosen) < count:
        nearest = np.max(directions @ directions[chosen].T, axis=1)  # cosine to the closest chosen view
        nearest[chosen] = np.inf
        chosen.append(int(np.argmin(nearest)))
    return [views[k] for k in sorted(chosen)]

