''' Views that would bring the camera too close to the table are dropped before planning. '''

import asyncio

import numpy as np

from recon.config import Config
from recon.planner import plan_scan
from recon.scene import DetectedObject


class FakeRobot:
    def __init__(self):
        self.asked = []

    def world_state(self, objects, margin):
        return None

    async def plan_best_roll(self, camera, pose_for_roll, world_state, previous):
        pose = pose_for_roll(0.0)
        self.asked.append(pose)
        return 0.0, pose, np.zeros(6)


def test_low_ring_is_dropped_for_a_small_object_but_not_for_a_tall_one():
    config = Config()
    config.views.elevations_deg = [5.0, 25.0]
    small = DetectedObject(0, np.array([300.0, -400.0, 12.5]), np.array([30.0, 30.0, 25.0]), 0.0, 100)
    robot = FakeRobot()
    plan = asyncio.run(plan_scan(robot, small, [], config))
    low = [c for c, reason in plan.unreachable if "above the table" in reason]
    assert len(low) == 12 and all(c.elevation_deg == 5.0 for c in low)
    assert all(pose[2, 3] - small.bottom_z >= config.views.min_camera_height for pose in robot.asked)
    assert all(v.candidate.eye[2] >= config.views.min_camera_height for v in plan.views)

    tall = DetectedObject(1, np.array([300.0, -400.0, 100.0]), np.array([150.0, 150.0, 200.0]), 0.0, 100)
    plan = asyncio.run(plan_scan(FakeRobot(), tall, [], config))
    assert not [c for c, reason in plan.unreachable if "above the table" in reason]
