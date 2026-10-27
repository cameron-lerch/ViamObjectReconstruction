"""The pose guard must stop everything when the frame system and the arm controller disagree."""

import asyncio

import numpy as np
import pytest

from recon.config import Config
from recon.robot import PoseMismatch, Robot
from recon.transforms import look_at, matrix_to_pose, rotation_z


class FakeArm:
    def __init__(self, end_pose):
        self.end_pose = end_pose

    async def get_end_position(self):
        return matrix_to_pose(self.end_pose)

    async def get_joint_positions(self):
        class Joints:
            values = [0.0] * 6
        return Joints()


def guarded_robot(frame_system_cam: np.ndarray, controller_end: np.ndarray, mount: np.ndarray) -> Robot:
    robot = Robot.__new__(Robot)
    robot.config = Config()
    robot.arm = FakeArm(controller_end)
    robot._camera_mount = mount

    async def frame_pose(frame):
        return frame_system_cam

    robot.frame_pose = frame_pose
    return robot


MOUNT = np.eye(4)
MOUNT[:3, 3] = [83, -14, 18]


def test_agreeing_poses_pass():
    end = look_at([150, -400, 500], [150, -400, 0])
    camera = end @ MOUNT
    assert np.allclose(asyncio.run(guarded_robot(camera, end, MOUNT).checked_camera_pose()), camera)


def test_camera_rolled_180_degrees_is_caught():
    end = look_at([150, -400, 500], [150, -400, 0])
    reported = end @ MOUNT
    reported[:3, :3] = reported[:3, :3] @ rotation_z(180)  # same position, image upside down
    with pytest.raises(PoseMismatch, match="disagree"):
        asyncio.run(guarded_robot(reported, end, MOUNT).checked_camera_pose())
