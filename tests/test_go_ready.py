''' Starting a run: one planned move to the ready pose, no questions asked. '''

import asyncio

import numpy as np

from recon.config import Config
from recon.robot import Robot
from recon.transforms import look_at


def robot_with_camera_at(T_camera):
    robot = Robot.__new__(Robot)
    robot.config, robot.live, robot.moves = Config(), None, []

    async def camera_pose():
        return T_camera

    async def plan_best_roll(frame, pose_for_roll, world_state=None):
        return 0.0, pose_for_roll(0.0), [0.0] * 6

    async def move(frame, pose, world_state=None, linear=False):
        robot.moves.append((frame, pose))
    robot.checked_camera_pose, robot.plan_best_roll, robot.move = camera_pose, plan_best_roll, move
    return robot


def test_moves_to_the_ready_pose_looking_down(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(AssertionError("asked the operator")))
    robot = robot_with_camera_at(look_at([300, -200, 500], [400, -300, 0]))
    asyncio.run(robot.go_ready())
    (frame, pose), = robot.moves
    assert frame == robot.config.machine.camera
    np.testing.assert_allclose(pose[:3, 3], robot.config.machine.ready_camera_mm)
    np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=1e-9)


def test_nothing_moves_when_already_ready_or_no_ready_pose():
    ready = Config().machine.ready_camera_mm
    robot = robot_with_camera_at(look_at(np.add(ready, [3, -4, 2]), np.add(ready, [3, -4, -98])))
    asyncio.run(robot.go_ready())
    assert robot.moves == []

    robot = robot_with_camera_at(look_at([300, -200, 500], [400, -300, 0]))
    robot.config.machine.ready_camera_mm = []
    asyncio.run(robot.go_ready())
    assert robot.moves == []
