''' Starting a run: one planned move to the home position (found from the mat), no questions asked. '''

import asyncio
import json

import numpy as np

from recon.config import Config
from recon.robot import Robot
from recon.transforms import look_at


HOME = [450.0, 170.0, 320.0]


def robot_with_camera_at(T_camera, tmp_path, home=HOME):
    robot = Robot.__new__(Robot)
    robot.config, robot.live, robot.moves = Config(), None, []
    robot.config.machine.home_file = str(tmp_path / "home.json")
    if home is not None:
        (tmp_path / "home.json").write_text(json.dumps({"camera_mm": home}))

    async def camera_pose():
        return T_camera

    async def plan_best_roll(frame, pose_for_roll, world_state=None):
        return 0.0, pose_for_roll(0.0), [0.0] * 6

    async def move(frame, pose, world_state=None, linear=False):
        robot.moves.append((frame, pose))
    robot.checked_camera_pose, robot.plan_best_roll, robot.move = camera_pose, plan_best_roll, move
    return robot


def test_moves_to_the_home_position_looking_down(monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(AssertionError("asked the operator")))
    robot = robot_with_camera_at(look_at([300, -200, 500], [400, -300, 0]), tmp_path)
    asyncio.run(robot.go_ready())
    (frame, pose), = robot.moves
    assert frame == robot.config.machine.camera
    np.testing.assert_allclose(pose[:3, 3], HOME)
    np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=1e-9)


def test_nothing_moves_when_already_home_or_no_home_found_yet(tmp_path):
    robot = robot_with_camera_at(look_at(np.add(HOME, [3, -4, 2]), np.add(HOME, [3, -4, -98])), tmp_path)
    asyncio.run(robot.go_ready())
    assert robot.moves == []

    robot = robot_with_camera_at(look_at([300, -200, 500], [400, -300, 0]), tmp_path / "new", home=None)
    asyncio.run(robot.go_ready())
    assert robot.moves == []
