''' Starting a run: step to the ready pose one joint group at a time, each step confirmed. '''

import asyncio

import numpy as np
import pytest

from recon.config import Config
from recon.robot import MoveDeclined, Robot


class FakeArm:
    def __init__(self, joints):
        self.joints = list(joints)
        self.moves = []

    async def get_joint_positions(self):
        return type("Joints", (), {"values": list(self.joints)})()

    async def get_end_position(self):
        from viam.proto.common import Pose
        return Pose(x=300, y=-400, z=400, o_z=-1)

    async def move_to_joint_positions(self, positions):
        self.moves.append([round(v) for v in positions.values])
        self.joints = list(positions.values)


def robot_at(joints):
    robot = Robot.__new__(Robot)
    robot.config, robot.live, robot.arm = Config(), None, FakeArm(joints)

    async def nothing():
        return None
    robot.wait_until_still = nothing
    robot.checked_camera_pose = nothing
    return robot


def test_steps_forearm_then_wrist_then_shoulder_elbow_then_base(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    robot = robot_at([0, 0, -121, 290, -103, 42])  # the free-moved pose of 2026-10-29
    asyncio.run(robot.go_ready())
    assert robot.arm.moves == [[0, 0, -121, 0, -103, 42], [0, 0, -121, 0, -40, -39],
                               [0, -37, -76, 0, -40, -39], [-41, -37, -76, 0, -40, -39]]


def test_nothing_moves_when_already_ready_and_n_stops(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    robot = robot_at([-40, -38, -77, 1, -40, -38])
    asyncio.run(robot.go_ready())
    assert robot.arm.moves == []

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    robot = robot_at([0, 0, -121, 290, -103, 42])
    with pytest.raises(MoveDeclined):
        asyncio.run(robot.go_ready())
    assert robot.arm.moves == []
