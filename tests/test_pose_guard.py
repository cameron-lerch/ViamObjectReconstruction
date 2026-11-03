''' The pose guard must stop everything when the frame system and the arm controller disagree. '''

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


def guarded_robot(frame_system_end: np.ndarray, controller_end: np.ndarray, mount: np.ndarray) -> Robot:
    robot = Robot.__new__(Robot)
    robot.config = Config()
    robot.arm = FakeArm(controller_end)
    robot._camera_mount = mount

    async def frame_pose(frame):
        assert frame == robot.config.machine.arm
        return frame_system_end

    robot.frame_pose = frame_pose
    return robot


MOUNT = np.eye(4)
MOUNT[:3, 3] = [83, -14, 18]


def test_agreeing_poses_pass_and_give_the_camera_through_our_mount():
    end = look_at([150, -400, 500], [150, -400, 0])
    assert np.allclose(asyncio.run(guarded_robot(end, end, MOUNT).checked_camera_pose()), end @ MOUNT)


def test_arm_end_rolled_180_degrees_is_caught():
    end = look_at([150, -400, 500], [150, -400, 0])
    reported = end.copy()
    reported[:3, :3] = reported[:3, :3] @ rotation_z(180)  # same position, wrist turned half a turn
    with pytest.raises(PoseMismatch, match="disagree"):
        asyncio.run(guarded_robot(reported, end, MOUNT).checked_camera_pose())


def moving_robot(*arrivals: np.ndarray) -> Robot:
    ''' A robot whose motion service always reports success, and leaves the arm's end at each
    of `arrivals` in turn (the last one repeats). '''
    arrivals = list(arrivals)
    robot = guarded_robot(np.eye(4), np.eye(4), MOUNT)
    robot.live, robot.confirm_moves = None, False
    robot._constraints = lambda linear: None

    class Motion:
        async def move(self, **kwargs):
            arrives_at = arrivals.pop(0) if len(arrivals) > 1 else arrivals[0]
            robot.arm.end_pose = arrives_at

            async def frame_pose(frame):
                return arrives_at
            robot.frame_pose = frame_pose
            return True
    robot.motion = Motion()

    async def still():
        return None
    robot.wait_until_still = still
    return robot


def test_a_move_that_ends_away_from_its_target_is_tried_again_then_caught():
    # Guards the bowl scan of 2026-11-03: the motion service reported success, but the camera
    # ended up facing away from the object, and that view was captured and used. Now a miss is
    # tried again once, and only a second miss fails the move (so the view is skipped).
    from recon.robot import MissedTarget
    target = look_at([400, -300, 300], [400, -300, 0])
    end = target @ np.linalg.inv(MOUNT)  # where the arm's end must go for the camera to be there
    elsewhere = end.copy()
    elsewhere[:3, 3] += [180, 230, 20]
    camera = Config().machine.camera
    asyncio.run(moving_robot(end).move(camera, target, world_state=object()))  # reached: no error
    asyncio.run(moving_robot(elsewhere, end).move(camera, target, world_state=object()))  # reached on the retry
    with pytest.raises(MissedTarget):
        asyncio.run(moving_robot(elsewhere, elsewhere).move(camera, target, world_state=object()))


def test_a_call_that_never_answers_is_asked_again_then_ends_the_run():
    # Guards the bowl scan of 2026-11-03: viam-server restarted mid-call, and the scan waited
    # for an answer for half an hour. Now a call with no answer is asked once more, and a
    # second miss raises RobotUnresponsive (scan.py then stops and prints the --resume command).
    from recon.robot import RobotUnresponsive
    robot = guarded_robot(np.eye(4), np.eye(4), MOUNT)
    calls = []

    async def answers_second_time():
        calls.append(1)
        if len(calls) == 1:
            await asyncio.sleep(3600)  # the lost answer
        return "joints"

    async def never_answers():
        await asyncio.sleep(3600)

    assert asyncio.run(robot.ask("the arm's joints", answers_second_time, timeout=0.05)) == "joints"
    with pytest.raises(RobotUnresponsive):
        asyncio.run(robot.ask("a frame", never_answers, timeout=0.05))
