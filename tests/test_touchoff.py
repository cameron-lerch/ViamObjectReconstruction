"""Touch-off geometry: the gripper points straight down, the model's reach, and safe contact spots."""

import asyncio

import numpy as np
from viam.proto.common import Geometry, Pose, RectangularPrism, Vector3

from recon.config import Config
from recon.robot import Robot
from recon.scene import DetectedObject, TablePlane
from recon.touchoff import contact_spots, modeled_reach, pointing_down


def test_pointing_down_is_a_proper_rotation_with_the_tool_axis_down():
    for roll in (0.0, 37.0, 180.0):
        T = pointing_down(400, -300, 80, roll)
        np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(T[:3, :3]), 1.0)
        np.testing.assert_allclose(T[:3, 3], [400, -300, 80])


def test_modeled_reach_is_the_farthest_point_along_the_tool_axis():
    class FakeGripper:
        async def get_geometries(self):
            # The armfarm14 gripper's model: case from -100 to 0, claws from -55 to +50.
            return [Geometry(center=Pose(z=-50, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=50, y=100, z=100))),
                    Geometry(center=Pose(z=-2.5, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=40, y=170, z=105)))]
    robot = Robot.__new__(Robot)
    robot.gripper = FakeGripper()
    assert asyncio.run(modeled_reach(robot)) == 50.0


def test_contact_spots_stay_on_bare_table_and_spread_out():
    config = Config()
    # armfarm14's search region and base exclusion.
    config.workspace.x_range, config.workspace.y_range, config.workspace.min_base_distance = [-350, 700], [-650, 350], 350
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), 12.0)  # table at z = -12
    objects = [DetectedObject(0, np.array([300.0, -450.0, 18.0]), np.array([30.0, 30.0, 60.0]), 0.0, 100),
               DetectedObject(1, np.array([220.0, -320.0, 18.0]), np.array([30.0, 30.0, 60.0]), 0.0, 100)]
    board = np.array([420.0, -330.0, -9.0])
    spots = contact_spots(objects, plane, config, board)
    assert len(spots) == 3
    for spot in spots:
        assert spot[2] == -12.0
        assert all(np.linalg.norm(spot[:2] - o.center[:2]) >= o.footprint_radius + 80 for o in objects)
        assert np.linalg.norm(spot[:2] - board[:2]) >= 150
        assert np.linalg.norm(spot[:2]) >= config.workspace.min_base_distance + 50
    gaps = [np.linalg.norm(a[:2] - b[:2]) for i, a in enumerate(spots) for b in spots[i + 1:]]
    assert min(gaps) > 100
