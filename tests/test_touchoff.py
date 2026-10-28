''' Touch-off geometry: the gripper points straight down, the model's reach, and safe contact spots. '''

import asyncio

import numpy as np
from viam.proto.common import Geometry, Pose, RectangularPrism, Vector3

from recon.config import Config
from recon.robot import Robot
from recon.scene import DetectedObject, TablePlane
from recon.touchoff import (CLEARANCE, TRAVEL, VICINITY, bare_table, candidate_spots, modeled_reach, pointing_down, spread_order,
                            step_size, table_comparison, travel_height)


def armfarm14() -> Config:
    config = Config()
    config.workspace.x_range, config.workspace.y_range, config.workspace.min_base_distance = [-350, 700], [-650, 350], 350
    return config


def block(x: float, y: float) -> DetectedObject:
    return DetectedObject(0, np.array([x, y, 18.0]), np.array([30.0, 30.0, 60.0]), 0.0, 100)


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


def test_spots_stay_clear_of_everything_on_the_table_and_spread_out():
    config = armfarm14()
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), 12.0)  # table at z = -12
    things = [block(300, -450), block(220, -320), block(450, -150)]  # targets and clutter alike
    board = np.array([420.0, -330.0, -9.0])
    spots = spread_order(candidate_spots(things, plane, config, board))
    assert len(spots) > 3
    for spot in spots:
        assert spot[2] == -12.0
        assert all(np.linalg.norm(spot[:2] - o.center[:2]) >= o.footprint_radius + CLEARANCE for o in things)
        assert np.linalg.norm(spot[:2] - board[:2]) >= 150
        assert np.linalg.norm(spot[:2]) >= config.workspace.base_clearance
    first = spots[:3]
    assert min(np.linalg.norm(a[:2] - b[:2]) for i, a in enumerate(first) for b in first[i + 1:]) > 300


def test_spots_only_where_the_survey_saw_bare_table():
    config = armfarm14()
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), 12.0)
    # The survey saw table only for y > -500 (beyond that: the table's edge).
    x, y = np.meshgrid(np.arange(-350, 700, 4.0), np.arange(-500, 350, 4.0))
    seen = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -12.0)])
    spots = candidate_spots([block(300, -400)], plane, config, on_table=bare_table(seen, plane))
    assert spots and all(s[1] >= -500 + 30 for s in spots)  # a 40mm disc, 95% of it seen


def test_travel_puts_the_fingertips_600mm_above_the_table():
    assert travel_height(-12.0, 76.0) == -12.0 + 76.0 + TRAVEL


def test_table_comparison_finds_a_tilt_but_not_a_constant_offset():
    flat = [{"spot": xy, "table_z": -12.0, "gripper_z": 64.0} for xy in ([400, -200], [100, -600], [500, -600])]
    assert table_comparison(flat)["tilt_deg"] < 1e-9
    # The arm's table rises 1mm per 100mm in x relative to the camera's.
    tilted = [dict(p, gripper_z=64.0 + 0.01 * p["spot"][0]) for p in flat]
    assert abs(table_comparison(tilted)["tilt_mm_per_100mm"] - 1.0) < 1e-9


def test_step_size():
    assert step_size("") == 1.0
    assert step_size("3") == 3.0
    assert step_size("20") == 5.0
    assert step_size("q") is None


def test_world_state_includes_clutter():
    robot = Robot.__new__(Robot)
    robot.config, robot.table_plane, robot.use_table_slab = Config(), None, True
    robot.clutter = [block(300, -450)]
    labels = [g.label for g in robot.world_state([], 10.0).obstacles[0].geometries]
    assert labels == ["clutter-0"]


def test_spots_stay_near_the_scan_targets():
    config = armfarm14()
    plane = TablePlane(np.array([0.0, 0.0, 1.0]), 12.0)
    targets = [block(300, -450), block(220, -320)]
    clutter = [block(550, 50)]  # a power strip far away is avoided, not approached
    spots = candidate_spots([*targets, *clutter], plane, config, targets=targets)
    assert spots
    for spot in spots:
        nearest = min(np.linalg.norm(spot[:2] - o.center[:2]) - o.footprint_radius for o in targets)
        assert CLEARANCE <= nearest <= CLEARANCE + VICINITY
