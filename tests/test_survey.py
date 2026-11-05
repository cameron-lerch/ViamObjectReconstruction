''' The survey finds the table from where the camera is and follows it to its edges (in
simulation: a synthetic table with nothing beyond its edges, and an arm with limited reach). '''

import asyncio
import io
import json

import numpy as np
import pytest
from PIL import Image

from recon import pipeline
from recon.config import Config
from recon.session import Session
from recon.synthetic import INTRINSICS, TABLE_Z, SyntheticScene, box, SceneObject
from recon.transforms import look_at

TABLE = (100.0, -500.0, 800.0, 700.0)  # x0, y0, length, width: x 100..900, y -500..200
REACH = 850.0  # how far from the base the simulated camera can go


class SimulatedArm:
    ''' Just enough of recon.robot.Robot for the survey: moves a camera around a synthetic scene. '''

    def __init__(self, scene: SyntheticScene, camera: np.ndarray):
        self.scene, self.T, self.moves = scene, camera, []
        self.config, self.live, self.last_frames, self.table_plane = Config(), None, None, None
        self.rng = np.random.default_rng(3)

    def world_state(self, objects, margin):
        return None

    async def checked_camera_pose(self):
        return self.T

    async def plan_best_roll(self, frame, pose_for_roll, world_state=None, reference_deg=None):
        pose = pose_for_roll(0.0)
        if np.hypot(*pose[:2, 3]) > REACH:
            return None, None, "zero IK solutions produced, goal positions appears to be physically unreachable"
        return 0.0, pose, np.zeros(6)

    async def move(self, frame, pose, world_state=None, linear=False):
        self.moves.append(pose)
        self.T = pose

    async def intrinsics(self):
        return INTRINSICS

    async def capture(self, frames=None, depth=True, keep=False):
        color, depth_mm = self.scene.render(self.T, self.rng)
        encoded = io.BytesIO()
        Image.fromarray(color).save(encoded, format="JPEG")
        return encoded.getvalue(), "jpg", depth_mm, self.T, {"frames": 1}


def survey(tmp_path, camera):
    scene = SyntheticScene([SceneObject(box(500, -150, (30, 30, 60), 10), (0.9, 0.4, 0.1))], table_box=TABLE)
    arm = SimulatedArm(scene, camera)
    session = Session(tmp_path / "session")
    folders = asyncio.run(pipeline.survey_table(arm, session, arm.config))
    return arm, folders


def test_starts_small_then_follows_the_table_to_its_edges(tmp_path):
    start = look_at([480, -120, TABLE_Z + 400], [520, -160, TABLE_Z])  # over the table, looking down at an angle
    arm, folders = survey(tmp_path, start)
    first, second = arm.moves[0], arm.moves[1] if len(arm.moves) > 1 else None
    # Look before moving: the first picture is from where the camera started (kept apart
    # from the survey views, which all look straight down, one per move).
    first_look = json.loads((tmp_path / "session" / "survey" / "first_look" / "view.json").read_text())
    np.testing.assert_allclose(first_look["T_world_cam"], start)
    assert len(folders) == len(arm.moves)
    # The first move only turns the camera straight down and rises over the same spot.
    np.testing.assert_allclose(first[:2, 3], start[:2, 3], atol=1e-6)
    assert first[2, 2] < -0.999 and first[2, 3] == pytest.approx(TABLE_Z + arm.config.workspace.survey_height, abs=5)
    # Every later move goes one lattice step to a neighbor of where the camera has been, over the table.
    x0, y0, length, width = TABLE
    step = arm.config.workspace.survey_spacing
    for pose in arm.moves[1:]:
        x, y = pose[:2, 3]
        assert x0 - step / 2 <= x <= x0 + length + step / 2 and y0 - step / 2 <= y <= y0 + width + step / 2
        assert np.hypot(x, y) <= REACH
    assert second is not None, "the survey never left its first position"
    # The table area it saw is the real table, edge to edge (the views see past the edges).
    clouds = []
    for folder in folders:
        points = pipeline.scene_points([folder], arm.config)
        plane = pipeline.fit_table_plane(points, arm.config.workspace)
        clouds.append((points, plane.height(points)))
    (seen_x0, seen_x1), (seen_y0, seen_y1) = pipeline.table_area(clouds)
    assert abs(seen_x0 - x0) < 30 and abs(seen_y0 - y0) < 30 and abs(seen_y1 - (y0 + width)) < 30
    assert seen_x1 <= x0 + length + 30  # the far edge only as far as the camera could reach to see it


@pytest.mark.parametrize("camera, why", [
    (look_at([480, -120, 400], [900, -120, 450]), "looking up, away from the table"),
    (look_at([1300, 600, 400], [1300, 600, 0]), "looking down, but past the table's edge"),
])
def test_wont_start_unless_the_camera_is_over_the_table(tmp_path, camera, why):
    with pytest.raises(pipeline.NeedsTable):
        survey(tmp_path, camera)
