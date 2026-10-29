''' --resume: finish an interrupted scan, skipping views already captured, after checking
the object hasn't moved. '''

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from recon import pipeline
from recon.config import Config
from recon.scene import DetectedObject, TablePlane
from recon.session import Session

OBJ = DetectedObject(0, np.array([300.0, -400.0, 30.0]), np.array([40.0, 40.0, 60.0]), 0.0, 100)
PLANE = TablePlane(np.array([0.0, 0.0, 1.0]), 0.0)


def planned(azimuth, elevation):
    candidate = SimpleNamespace(azimuth_deg=azimuth, elevation_deg=elevation, describe=lambda: f"az {azimuth} el {elevation}")
    return SimpleNamespace(candidate=candidate, roll_deg=0.0, T_world_cam=np.eye(4))


class FakeRobot:
    live = None

    def __init__(self):
        self.moves = 0

    def world_state(self, *args, **kwargs):
        return None

    async def move(self, *args, **kwargs):
        self.moves += 1


def session_with_views(tmp_path, done, frames=50):
    (tmp_path / "session").mkdir()
    session = Session(tmp_path / "session")
    session.save_scene([OBJ], PLANE)
    session.write_json(session.object_dir(0) / "scanned_object.json", OBJ.to_dict())
    for index, (azimuth, elevation) in done:
        session.write_json(session.views_dir(0) / f"view_{index:03d}" / "view.json",
                           {"azimuth_deg": azimuth, "elevation_deg": elevation, "frames": frames, "T_world_cam": np.eye(4).tolist()})
    return session


def run_resume(monkeypatch, session, config):
    captured = []

    async def plan_scan(robot, target, others, config):
        return SimpleNamespace(views=[planned(0, 90), planned(0, 72), planned(30, 72), planned(60, 72)])

    async def capture_view(robot, folder, extra=None, frames=None, depth=True, keep_frames=False):
        captured.append((folder.name, extra["azimuth_deg"], extra["elevation_deg"]))
        return folder

    monkeypatch.setattr(pipeline, "plan_scan", plan_scan)
    monkeypatch.setattr(pipeline, "capture_view", capture_view)
    asyncio.run(pipeline.scan_object(FakeRobot(), session, OBJ, [], PLANE, config, True, False, resume=True))
    return captured


def test_resume_skips_captured_views_and_numbers_after_them(monkeypatch, tmp_path):
    config = Config()
    config.capture.frames_per_view = 50
    session = session_with_views(tmp_path, [(0, (0, 90)), (1, (0, 72))])
    assert run_resume(monkeypatch, session, config) == [("view_002", 30, 72), ("view_003", 60, 72)]


def test_resume_refuses_different_frame_settings(monkeypatch, tmp_path):
    config = Config()  # 5 frames per view, but the first part was a full scan (50)
    session = session_with_views(tmp_path, [(0, (0, 90))])
    with pytest.raises(RuntimeError, match="same options"):
        run_resume(monkeypatch, session, config)


@pytest.mark.parametrize("shift, moved", [(1.0, False), (5.0, True)])
def test_check_refuses_an_object_that_moved(monkeypatch, tmp_path, shift, moved):
    session = session_with_views(tmp_path, [(0, (0, 90)), (1, (0, 72))])
    positions = iter([(np.array([300.0, -400.0]), 60.0), (np.array([300.0 + shift, -400.0]), 60.0)])

    async def capture_view(robot, folder, extra=None, frames=None, depth=True, keep_frames=False):
        return folder

    monkeypatch.setattr(pipeline, "capture_view", capture_view)
    monkeypatch.setattr(pipeline, "load_view", lambda folder: folder)
    monkeypatch.setattr(pipeline, "object_position", lambda view, obj, plane, config: next(positions))
    check = pipeline.check_unmoved(FakeRobot(), session, OBJ, [], PLANE, Config())
    if moved:
        with pytest.raises(pipeline.ObjectMoved):
            asyncio.run(check)
    else:
        asyncio.run(check)
