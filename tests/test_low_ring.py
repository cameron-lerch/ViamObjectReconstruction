''' The lowest ring (below views.confirm_below_deg) waits for the operator once per object. '''

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from recon import pipeline
from recon.config import Config
from tests.test_resume import OBJ, PLANE, FakeRobot, planned, session_with_views


def scan(monkeypatch, tmp_path, answer):
    captured, asked = [], []

    async def plan_object(robot, obj, others, config):
        return SimpleNamespace(views=[planned(0, 90), planned(0, 50), planned(0, 10), planned(30, 10)]), None, None

    async def capture_view(robot, folder, extra=None, frames=None, depth=True, keep_frames=False):
        captured.append(extra["elevation_deg"])
        return folder

    monkeypatch.setattr(pipeline, "plan_object", plan_object)
    monkeypatch.setattr(pipeline, "save_plan", lambda *args: None)
    monkeypatch.setattr(pipeline, "capture_view", capture_view)
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or answer)
    session = session_with_views(tmp_path, [])
    asyncio.run(pipeline.scan_object(FakeRobot(), session, OBJ, [], PLANE, Config(), True, False))
    return captured, asked


@pytest.mark.parametrize("answer, expected", [("", [90, 50, 10, 10]), ("n", [90, 50])])
def test_asks_once_before_the_low_ring(monkeypatch, tmp_path, answer, expected):
    captured, asked = scan(monkeypatch, tmp_path, answer)
    assert captured == expected
    assert len(asked) == 1 and "2 low view(s) at 10deg" in asked[0]
