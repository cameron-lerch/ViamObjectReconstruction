''' scan.py without a command: survey, ask about each object, scan, park, then build meshes. '''

import asyncio

import numpy as np

import scan
from recon.config import Config
from recon.scene import DetectedObject


def block(k: int) -> DetectedObject:
    return DetectedObject(k, np.array([300.0 + 60 * k, -400.0, 18.0]), np.array([30.0, 30.0, 60.0]), 0.0, 100, (0.9, 0.4, 0.1))


class FakeSession:
    root = "data/session"


def run(monkeypatch, answers, objects=3, execute=True):
    calls, answers = [], list(answers)

    async def survey_step(robot, config, execute):
        calls.append("survey")
        return FakeSession(), [block(k) for k in range(objects)], None

    async def scan_object(robot, session, obj, others, plane, config, execute, step, max_views):
        calls.append(f"scan {obj.id}")

    async def plan_object(robot, obj, others, config):
        calls.append(f"plan {obj.id}")
        return None, None, None

    async def go_home(robot, session, config):
        calls.append("home")

    monkeypatch.setattr(scan, "survey_step", survey_step)
    monkeypatch.setattr(scan.pipeline, "scan_object", scan_object)
    monkeypatch.setattr(scan.pipeline, "plan_object", plan_object)
    monkeypatch.setattr(scan.pipeline, "save_plan", lambda *a: None)
    monkeypatch.setattr(scan.pipeline, "go_home", go_home)
    monkeypatch.setattr(scan.pipeline, "finish_object", lambda session, oid, plane, config: calls.append(f"mesh {oid}"))
    monkeypatch.setattr(scan, "compare_object", lambda *a, **k: None)
    monkeypatch.setattr(scan, "load_truth", lambda path: {})
    monkeypatch.setattr(scan, "ask_object", lambda obj: answers.pop(0))
    asyncio.run(scan.scan_everything(None, Config(), execute, None))
    return calls


def test_asks_before_each_object_and_parks_before_building_meshes(monkeypatch):
    calls = run(monkeypatch, ["y", "n", "y"])
    assert calls == ["survey", "scan 0", "scan 2", "home", "mesh 0", "mesh 2"]


def test_all_the_rest_stops_asking(monkeypatch):
    calls = run(monkeypatch, ["n", "a"], objects=4)  # no more answers needed after 'a'
    assert calls == ["survey", "scan 1", "scan 2", "scan 3", "home", "mesh 1", "mesh 2", "mesh 3"]


def test_quit_still_parks_and_finishes_what_was_scanned(monkeypatch):
    calls = run(monkeypatch, ["y", "q"])
    assert calls == ["survey", "scan 0", "home", "mesh 0"]


def test_plan_only_plans_every_object_and_moves_nothing(monkeypatch):
    assert run(monkeypatch, [], execute=False) == ["survey", "plan 0", "plan 1", "plan 2"]


def test_describe_for_person():
    assert scan.describe_for_person(block(1)) == "object 1: orange, 30 x 30 x 60mm, centered at (360, -400)"
