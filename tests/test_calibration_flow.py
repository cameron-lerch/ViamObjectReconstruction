''' calibration.py: the camera, table and gripper steps run in order, with the right inputs. '''

import asyncio

import calibration


def run(monkeypatch, execute=True, touch_off=None, answer=False, board_fails=False):
    calls = []

    async def board_step(robot, config, execute, save_mount):
        calls.append(("board", execute, save_mount))
        if board_fails:
            raise RuntimeError("The board isn't in view")
        return {"board_mount": []} if execute else None

    async def survey_step(robot, config, execute):
        calls.append(("survey", execute))
        return "the new survey", ["object"], None

    async def touch_off_step(robot, config, execute, session):
        calls.append(("touch-off", execute, session))

    monkeypatch.setattr(calibration, "board_step", board_step)
    monkeypatch.setattr(calibration, "survey_step", survey_step)
    monkeypatch.setattr(calibration, "touch_off_step", touch_off_step)
    monkeypatch.setattr(calibration, "ask", lambda question: answer)
    asyncio.run(calibration.calibrate_scene(None, None, execute, touch_off))
    return calls


def test_runs_camera_then_table_then_touch_off_on_that_survey(monkeypatch):
    calls = run(monkeypatch, answer=True)
    assert calls == [("board", True, True), ("survey", True), ("touch-off", True, "the new survey")]


def test_touch_off_is_skipped_when_declined(monkeypatch):
    assert [c[0] for c in run(monkeypatch, answer=False)] == ["board", "survey"]
    assert [c[0] for c in run(monkeypatch, touch_off=False, answer=True)] == ["board", "survey"]


def test_plan_only_plans_every_step_and_saves_nothing(monkeypatch):
    calls = run(monkeypatch, execute=False)
    assert calls == [("board", False, False), ("survey", False), ("touch-off", False, "the new survey")]


def test_stops_when_the_board_isnt_found(monkeypatch):
    assert run(monkeypatch, board_fails=True) == [("board", True, True)]
