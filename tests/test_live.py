''' The live viewer's controls: Pause holds the next move, Stop ends the scan, read-only refuses both. '''

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from recon.config import Config
from recon.live import Live, StopRequested


def test_pause_holds_the_next_move_until_resumed_and_stop_ends_it(tmp_path):
    live = Live(Config(), tmp_path)

    async def scenario():
        live.pause()
        waiting = asyncio.create_task(live.checkpoint())
        await asyncio.sleep(0.5)
        assert not waiting.done() and live.status()["state"] == "paused"
        live.resume()
        await asyncio.wait_for(waiting, 2)
        live.stop()
        with pytest.raises(StopRequested):
            await live.checkpoint()

    asyncio.run(scenario())
    live.close()


def test_the_buttons_work_over_http_and_not_on_a_finished_scan(tmp_path):
    config = Config()
    config.live.port = 18765
    live = Live(config, tmp_path / "running")
    url = live.start_server()
    request = urllib.request.Request(f"{url}/api/pause", method="POST")
    assert json.loads(urllib.request.urlopen(request).read())["paused"] is True
    live.close()

    replay = Live(config, tmp_path / "replay", read_only=True)
    url = replay.start_server()
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(urllib.request.Request(f"{url}/api/stop", method="POST"))
    assert refused.value.code == 403
    replay.close()


def test_every_step_is_kept_and_a_live_scan_follows_the_newest(tmp_path):
    import sys
    sys.path.insert(0, "tests")
    from test_experiment import OBJ, saved_scan
    from recon.session import Session
    scan = saved_scan(tmp_path, frames=1)
    _, plane = Session(scan).load_scene()
    folders = sorted(p.parent for p in (scan / "object_00" / "views").glob("view_*/view.json"))[:3]
    live = Live(Config(), tmp_path / "live")
    live.start_object("scan", OBJ, plane, len(folders))
    for index, folder in enumerate(folders):
        live.captured(folder, plane)
        live.view_captured(folder, index)
    live.wait()
    status = live.status()
    assert status["history"] == 3 and status["current"] == 3 and len(status["steps"]) == 3
    for kind in ("mesh", "points", "all_points", "color", "depth", "height"):
        assert (tmp_path / "live" / f"{kind}_003.{'ply' if kind in ('mesh', 'points', 'all_points') else 'jpg'}").exists()
    live.complete("3 views, final mesh ready")
    assert live.status()["state"] == "complete"
    live.close()


def test_a_replay_prepares_every_step_and_ticks_through_them(tmp_path):
    import time
    live = Live(Config(), tmp_path, read_only=True, replay=True)
    live.update(history=3, views=3)
    live.tick(0.05)
    time.sleep(0.5)
    assert live.status()["current"] == 3
    assert live.status()["monitor"] is False
    live.close()


def test_the_monitoring_camera_streams_as_mjpeg(tmp_path):
    import threading
    config = Config()
    config.live.port = 18775
    live = Live(config, tmp_path)
    url = live.start_server()
    received = []

    def read():
        with urllib.request.urlopen(f"{url}/stream/monitor", timeout=5) as stream:
            assert stream.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            received.append(stream.read(200))

    reader = threading.Thread(target=read)
    reader.start()
    import time
    for k in range(3):  # three frames arrive
        with live._frame_ready:
            live._frame = b"\xff\xd8 frame %d " % k + b"x" * 100
            live._frame_ready.notify_all()
        time.sleep(0.1)
    reader.join(timeout=5)
    live.close()
    assert received and b"--frame" in received[0] and b"Content-Type: image/jpeg" in received[0]
