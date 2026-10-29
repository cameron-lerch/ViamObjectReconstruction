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
