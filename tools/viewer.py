''' Replay a finished scan in the live viewer, view by view, ending on its final mesh.

    python tools/viewer.py data/20261029-131913              # object 0
    python tools/viewer.py data/20261029-131913 --object 1 --delay 0.2

Open the printed link in a browser. The buttons are off (nothing is running); Ctrl+C quits.
'''

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon.config import load_config  # noqa: E402
from recon.live import Live  # noqa: E402
from recon.pipeline import scanned_object  # noqa: E402
from recon.session import Session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session")
    parser.add_argument("--object", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between views")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(ROOT / "config.toml")
    config.live.monitor_camera = ""
    session = Session(Path(args.session))
    _, plane = session.load_scene()
    obj = scanned_object(session, args.object)
    folders = sorted(path.parent for path in session.views_dir(args.object).glob("*/view.json"))

    live = Live(config, config.resolve(config.data_dir) / "live_replay", read_only=True)
    print("Open", live.start_server())
    live.start_object(session.root.name, obj, plane, len(folders))
    for index, folder in enumerate(folders):
        live.captured(folder)
        live.view_captured(folder, index)
        time.sleep(args.delay)
    live.message("Replaying: every view fused (arm poses, no final alignment)", "done")
    final = session.recon_dir(args.object) / "mesh.ply"
    if final.exists():
        live._fusion.submit(lambda: None).result()  # let the last view finish first
        shutil.copyfile(final, live.folder / "mesh.ply")
        live._bump("mesh")
        live.message("The final closed mesh (recon/mesh.ply)", "done")
    print("Showing the final mesh. Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        live.close()


if __name__ == "__main__":
    main()
