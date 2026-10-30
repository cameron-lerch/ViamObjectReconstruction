''' Watch a saved scan in the live viewer, as it looked while it ran.

    python data_viewer.py                       # the most recent scan
    python data_viewer.py 20261030-100851       # a session folder in data/
    python data_viewer.py 20261030-100851 --object 1 --seconds 2

Open the printed link. The page is the same one a scan shows (mesh, point cloud, the
RGB-D camera's color, depth, height and map), stepping through the scan one view about
every 3 seconds. Every view is prepared up front, so each panel's playback can go
anywhere, end included, and the final mesh is there from the start. Nothing talks to a
robot; the buttons that would move one are hidden. Press Enter to close.
'''

import argparse
import logging
import sys
from pathlib import Path

from recon.config import load_config
from recon.live import Live
from recon.pipeline import scanned_object
from recon.session import Session

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def latest_scan() -> Path:
    ''' The most recent session with captured scan views. '''
    # data/<session>/object_NN/views/view_NNN/view.json; real scans are named by date and time
    # (demo-* sessions are synthetic).
    scans = {path.parents[3] for path in DATA.glob("*/object_*/views/view_*/view.json") if path.parents[3].name[:1].isdigit()}
    if not scans:
        raise SystemExit(f"No scans with views in {DATA}")
    return max(scans, key=lambda path: path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="?", help="a session folder in data/ (default: the most recent scan)")
    parser.add_argument("--object", type=int, help="object id (default: the first one with views)")
    parser.add_argument("--seconds", type=float, default=3.0, help="seconds per view (default 3)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("viam").setLevel(logging.WARNING)

    root = Path(args.session) if args.session and Path(args.session).is_dir() else DATA / args.session if args.session else latest_scan()
    if not (root / "session.json").exists():
        raise SystemExit(f"{root} is not a scan session (no session.json)")
    session = Session(root)
    _, plane = session.load_scene()
    with_views = sorted(int(p.name.split("_")[1]) for p in root.glob("object_*") if any(p.glob("views/view_*/view.json")))
    if not with_views:
        raise SystemExit(f"{root} has no captured scan views")
    object_id = args.object if args.object is not None else with_views[0]
    obj = scanned_object(session, object_id)
    folders = sorted(path.parent for path in session.views_dir(object_id).glob("view_*/view.json"))

    config = load_config(ROOT / "config.toml")
    live = Live(config, DATA / "live_replay", read_only=True, replay=True)
    print(f"Saved scan {root.name}, object {object_id}: {len(folders)} views. Open", live.start_server())
    if (root / "survey_map.png").exists():
        live.show_map(root / "survey_map.png")
    live.start_object(root.name, obj, plane, len(folders))
    live.update(state="replay")
    for index, folder in enumerate(folders):
        live.captured(folder, plane)
        live.view_captured(folder, index)
    final = session.recon_dir(object_id) / "mesh.ply"
    if final.exists():
        live.show_final(final)
    live.message("a saved scan" + ("" if final.exists() else "; no final mesh was built for it"))
    live.tick(args.seconds)
    try:
        input("Preparing every view in the background; the page steps through them as they're ready.\n"
              "Press ENTER to close the viewer.\n")
    except (KeyboardInterrupt, EOFError):
        pass
    live.close()


if __name__ == "__main__":
    sys.exit(main())
