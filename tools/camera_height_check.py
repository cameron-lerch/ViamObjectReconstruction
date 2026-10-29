''' A ruler check of the camera mount: put the camera straight above the calibration
board at a known height, then measure it.

    python tools/camera_height_check.py               # plan only: where it would go
    python tools/camera_height_check.py --execute     # move there (asks first)
    python tools/camera_height_check.py --height 120 --execute

Our mount says where the lens (the depth camera's optical center) is. Looking straight
down, the lens should be `--height` above the board's surface, and the camera's front
face (its glass) a few millimeters lower. Measure from the board to the front face with
a ruler: within about 5mm of the printed prediction means the mount, and the collision
box placed by it, are where the real camera is.
'''

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon.config import load_config  # noqa: E402
from recon.pipeline import move_camera_looking_down  # noqa: E402
from recon.session import Session  # noqa: E402
from recon.transforms import look_at  # noqa: E402
from scan import with_robot  # noqa: E402

log = logging.getLogger("camera_height_check")
ROOT = Path(__file__).resolve().parent.parent


async def check(robot, config, height: float, execute: bool) -> None:
    session = Session.latest(config.resolve(config.data_dir), surveyed=True)
    _, plane = session.load_scene()
    robot.set_scene(plane, session.load_clutter())
    board = session.read().get("board")
    if not board:
        raise RuntimeError(f"No calibration board in {session.root}; run calibration.py first")
    x, y, board_z = board["center"]
    thickness = board_z - plane.z_at(x, y)
    mount = await robot.camera_mount()
    names = config.machine
    log.info("Board at (%.0f, %.0f), its surface %.1fmm above the table as measured (the board's thickness)", x, y, thickness)
    log.info("Our mount puts the lens (%.0f, %.0f, %.0f)mm from the arm's end", *mount[:3, 3])
    log.info("Target: the lens %.0fmm above the board's surface, looking straight down", height)
    log.info("  predicted: the camera's front face about %.0f-%.0fmm above the board (a RealSense's lens sits a few mm "
             "behind its glass)", height - 5, height)
    log.info("  our collision box reaches %.0fmm below the lens, to %.0fmm above the board",
             names.camera_body_offset_mm + names.camera_body_mm[2] / 2, height - (names.camera_body_offset_mm + names.camera_body_mm[2] / 2))
    if not execute:
        _, pose, reason = await robot.plan_best_roll(
            names.camera, lambda r: look_at([x, y, board_z + height], [x, y, board_z + height - 100], r))
        log.info("Plan only: %s. Add --execute to move there.", "reachable" if pose is not None else f"not reachable ({reason})")
        return
    if not await move_camera_looking_down(robot, x, y, board_z + height):
        return
    pose = await robot.checked_camera_pose()
    log.info("Done: the lens is %.1fmm above the board's surface by our mount (straight down within %.1fdeg). "
             "Measure from the board to the camera's front face and compare.",
             pose[2, 3] - board_z, np.degrees(np.arccos(min(1.0, -pose[2, 2]))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height", type=float, default=100.0, help="lens height above the board (mm, default 100)")
    parser.add_argument("--execute", action="store_true", help="move the arm (asks first)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("viam", "grpclib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config = load_config(ROOT / "config.toml")
    config.live.enabled = False
    if args.height < 60:
        sys.exit("Keep the lens at least 60mm above the board: the collision box and the table's 15mm safety slab need the room.")
    if args.execute and input(f"The arm will move the camera {args.height:.0f}mm above the calibration board. Start? [y/N] ").strip().lower() not in ("y", "yes"):
        return
    asyncio.run(with_robot(config, lambda robot: check(robot, config, args.height, args.execute), go_ready=args.execute))


if __name__ == "__main__":
    main()
