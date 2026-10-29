''' Calibrate the scene in one go: the camera, the table, and (if you want) the gripper.

    python calibration.py              # asks once before the arm moves
    python calibration.py --plan-only  # plans every step, moves nothing

1. Camera: find the ChArUco board on the table, view it from 12 directions and
   solve for where the camera sits on the arm. The result becomes the camera
   mount for every move and capture.
2. Table: survey the table from above with the newly calibrated camera (the
   table's height, the objects on it, and the board).
3. Gripper (asks first; off while no gripper is mounted): the touch-off. You stand at the arm with a sheet of
   paper; the closed gripper steps down onto the table at 3 open spots near the
   objects until the paper drags, which measures how far the fingertips reach.
   Only needed for a new arm or gripper, or if something was changed or bumped.

Each step is also available on its own (see `python scan.py --help`), with
options for watching the arm move by move.
'''

import argparse
import asyncio
import logging

from recon.config import load_config
from scan import board_step, survey_step, touch_off_step, with_robot

log = logging.getLogger("calibration")


def ask(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


async def calibrate_scene(robot, config, execute: bool, touch_off: bool | None) -> None:
    ''' The three steps in order. `touch_off`: True or False to decide up front, None to ask. '''
    log.info("\n== 1/3 Camera: calibrating with the ChArUco board ==")
    try:
        board = await board_step(robot, config, execute, save_mount=execute)
    except RuntimeError as problem:
        log.error("The camera calibration stopped: %s. Is the ChArUco board flat on the table and uncovered?", problem)
        return
    if execute and board is None:
        log.error("The camera calibration didn't finish; stopping.")
        return

    log.info("\n== 2/3 Table: surveying with the calibrated camera ==")
    session, objects, _ = await survey_step(robot, config, execute)

    log.info("\n== 3/3 Gripper: touch-off ==")
    if not config.machine.gripper:
        log.info("Skipped: no gripper is mounted (the touch-off measures a gripper's fingertips).")
        touch_off = False
    elif touch_off is None:
        touch_off = execute and await asyncio.to_thread(
            ask, "Run the touch-off now? You need to be at the arm with a sheet of paper.")
    if config.machine.gripper and (touch_off or not execute):
        await touch_off_step(robot, config, execute, session)
    elif config.machine.gripper:
        log.info("Skipped; the last touch-off's result stays in use (calibration/*touch_off.json).")

    log.info("\nCalibration %s. %d object(s) on the table. Next: python scan.py",
             "done" if execute else "planned (nothing moved)", len(objects))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true", help="plan every step without moving the arm")
    parser.add_argument("--touch-off", choices=["ask", "yes", "no"], default="ask", help="run the touch-off (default: ask)")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("viam", "grpclib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config = load_config(args.config)

    execute = not args.plan_only
    if execute and not ask("The arm will move to calibrate the camera and survey the table "
                           "(about 5 minutes; Ctrl+C or the e-stop stops it). Start?"):
        return
    touch_off = {"ask": None, "yes": True, "no": False}[args.touch_off]
    asyncio.run(with_robot(config, lambda robot: calibrate_scene(robot, config, execute, touch_off), go_ready=execute))


if __name__ == "__main__":
    main()
