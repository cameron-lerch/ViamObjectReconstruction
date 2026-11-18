''' Check the machine side without moving anything: can the planner still reach survey
positions it reached before, and what does the machine's own configuration say?

    python analysis/check_machine.py

Asks the motion planner for paths (plan only, nothing moves) from where the arm is now to
overhead survey positions this arm reached on earlier runs, with the same collision setup
as a scan (our calibrated camera boxes, the table). A position that planned fine before
and is refused now, with the same camera mount, points at a change on the machine side
(the arm's module and its collision model, frames, obstacles, viam-server).

It also saves a snapshot of the machine side to data/machine_checks/<time>-<rig>.json
(viam-server version, frame system, the arm's and gripper's collision geometry, the arm's
kinematics), so later snapshots can be compared with it.
'''

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.protobuf import json_format  # noqa: E402

from recon.config import load_config  # noqa: E402
from recon.robot import Robot, connect  # noqa: E402
from recon.transforms import look_at  # noqa: E402

# Overhead survey positions (x, y, mm; 500mm above the table) this arm planned and reached
# on earlier runs: 2026-11-12 (scans that worked) and 2026-11-17 11:44 and 13:16.
KNOWN_GOOD = [(530, 186), (530, -94), (250, -94), (514, 322), (235, 322), (234, 42), (514, 42), (504, 28), (224, 28)]
# And the ones the 2026-11-17 surveys were refused, for comparison.
REFUSED = [(233, 133), (513, -147), (416, -169), (136, 111)]


def message_dict(message) -> dict:
    try:
        return json_format.MessageToDict(message)
    except Exception:  # not a protobuf message
        return {"repr": repr(message)}


async def snapshot(machine, robot: Robot, config) -> dict:
    found = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rig": config.machine.rig, "address": config.machine.address,
             "resources": sorted(f"{r.subtype}:{r.name}" for r in machine.resource_names)}
    for name, call in (("version", lambda: machine.get_version()),
                       ("frame_system", lambda: machine.get_frame_system_config()),
                       ("arm_geometries", lambda: robot.arm.get_geometries()),
                       ("arm_kinematics", lambda: robot.arm.get_kinematics()),
                       ("gripper_geometries", lambda: robot.gripper.get_geometries() if robot.gripper else [])):
        try:
            value = await call()
        except Exception as error:
            found[name] = {"error": str(error)}
            continue
        if isinstance(value, (list, tuple)):
            found[name] = [message_dict(v) if hasattr(v, "DESCRIPTOR") else repr(v) for v in value]
        elif hasattr(value, "DESCRIPTOR"):
            found[name] = message_dict(value)
        else:
            found[name] = repr(value)
    return found


async def main() -> None:
    config = load_config("config.toml")
    machine = await connect(config)
    try:
        robot = Robot(machine, config)
        start = await robot.checked_camera_pose()
        joints = [round(v, 1) for v in (await robot.arm.get_joint_positions()).values]
        print(f"Arm now: camera at ({start[0, 3]:.0f}, {start[1, 3]:.0f}, {start[2, 3]:.0f}), joints {joints}")
        plane_z = 0.0
        camera = config.machine.camera
        for label, spots in (("reached before", KNOWN_GOOD), ("refused today", REFUSED)):
            print(f"\nPlanning (nothing moves) to positions {label}:")
            for x, y in spots:
                z = plane_z + config.workspace.survey_height
                roll, pose, reason = await robot.plan_best_roll(camera, lambda r: look_at([x, y, z], [x, y, z - 100], r))
                print(f"  ({x:4d}, {y:4d}, {z:.0f}): " + (f"plans (roll {roll:.0f}deg)" if pose is not None else f"REFUSED: {reason}"))
        found = await snapshot(machine, robot, config)
        folder = config.resolve(config.data_dir) / "machine_checks"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{config.machine.rig}.json"
        path.write_text(json.dumps(found, indent=1, default=str) + "\n")
        print(f"\nMachine snapshot saved: {path}")
        if isinstance(found.get("version"), dict):
            print("viam-server:", found["version"])
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
