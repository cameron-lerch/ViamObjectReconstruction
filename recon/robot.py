''' Everything that talks to the Viam machine: planning, moving, capturing and gripping.

Reachability checks use the builtin motion service's DoCommand {"plan": ...},
which runs the full planner (IK, joint limits, collisions with the configured
table and walls plus the obstacles we pass) without moving the arm.
'''

import asyncio
import json
import sys
import logging
import time

import numpy as np
from google.protobuf import json_format
from grpclib.exceptions import GRPCError
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import (
    Geometry,
    GeometriesInFrame,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Transform,
    Vector3,
    WorldState,
)
from viam.proto.service.motion import Constraints, LinearConstraint, MoveRequest
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from .config import Config
from .rgbd import Intrinsics, decode_depth, decode_depth_png, median_depth
from .scene import DetectedObject, TablePlane
from .transforms import look_at, make_transform, matrix_to_pose, pose_to_matrix, rotation_angle_deg

log = logging.getLogger(__name__)

DEPTH_MIME = "image/vnd.viam.dep"


async def connect(config: Config) -> RobotClient:
    path = config.resolve(config.machine.credentials)
    try:
        credentials = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f'Cannot read credentials from {path}; expected {{"api_key": ..., "api_key_id": ...}}') from error
    # The same names as the Viam app's CONNECT tab (API keys) and the SDK's code sample.
    options = RobotClient.Options.with_api_key(api_key=credentials["api_key"], api_key_id=credentials["api_key_id"])
    # The SDK's background health check allows only 1s for a reply. While a 1.8MB
    # depth frame downloads over a cloud link that misses, and the SDK then closes
    # the channel mid-download. With the check off, a real disconnect still raises
    # on the next call and stops the script; viam-server's session heartbeat (which
    # stops the arm if this client goes away) is unaffected.
    options.check_connection_interval = 0
    options.attempt_reconnect_interval = 0
    return await RobotClient.at_address(config.machine.address, options)


def box_geometry(obj: DetectedObject, margin: float, label: str, pose: np.ndarray | None = None) -> Geometry:
    dims = obj.size + 2 * margin
    return Geometry(
        center=matrix_to_pose(obj.pose if pose is None else pose),
        box=RectangularPrism(dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])),
        label=label,
    )


def short_reason(error: GRPCError) -> str:
    message = error.message or str(error)
    if "too far" in message:
        return "out of reach"
    if "violation between" in message:
        start = message.index("violation between") + len("violation between ")
        return "collision: " + message[start:].split(" geometries")[0]
    return message[:120]


POSE_TOLERANCE_MM = 2.0
POSE_TOLERANCE_DEG = 1.0


# After a move, the arm's end must be this close to where it was sent. The motion service
# reported success on moves that ended elsewhere (a bowl scan, 2026-11-03: twice the camera
# ended up facing away from the object), and a view captured there would be placed wrongly.
REACH_TOLERANCE_MM = 10.0
REACH_TOLERANCE_DEG = 5.0


class PoseMismatch(RuntimeError):
    ''' Viam's frame system and the arm controller disagree about where the arm is. '''


class MissedTarget(RuntimeError):
    ''' A move finished, but the arm's end isn't where it was sent. '''


class RobotUnresponsive(RuntimeError):
    ''' A call to the robot got no answer, twice: the connection or viam-server is likely gone
    (on 2026-11-03 viam-server restarted mid-scan, and the scan waited for an answer forever). '''


class MoveDeclined(Exception):
    ''' The operator declined a move. '''


def ask_operator(prompt: str) -> str:
    ''' input(), after dropping anything typed while the arm was working: a stray key (an
    arrow key and Enter, on 2026-11-05) otherwise answers the next question by itself. '''
    try:
        import termios
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (ImportError, OSError):  # not a terminal (or not Unix): nothing to drop
        pass
    return input(prompt)


class Robot:
    def __init__(self, machine: RobotClient, config: Config):
        self.machine = machine
        self.config = config
        names = config.machine
        self.arm = Arm.from_robot(machine, names.arm)
        self.camera = Camera.from_robot(machine, names.camera)
        self.gripper = Gripper.from_robot(machine, names.gripper) if names.gripper else None
        self.motion = MotionClient.from_robot(machine, names.motion)
        self._intrinsics: Intrinsics | None = None
        self.table_plane: TablePlane | None = None
        self.clutter: list[DetectedObject] = []  # things on the table that aren't scan targets
        # Off only for touch-off, which has to come close to the table on purpose.
        self.use_table_slab = True
        self._camera_mount: np.ndarray | None = None
        # calibration.py starts fresh: the saved camera calibration is ignored until it saves a new one.
        self.ignore_saved_calibration = False
        # Preview every move and ask the operator (see _confirm_move); with
        # confirm_above_deg set, only ask when some joint turns at least that far.
        self.confirm_moves = False
        self.confirm_above_deg: float | None = None
        # The live viewer (recon/live.py), if one is running: its Pause and Stop act before every move.
        self.live = None
        self.last_frames = None  # every raw frame of the last capture made with keep=True

    def set_measured_table(self, plane: TablePlane) -> None:
        ''' Use the table as perception measured it, in addition to the machine's configured
        table, which may sit lower than the real one. '''
        self.table_plane = plane

    def set_scene(self, plane: TablePlane, clutter: list[DetectedObject]) -> None:
        ''' The measured table, plus everything else standing on it, as obstacles for every plan. '''
        self.set_measured_table(plane)
        self.clutter = list(clutter)

    def table_slab(self) -> list[Geometry]:
        ''' The table as an obstacle whose top sits a safety margin ABOVE the measured surface.

        The machine's configured table sat 6-11mm below the real one, and the gripper
        hit the table on a low view. This slab covers the search region (plus 150mm)
        with a square cut out around the arm's base (which stands on the table), as
        four boxes. Its top is the measured surface's highest point over each box,
        plus `table_safety_margin`, plus any extra reach of the real fingertips over
        the gripper's collision model found by `scan.py touch-off`.
        '''
        workspace = self.config.workspace
        if not (workspace.x_range and workspace.y_range):
            return []  # the survey hasn't found the table yet: Viam's own table only
        (x0, x1), (y0, y1) = workspace.x_range, workspace.y_range
        # 150mm past the search region, so paths that swing a little outside it are covered too.
        x0, x1, y0, y1 = x0 - 150, x1 + 150, y0 - 150, y1 + 150
        c = workspace.base_clearance
        boxes = [(x0, x1, c, y1), (x0, x1, y0, -c), (c, x1, -c, c), (x0, -c, -c, c)]
        margin = workspace.table_safety_margin + self.fingertip_extra_reach()
        slab, thickness = [], 100.0
        for k, (bx0, bx1, by0, by1) in enumerate(boxes):
            if bx1 <= bx0 or by1 <= by0:
                continue
            top = max(self.table_plane.z_at(x, y) for x in (bx0, bx1) for y in (by0, by1)) + margin
            slab.append(Geometry(
                center=Pose(x=(bx0 + bx1) / 2, y=(by0 + by1) / 2, z=top - thickness / 2, o_z=1),
                box=RectangularPrism(dims_mm=Vector3(x=bx1 - bx0, y=by1 - by0, z=thickness)),
                label=f"measured-table-{k}"))
        return slab

    def fingertip_extra_reach(self) -> float:
        ''' How much further the real fingertips reach than the gripper's collision model,
        from the touch-off file (0 when there is none or they don't). '''
        path = self.config.resolve(self.config.machine.touch_off_file) if self.config.machine.touch_off_file else None
        if path is None or not path.exists() or not self.config.machine.gripper:
            return 0.0
        return max(0.0, float(json.loads(path.read_text()).get("extra_reach_mm", 0.0)))

    async def intrinsics(self) -> Intrinsics:
        if self._intrinsics is None:
            properties = await self.ask("the camera's properties", lambda: self.camera.get_properties())
            self._intrinsics = Intrinsics.from_viam(properties.intrinsic_parameters)
        return self._intrinsics

    async def frame_pose(self, frame: str) -> np.ndarray:
        ''' Current pose of any frame in the world, according to Viam's frame system. '''
        result = await self.ask(f"the pose of {frame!r}", lambda: self.machine.transform_pose(
            PoseInFrame(reference_frame=frame, pose=Pose(o_z=1)), "world"))
        return pose_to_matrix(result.pose)

    def start_fresh(self) -> None:
        ''' Forget this arm's saved camera calibration for the rest of the run (calibration.py
        finds it again, starting from the camera frame in the Viam app). '''
        self.ignore_saved_calibration = True
        self._camera_mount = None

    def use_camera_mount(self, mount: np.ndarray) -> None:
        ''' Use `mount` for every move and capture from now on (in this run only; save_mount keeps it). '''
        self._camera_mount = np.asarray(mount, dtype=float)

    async def camera_mount(self) -> np.ndarray:
        ''' The camera's pose relative to the arm's end.

        From our own calibration file when `[machine] camera_mount_file` names one that
        exists (so the camera frame in the Viam app only affects its 3D scene). A new arm
        has none until its first calibration, which then starts from the camera frame
        configured for this machine in the Viam app (whatever it's attached to: the arm's
        end, or a gripper on it), never from another arm's calibration.
        '''
        self._mount_from_file()
        if self._camera_mount is None:
            names = self.config.machine
            if getattr(self, "ignore_saved_calibration", False):
                log.info("Starting fresh: the camera mount starts from the camera frame in the Viam app")
            else:
                log.info("No camera calibration for this arm yet (%s): starting from the camera frame in the Viam app",
                         names.camera_mount_file or "camera_mount_file not set")
            self._camera_mount = np.linalg.inv(await self.frame_pose(names.arm)) @ await self.frame_pose(names.camera)
        return self._camera_mount

    async def checked_camera_pose(self) -> np.ndarray:
        ''' Camera pose from the arm controller and our camera mount, after checking that
        Viam's frame system agrees with the controller about where the arm's end is.

        On armfarm14 the frame system once reported the camera at its planned target
        while the arm's joints (and the image) showed it rolled 180deg away. The planner
        checks collisions with the frame system, so a disagreement stops everything.
        '''
        frame_system = await self.frame_pose(self.config.machine.arm)
        controller = pose_to_matrix(await self.ask("the arm's position", lambda: self.arm.get_end_position()))
        shift = float(np.linalg.norm(frame_system[:3, 3] - controller[:3, 3]))
        angle = rotation_angle_deg(frame_system[:3, :3].T @ controller[:3, :3])
        if shift > POSE_TOLERANCE_MM or angle > POSE_TOLERANCE_DEG:
            joints = [round(v, 1) for v in (await self.ask("the arm's joints", lambda: self.arm.get_joint_positions())).values]
            raise PoseMismatch(
                f"Frame system and arm controller disagree about the arm's end by {shift:.1f}mm / {angle:.1f}deg.\n"
                f"  frame system: position {frame_system[:3, 3].round(1)}, x-axis {frame_system[:3, 0].round(3)}\n"
                f"  controller:   position {controller[:3, 3].round(1)}, x-axis {controller[:3, 0].round(3)}\n"
                f"  joints (deg): {joints}")
        return controller @ await self.camera_mount()

    async def _resolve(self, frame: str, T_world: np.ndarray) -> tuple[str, np.ndarray]:
        ''' Camera targets become arm-end targets through our camera mount, so motion
        never depends on the camera frame configured in the Viam app. '''
        if frame == self.config.machine.camera:
            return self.config.machine.arm, T_world @ np.linalg.inv(await self.camera_mount())
        return frame, T_world

    def _mount_from_file(self) -> np.ndarray | None:
        if self._camera_mount is None and self.calibrated():
            path = self.config.resolve(self.config.machine.camera_mount_file)
            self._camera_mount = np.array(json.loads(path.read_text())["mount"])
        return self._camera_mount

    def calibrated(self) -> bool:
        ''' Whether this arm has its own camera calibration (the mount file exists). '''
        file = self.config.machine.camera_mount_file
        return bool(file) and self.config.resolve(file).exists() and not getattr(self, "ignore_saved_calibration", False)

    def camera_guard(self) -> list[Transform]:
        ''' Collision boxes for the camera body and its bracket, attached to the arm's end
        through our calibrated mount, so plans keep the real camera clear of everything
        (see MachineConfig.camera_body_mm). Only from a calibrated mount: before an arm's
        first calibration, the mount in use is a rough guess (boxes placed by a wrong
        guess can overlap the arm itself, and then nothing can move), so the camera
        geometry in the Viam app is all the planner checks. '''
        if not self.calibrated():
            return []
        mount = self._mount_from_file()
        if mount is None:
            return []
        names = self.config.machine
        body = mount @ make_transform(np.eye(3), [0.0, 0.0, names.camera_body_offset_mm])
        reach = mount[:3, 3]
        along = reach / np.linalg.norm(reach)
        side = np.cross(along, [0.0, 0.0, 1.0] if abs(along[2]) < 0.9 else [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        bracket = make_transform(np.column_stack([along, side, np.cross(along, side)]), reach / 2)
        guards = []
        for label, pose, dims in (("camera-body-guard", body, names.camera_body_mm),
                                  ("camera-bracket-guard", bracket,
                                   [np.linalg.norm(reach), names.camera_bracket_mm, names.camera_bracket_mm])):
            guards.append(Transform(
                reference_frame=label,
                pose_in_observer_frame=PoseInFrame(reference_frame=names.arm, pose=matrix_to_pose(pose)),
                physical_object=Geometry(center=matrix_to_pose(np.eye(4)),
                                         box=RectangularPrism(dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])),
                                         label=label)))
        return guards

    def world_state(self, obstacles: list[DetectedObject], margin: float,
                    carried: tuple[DetectedObject, np.ndarray] | None = None) -> WorldState | None:
        ''' Detected objects, the clutter around them and the measured table as obstacles,
        plus an object held in the gripper (with its pose relative to the gripper). '''
        geometries = [box_geometry(obj, margin, f"object-{obj.id}") for obj in obstacles]
        geometries += [box_geometry(obj, margin, f"clutter-{obj.id}") for obj in self.clutter]
        if self.table_plane is not None and self.use_table_slab:
            geometries += self.table_slab()
        transforms = self.camera_guard()
        if carried is not None:
            obj, T_gripper_obj = carried
            transforms.append(Transform(
                reference_frame=f"carried-object-{obj.id}",
                pose_in_observer_frame=PoseInFrame(reference_frame=self.config.machine.gripper, pose=matrix_to_pose(T_gripper_obj)),
                physical_object=box_geometry(obj, 5.0, f"carried-object-{obj.id}", np.eye(4)),
            ))
        if not geometries and not transforms:
            return None
        obstacle_list = [GeometriesInFrame(reference_frame="world", geometries=geometries)] if geometries else []
        return WorldState(obstacles=obstacle_list, transforms=transforms)

    async def plan(self, frame: str, T_world: np.ndarray, world_state: WorldState | None = None,
                   linear: bool = False) -> tuple[list | None, str]:
        ''' Plan without moving. Returns (trajectory, "") or (None, reason). '''
        if world_state is None:  # still the table, the clutter and the camera guard
            world_state = self.world_state([], 0.0)
        frame, T_world = await self._resolve(frame, T_world)
        request = MoveRequest(
            name=self.config.machine.motion,
            component_name=frame,
            destination=PoseInFrame(reference_frame="world", pose=matrix_to_pose(T_world)),
            world_state=world_state,
            constraints=self._constraints(linear),
        )
        try:
            result = await self.motion.do_command({"plan": json_format.MessageToJson(request)},
                                                  timeout=self.config.capture.plan_timeout_s)
        except GRPCError as error:
            return None, short_reason(error)
        except (TimeoutError, asyncio.TimeoutError):  # the planner searched without finding a path
            return None, f"the planner found no path within {self.config.capture.plan_timeout_s:.0f}s"
        return result.get("plan"), ""

    async def plan_best_roll(self, frame: str, pose_for_roll, world_state: WorldState | None = None,
                             reference_deg: np.ndarray | None = None):
        ''' Plan the pose at every configured camera roll and keep the one needing the least
        joint motion from `reference_deg` (default: the current joints).

        The roll about the optical axis barely matters to the image, but the wrong one can
        spin the wrist a long way and wind up the camera cable, so rolls that would take
        the last joint past `max_wrist_deg` are skipped.
        Returns (roll, pose, end joints in degrees), or (None, None, failure reason).
        '''
        best, reasons = None, []
        for roll in self.config.views.rolls_deg:
            pose = pose_for_roll(roll)
            trajectory, reason = await self.plan(frame, pose, world_state)
            if trajectory is None:
                reasons.append(reason)
                continue
            arm = self.config.machine.arm
            start = np.degrees(trajectory[0][arm]) if reference_deg is None else reference_deg
            end = np.degrees(trajectory[-1][arm])
            if abs(end[-1]) > self.config.views.max_wrist_deg:
                reasons.append(f"wrist (last joint) would reach {end[-1]:.0f}deg")
                continue
            motion = float(np.abs(end - start).max())
            if best is None or motion < best[0]:
                best = (motion, roll, pose, end)
        if best is None:
            return None, None, max(set(reasons), key=reasons.count) if reasons else "no rolls configured"
        return best[1], best[2], best[3]

    async def ask(self, what: str, call, timeout: float | None = None):
        ''' Await `call()` (a call to the robot) with a deadline. A call that gets no answer is
        asked once more; a second miss raises RobotUnresponsive, which ends the run. '''
        timeout = timeout or self.config.capture.call_timeout_s
        for attempt in (1, 2):
            try:
                return await asyncio.wait_for(call(), timeout)
            except TimeoutError:
                if attempt == 2:
                    raise RobotUnresponsive(f"no answer from the robot to {what} within {timeout:.0f}s, twice") from None
                log.warning("  no answer from the robot to %s within %.0fs; asking again", what, timeout)

    async def move(self, frame: str, T_world: np.ndarray, world_state: WorldState | None = None,
                   linear: bool = False) -> None:
        ''' Move `frame` to `T_world`. A move that finishes away from its target is tried again
        from wherever the arm ended up (`capture.move_retries` times) before it fails. '''
        attempts = 1 + max(self.config.capture.move_retries, 0)
        for attempt in range(1, attempts + 1):
            try:
                await self._move_once(frame, T_world, world_state, linear)
                return
            except MissedTarget as missed:
                if attempt == attempts:
                    raise
                log.warning("  %s; trying the move again (%d of %d)", missed, attempt + 1, attempts)

    async def _move_once(self, frame: str, T_world: np.ndarray, world_state: WorldState | None,
                         linear: bool) -> None:
        if self.live:
            await self.live.checkpoint()
        if world_state is None:  # still the table, the clutter and the camera guard
            world_state = self.world_state([], 0.0)
        if self.confirm_moves:
            await self._confirm_move(frame, T_world, world_state, linear)
        frame, T_world = await self._resolve(frame, T_world)
        try:
            moved = await self.motion.move(
                component_name=frame,
                destination=PoseInFrame(reference_frame="world", pose=matrix_to_pose(T_world)),
                world_state=world_state,
                constraints=self._constraints(linear),
                timeout=self.config.capture.move_timeout_s,
            )
        except Exception:
            if self.live and self.live.stopping:  # STOP halted this move
                await self.live.checkpoint()
            raise
        if self.live and self.live.stopping:
            await self.live.checkpoint()
        if not moved:
            raise RuntimeError(f"Motion service did not complete the move of {frame!r}")
        await self.wait_until_still()
        reached = await self.checked_camera_pose()
        if frame == self.config.machine.arm:  # camera targets were resolved to the arm's end
            end = reached @ np.linalg.inv(await self.camera_mount())
            shift = float(np.linalg.norm(end[:3, 3] - T_world[:3, 3]))
            angle = rotation_angle_deg(T_world[:3, :3].T @ end[:3, :3])
            if shift > REACH_TOLERANCE_MM or angle > REACH_TOLERANCE_DEG:
                raise MissedTarget(f"the move finished {shift:.0f}mm / {angle:.0f}deg from its target "
                                   f"(arm's end at {end[:3, 3].round(0)}, sent to {T_world[:3, 3].round(0)})")

    async def _confirm_move(self, frame: str, T_world: np.ndarray, world_state: WorldState | None,
                            linear: bool) -> None:
        ''' Show the target and the largest joint rotation of a plan, and wait for the operator.

        The move itself is planned again by the motion service, so its path can
        differ from this preview, but the destination is the same.
        '''
        trajectory, reason = await self.plan(frame, T_world, world_state, linear)
        if trajectory is None:
            raise RuntimeError(f"Cannot plan the move of {frame!r}: {reason}")
        arm = self.config.machine.arm
        start, end = np.degrees(trajectory[0][arm]), np.degrees(trajectory[-1][arm])
        change = end - start
        joint = int(np.argmax(np.abs(change)))
        x, y, z = T_world[:3, 3]
        dx, dy, dz = T_world[:3, 2]
        print(f"\nNext move: {frame} to ({x:.0f}, {y:.0f}, {z:.0f})mm, pointing ({dx:+.2f}, {dy:+.2f}, {dz:+.2f})"
              f"{', straight line' if linear else ''}")
        print(f"  planned joint change (deg): {np.round(change, 1).tolist()}; "
              f"largest is joint {joint + 1} by {change[joint]:+.0f}deg, {len(trajectory)} waypoints")
        if self.confirm_above_deg is not None and abs(change[joint]) < self.confirm_above_deg:
            print(f"  moving (every joint under {self.confirm_above_deg:.0f}deg)")
            return
        answer = await asyncio.to_thread(ask_operator, "  Press Enter to move, or type n to stop: ")
        if answer.strip().lower() in ("n", "no", "q", "stop"):
            raise MoveDeclined(f"Move of {frame} declined by the operator")

    @staticmethod
    def _constraints(linear: bool) -> Constraints | None:
        if not linear:
            return None
        # Stay within 2mm of the straight line and 2deg of the orientation (judgment: tight
        # enough for the touch-off's 1mm steps, loose enough for the planner to solve).
        return Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=2.0, orientation_tolerance_degs=2.0)])

    async def go_ready(self) -> None:
        ''' Move the camera to the home position saved in `[machine] home_file` (looking
        straight down at the calibration mat; see MachineConfig.home_file), with one planned
        move that avoids the table and everything the survey saw. Nothing moves when the
        camera is already within 10mm of it, or when no home position has been found yet. '''
        path = self.config.resolve(self.config.machine.home_file) if self.config.machine.home_file else None
        if path is None or not path.exists():
            log.info("No home position yet: it's set when the survey finds the ChArUco mat; starting from where the arm is")
            return
        home = np.array(json.loads(path.read_text())["camera_mm"], dtype=float)
        camera = await self.checked_camera_pose()
        if np.linalg.norm(camera[:3, 3] - home) <= 10.0 and camera[2, 2] < -0.99:
            log.info("Camera already at the home position")
            return
        log.info("Moving to the home position: camera at (%.0f, %.0f, %.0f), looking down at the mat", *home)
        frame = self.config.machine.camera
        _, pose, reason = await self.plan_best_roll(frame, lambda roll: look_at(home, home - [0.0, 0.0, 100.0], roll))
        if pose is None:
            raise RuntimeError(f"Can't reach the home position ({path.name}): {reason}. If the mat or the arm moved, "
                               "delete that file: the next run finds the mat again")
        await self.move(frame, pose)

    async def wait_until_still(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while await self.ask("whether the arm is moving", lambda: self.arm.is_moving()):
            if time.monotonic() > deadline:
                raise TimeoutError("Arm still reports motion")
            await asyncio.sleep(0.05)

    async def capture(self, frames: int | None = None, depth: bool = True,
                      keep: bool = False) -> tuple[bytes, str, np.ndarray, np.ndarray, dict]:
        ''' Capture a still view: wait for the arm to settle, then take the median of several fresh frames.

        Frames are accepted only if the camera stamped them at least `settle_s`
        after the arm stopped (by the robot's own clock), so frames buffered
        during motion are never used. The camera pose is read after settling.
        With `depth=False` only color is fetched (about 0.15s instead of 2-3s over
        a cloud connection, where a depth image is 1.8MB) and the depth is empty.
        Returns (color bytes, color extension, depth mm, T_world_cam, metadata). With
        `keep`, every frame's color bytes and depth are also left in `self.last_frames`.
        '''
        capture = self.config.capture
        names = self.config.machine
        frames = frames or capture.frames_per_view
        await self.wait_until_still()
        # The robot's clock at the stop, from a cheap color-only frame.
        _, metadata = await self.ask("a color frame", lambda: self.camera.get_images(filter_source_names=[names.color_source]),
                                     capture.frame_request_timeout_s)
        stopped_at = metadata.captured_at.ToMicroseconds() / 1e6
        await asyncio.sleep(capture.settle_s)

        sources = [names.color_source, names.depth_source] if depth else [names.color_source]
        depths, stamps, color, color_ext = [], [], None, "jpg"
        colors = []
        deadline = time.monotonic() + capture.frame_timeout_s * frames
        while len(stamps) < frames:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Got only {len(stamps)} of {frames} fresh frames")
            images, metadata = await self.ask("a frame", lambda: self.camera.get_images(filter_source_names=sources),
                                              capture.frame_request_timeout_s)
            stamp = metadata.captured_at.ToMicroseconds() / 1e6
            if stopped_at and (stamp < stopped_at + capture.settle_s or stamp in stamps):
                await asyncio.sleep(0.02)
                continue
            for image in images:
                # Depth by its source name or Viam's depth format: another camera module (the
                # Orbbec's, say) may send depth as a 16-bit PNG instead.
                if image.mime_type == DEPTH_MIME or (depth and getattr(image, "name", "") == names.depth_source):
                    depths.append(decode_depth(image.data) if image.mime_type == DEPTH_MIME else decode_depth_png(image.data))
                else:
                    if keep:
                        colors.append(image.data)
                    if color is None:
                        color = image.data
                        color_ext = "png" if "png" in image.mime_type else "jpg"
            stamps.append(stamp)

        T_world_cam = await self.checked_camera_pose()
        joints = list((await self.ask("the arm's joints", lambda: self.arm.get_joint_positions())).values)
        meta = {"captured_at": stamps, "joints_deg": joints, "frames": frames, "depth": depth,
                "camera_mount": (await self.camera_mount()).tolist()}
        self.last_frames = (colors, depths, color_ext) if keep else None
        if depths:
            fused = await asyncio.to_thread(median_depth, depths)  # ~1.4s for 50 frames: off the loop
        else:
            intrinsics = await self.intrinsics()
            fused = np.zeros((intrinsics.height, intrinsics.width), np.uint16)
        return color, color_ext, fused, T_world_cam, meta
