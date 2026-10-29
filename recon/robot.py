''' Everything that talks to the Viam machine: planning, moving, capturing and gripping.

Reachability checks use the builtin motion service's DoCommand {"plan": ...},
which runs the full planner (IK, joint limits, collisions with the configured
table and walls plus the obstacles we pass) without moving the arm.
'''

import asyncio
import json
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
from viam.proto.component.arm import JointPositions
from viam.proto.service.motion import Constraints, LinearConstraint, MoveRequest
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from .config import Config
from .rgbd import Intrinsics, decode_depth, median_depth
from .scene import DetectedObject, TablePlane
from .transforms import make_transform, matrix_to_pose, pose_to_matrix, rotation_angle_deg

log = logging.getLogger(__name__)

DEPTH_MIME = "image/vnd.viam.dep"


async def connect(config: Config) -> RobotClient:
    path = config.resolve(config.machine.credentials)
    try:
        credentials = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f'Cannot read credentials from {path}; expected {{"key": ..., "key_id": ...}}') from error
    options = RobotClient.Options.with_api_key(
        api_key=credentials.get("key") or credentials["api_key"],
        api_key_id=credentials.get("key_id") or credentials["api_key_id"],
    )
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


class PoseMismatch(RuntimeError):
    ''' Viam's frame system and the arm controller disagree about where the arm is. '''


class MoveDeclined(Exception):
    ''' The operator declined a move. '''


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
            properties = await self.camera.get_properties()
            self._intrinsics = Intrinsics.from_viam(properties.intrinsic_parameters)
        return self._intrinsics

    async def frame_pose(self, frame: str) -> np.ndarray:
        ''' Current pose of any frame in the world, according to Viam's frame system. '''
        result = await self.machine.transform_pose(PoseInFrame(reference_frame=frame, pose=Pose(o_z=1)), "world")
        return pose_to_matrix(result.pose)

    async def camera_mount(self) -> np.ndarray:
        ''' The camera's pose relative to the arm's end.

        From our own calibration file when `[machine] camera_mount_file` is set (so the
        camera frame in the Viam app only affects its 3D scene), otherwise from the
        machine's frame configuration.
        '''
        self._mount_from_file()
        if self._camera_mount is None:
            for part in await self.machine.get_frame_system_config():
                frame = part.frame
                if frame.reference_frame == self.config.machine.camera:
                    parent = frame.pose_in_observer_frame.reference_frame
                    if parent != self.config.machine.arm:
                        raise RuntimeError(f"Camera frame's parent is {parent!r}, expected the arm")
                    self._camera_mount = pose_to_matrix(frame.pose_in_observer_frame.pose)
            if self._camera_mount is None:
                raise RuntimeError("Camera frame not found in the frame system")
        return self._camera_mount

    async def checked_camera_pose(self) -> np.ndarray:
        ''' Camera pose from the arm controller and our camera mount, after checking that
        Viam's frame system agrees with the controller about where the arm's end is.

        On armfarm14 the frame system once reported the camera at its planned target
        while the arm's joints (and the image) showed it rolled 180deg away. The planner
        checks collisions with the frame system, so a disagreement stops everything.
        '''
        frame_system = await self.frame_pose(self.config.machine.arm)
        controller = pose_to_matrix(await self.arm.get_end_position())
        shift = float(np.linalg.norm(frame_system[:3, 3] - controller[:3, 3]))
        angle = rotation_angle_deg(frame_system[:3, :3].T @ controller[:3, :3])
        if shift > POSE_TOLERANCE_MM or angle > POSE_TOLERANCE_DEG:
            joints = [round(v, 1) for v in (await self.arm.get_joint_positions()).values]
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
        if self._camera_mount is None and self.config.machine.camera_mount_file:
            path = self.config.resolve(self.config.machine.camera_mount_file)
            self._camera_mount = np.array(json.loads(path.read_text())["mount"])
        return self._camera_mount

    def camera_guard(self) -> list[Transform]:
        ''' Collision boxes for the camera body and its bracket, attached to the arm's end
        through our calibrated mount, so plans keep the real camera clear of everything
        (see MachineConfig.camera_body_mm). '''
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
            result = await self.motion.do_command({"plan": json_format.MessageToJson(request)}, timeout=60)
        except GRPCError as error:
            return None, short_reason(error)
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

    async def move(self, frame: str, T_world: np.ndarray, world_state: WorldState | None = None,
                   linear: bool = False) -> None:
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
        await self.checked_camera_pose()

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
        answer = await asyncio.to_thread(input, "  Press Enter to move, or type n to stop: ")
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
        ''' Step from wherever the arm is to the ready pose (camera over the mat, looking down),
        one joint group at a time, each step shown and confirmed by the operator.

        Free-moving the arm by hand can leave it anywhere (on 2026-10-29: joint 4 at 290deg),
        and one planned move from such a pose to a camera-down pose turned every joint at
        once, joint 4 by 290deg, fast enough to trip the arm's own speed limit. Steps here
        turn one group at a time, in an order that keeps the arm's shape: the forearm roll
        (4), the wrist (5, 6), the shoulder and elbow (2, 3), then the base (1). They go
        straight to joint angles, without the planner's collision checks, which is why
        each one waits for the operator watching the arm.
        '''
        names = self.config.machine
        ready = np.array(names.ready_joints_deg, float)
        current = np.array((await self.arm.get_joint_positions()).values, float)
        T_end = pose_to_matrix(await self.arm.get_end_position())
        log.info("Arm joints now (deg): %s", np.round(current).astype(int).tolist())
        log.info("  arm's end at (%.0f, %.0f, %.0f)mm, pointing (%+.2f, %+.2f, %+.2f)", *T_end[:3, 3], *T_end[:3, 2])
        log.info("Ready pose (deg):     %s", np.round(ready).astype(int).tolist())
        if np.all(np.abs(current - ready) <= names.ready_tolerance_deg):
            log.info("  already at the ready pose")
            return
        steps = [("forearm roll (joint 4)", [3]), ("wrist (joints 5, 6)", [4, 5]),
                 ("shoulder and elbow (joints 2, 3)", [1, 2]), ("base (joint 1)", [0])]
        for name, joints in steps:
            if self.live:
                await self.live.checkpoint()
            target = current.copy()
            target[joints] = ready[joints]
            change = target - current
            if np.all(np.abs(change[joints]) <= names.ready_tolerance_deg):
                continue
            moves = ", ".join(f"joint {j + 1} {current[j]:.0f} -> {ready[j]:.0f} ({change[j]:+.0f})" for j in joints
                              if abs(change[j]) > names.ready_tolerance_deg)
            print(f"\nTo the ready pose, {name}: {moves}deg")
            print("  (straight to joint angles: no collision checks, so watch the arm)")
            answer = await asyncio.to_thread(input, "  Press Enter to move, or type n to stop: ")
            if answer.strip().lower().startswith("n"):
                raise MoveDeclined(f"Ready pose: the {name} step was declined")
            await self.arm.move_to_joint_positions(JointPositions(values=target.tolist()))
            await self.wait_until_still()
            current = np.array((await self.arm.get_joint_positions()).values, float)
        await self.checked_camera_pose()
        log.info("At the ready pose: %s", np.round(current).astype(int).tolist())

    async def wait_until_still(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while await self.arm.is_moving():
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
        _, metadata = await self.camera.get_images(filter_source_names=[names.color_source])
        stopped_at = metadata.captured_at.ToMicroseconds() / 1e6
        await asyncio.sleep(capture.settle_s)

        sources = [names.color_source, names.depth_source] if depth else [names.color_source]
        depths, stamps, color, color_ext = [], [], None, "jpg"
        colors = []
        deadline = time.monotonic() + capture.frame_timeout_s * frames
        while len(stamps) < frames:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Got only {len(stamps)} of {frames} fresh frames")
            images, metadata = await self.camera.get_images(filter_source_names=sources)
            stamp = metadata.captured_at.ToMicroseconds() / 1e6
            if stopped_at and (stamp < stopped_at + capture.settle_s or stamp in stamps):
                await asyncio.sleep(0.02)
                continue
            for image in images:
                if image.mime_type == DEPTH_MIME:
                    depths.append(decode_depth(image.data))
                else:
                    if keep:
                        colors.append(image.data)
                    if color is None:
                        color = image.data
                        color_ext = "png" if "png" in image.mime_type else "jpg"
            stamps.append(stamp)

        T_world_cam = await self.checked_camera_pose()
        joints = list((await self.arm.get_joint_positions()).values)
        meta = {"captured_at": stamps, "joints_deg": joints, "frames": frames, "depth": depth,
                "camera_mount": (await self.camera_mount()).tolist()}
        self.last_frames = (colors, depths, color_ext) if keep else None
        if depths:
            fused = median_depth(depths)
        else:
            intrinsics = await self.intrinsics()
            fused = np.zeros((intrinsics.height, intrinsics.width), np.uint16)
        return color, color_ext, fused, T_world_cam, meta
