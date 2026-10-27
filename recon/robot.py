"""Everything that talks to the Viam machine: planning, moving, capturing and gripping.

Reachability checks use the builtin motion service's DoCommand {"plan": ...},
which runs the full planner (IK, joint limits, collisions with the configured
table and walls plus the obstacles we pass) without moving the arm.
"""

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
from viam.proto.service.motion import Constraints, LinearConstraint, MoveRequest
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from .config import Config
from .rgbd import Intrinsics, decode_depth, median_depth
from .scene import DetectedObject, TablePlane
from .transforms import matrix_to_pose, pose_to_matrix, rotation_angle_deg

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
    """Viam's frame system and the arm controller disagree about where the arm is."""


class MoveDeclined(Exception):
    """The operator declined a move."""


class Robot:
    def __init__(self, machine: RobotClient, config: Config):
        self.machine = machine
        self.config = config
        names = config.machine
        self.arm = Arm.from_robot(machine, names.arm)
        self.camera = Camera.from_robot(machine, names.camera)
        self.gripper = Gripper.from_robot(machine, names.gripper)
        self.motion = MotionClient.from_robot(machine, names.motion)
        self._intrinsics: Intrinsics | None = None
        self.table_plane: TablePlane | None = None
        self._camera_mount: np.ndarray | None = None
        # Preview every move and ask the operator (see _confirm_move); with
        # confirm_above_deg set, only ask when some joint turns at least that far.
        self.confirm_moves = False
        self.confirm_above_deg: float | None = None

    def set_measured_table(self, plane: TablePlane) -> None:
        """Use the table as perception measured it, in addition to the machine's configured
        table, which may sit lower than the real one."""
        self.table_plane = plane

    def table_patch(self, obj: DetectedObject) -> Geometry:
        """A slab of measured table around an object, where low views and grasps go.

        Its top is the measured surface's highest point over the patch. The patch
        stops short of the arm base, which stands on the table.
        """
        workspace = self.config.workspace
        center = obj.center[:2]
        half = obj.footprint_radius + workspace.table_patch_size / 2
        half = min(half, (np.linalg.norm(center) - workspace.min_base_distance) / np.sqrt(2))
        corners = [(center[0] + dx, center[1] + dy) for dx in (-half, half) for dy in (-half, half)]
        top = max(self.table_plane.z_at(x, y) for x, y in corners) - workspace.table_obstacle_margin
        thickness = 100.0
        return Geometry(
            center=Pose(x=center[0], y=center[1], z=top - thickness / 2, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=2 * half, y=2 * half, z=thickness)),
            label=f"measured-table-{obj.id}",
        )

    async def intrinsics(self) -> Intrinsics:
        if self._intrinsics is None:
            properties = await self.camera.get_properties()
            self._intrinsics = Intrinsics.from_viam(properties.intrinsic_parameters)
        return self._intrinsics

    async def frame_pose(self, frame: str) -> np.ndarray:
        """Current pose of any frame in the world, according to Viam's frame system."""
        result = await self.machine.transform_pose(PoseInFrame(reference_frame=frame, pose=Pose(o_z=1)), "world")
        return pose_to_matrix(result.pose)

    async def camera_mount(self) -> np.ndarray:
        """The camera's configured pose relative to the arm's end (its frame-system parent)."""
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
        """Camera pose from the frame system, verified against the arm controller.

        On armfarm14 the frame system once reported the camera at its planned
        target while the arm's joints (and the image) showed it rolled 180deg away.
        The planner checks collisions with the frame system, so a disagreement
        stops everything. Returns the pose computed from the controller.
        """
        frame_system = await self.frame_pose(self.config.machine.camera)
        controller = pose_to_matrix(await self.arm.get_end_position()) @ await self.camera_mount()
        shift = float(np.linalg.norm(frame_system[:3, 3] - controller[:3, 3]))
        angle = rotation_angle_deg(frame_system[:3, :3].T @ controller[:3, :3])
        if shift > POSE_TOLERANCE_MM or angle > POSE_TOLERANCE_DEG:
            joints = [round(v, 1) for v in (await self.arm.get_joint_positions()).values]
            raise PoseMismatch(
                f"Frame system and arm controller disagree about the camera by {shift:.1f}mm / {angle:.1f}deg.\n"
                f"  frame system: position {frame_system[:3, 3].round(1)}, x-axis {frame_system[:3, 0].round(3)}\n"
                f"  controller:   position {controller[:3, 3].round(1)}, x-axis {controller[:3, 0].round(3)}\n"
                f"  joints (deg): {joints}")
        return controller

    def world_state(self, obstacles: list[DetectedObject], margin: float,
                    carried: tuple[DetectedObject, np.ndarray] | None = None) -> WorldState | None:
        """Detected objects and the measured table as obstacles, plus an object held in
        the gripper (with its pose relative to the gripper)."""
        geometries = [box_geometry(obj, margin, f"object-{obj.id}") for obj in obstacles]
        if self.table_plane is not None:
            geometries += [self.table_patch(obj) for obj in obstacles if np.linalg.norm(obj.center[:2]) > self.config.workspace.min_base_distance]
        transforms = []
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
        """Plan without moving. Returns (trajectory, "") or (None, reason)."""
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
        """Plan the pose at every configured camera roll and keep the one needing the least
        joint motion from `reference_deg` (default: the current joints).

        The roll about the optical axis barely matters to the image, but the wrong one can
        spin the wrist a long way and wind up the camera cable, so rolls that would take
        the last joint past `max_wrist_deg` are skipped.
        Returns (roll, pose, end joints in degrees), or (None, None, failure reason).
        """
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
        if self.confirm_moves:
            await self._confirm_move(frame, T_world, world_state, linear)
        moved = await self.motion.move(
            component_name=frame,
            destination=PoseInFrame(reference_frame="world", pose=matrix_to_pose(T_world)),
            world_state=world_state,
            constraints=self._constraints(linear),
            timeout=self.config.capture.move_timeout_s,
        )
        if not moved:
            raise RuntimeError(f"Motion service did not complete the move of {frame!r}")
        await self.wait_until_still()
        await self.checked_camera_pose()

    async def _confirm_move(self, frame: str, T_world: np.ndarray, world_state: WorldState | None,
                            linear: bool) -> None:
        """Show the target and the largest joint rotation of a plan, and wait for the operator.

        The move itself is planned again by the motion service, so its path can
        differ from this preview, but the destination is the same.
        """
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
        return Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=2.0, orientation_tolerance_degs=2.0)])

    async def wait_until_still(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while await self.arm.is_moving():
            if time.monotonic() > deadline:
                raise TimeoutError("Arm still reports motion")
            await asyncio.sleep(0.05)

    async def capture(self, frames: int | None = None) -> tuple[bytes, str, np.ndarray, np.ndarray, dict]:
        """Capture a still view: wait for the arm to settle, then take the median of several fresh frames.

        Frames are accepted only if the camera stamped them at least `settle_s`
        after the arm stopped (by the robot's own clock), so frames buffered
        during motion are never used. The camera pose is read after settling.
        Returns (color bytes, color extension, depth mm, T_world_cam, metadata).
        """
        capture = self.config.capture
        frames = frames or capture.frames_per_view
        await self.wait_until_still()
        _, metadata = await self.camera.get_images()
        stopped_at = metadata.captured_at.ToMicroseconds() / 1e6
        await asyncio.sleep(capture.settle_s)

        depths, stamps, color, color_ext = [], [], None, "jpg"
        # Each get_images can take seconds over a cloud connection (depth alone is ~1.8MB).
        deadline = time.monotonic() + capture.frame_timeout_s * frames
        while len(depths) < frames:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Got only {len(depths)} of {frames} fresh frames")
            images, metadata = await self.camera.get_images()
            stamp = metadata.captured_at.ToMicroseconds() / 1e6
            if stopped_at and (stamp < stopped_at + capture.settle_s or stamp in stamps):
                await asyncio.sleep(0.02)
                continue
            for image in images:
                if image.mime_type == DEPTH_MIME:
                    depths.append(decode_depth(image.data))
                elif color is None:
                    color = image.data
                    color_ext = "png" if "png" in image.mime_type else "jpg"
            stamps.append(stamp)

        T_world_cam = await self.checked_camera_pose()
        joints = list((await self.arm.get_joint_positions()).values)
        meta = {"captured_at": stamps, "joints_deg": joints, "frames": frames,
                "camera_mount": (await self.camera_mount()).tolist()}
        return color, color_ext, median_depth(depths), T_world_cam, meta
