"""Rigid-transform helpers.

Poses are 4x4 homogeneous matrices in millimeters, mapping points from a child
frame into its parent (usually the world). Viam describes orientation with an
orientation vector (o_x, o_y, o_z, theta in degrees): the frame's z-axis points
along (o_x, o_y, o_z) and theta spins the frame about that axis. In rotation
terms that is R = Rz(lon) @ Ry(lat) @ Rz(theta), as in rdk's spatialmath.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

_POLE_EPSILON = 1e-9


def ov_to_matrix(o_x: float, o_y: float, o_z: float, theta_deg: float) -> np.ndarray:
    axis = np.array([o_x, o_y, o_z], dtype=float)
    axis /= np.linalg.norm(axis)
    lat = math.acos(np.clip(axis[2], -1.0, 1.0))
    lon = math.atan2(axis[1], axis[0]) if 1 - abs(axis[2]) > _POLE_EPSILON else 0.0
    return Rotation.from_euler("ZYZ", [lon, lat, math.radians(theta_deg)]).as_matrix()


def matrix_to_ov(rotation: np.ndarray) -> tuple[float, float, float, float]:
    axis = rotation[:, 2] / np.linalg.norm(rotation[:, 2])
    lat = math.acos(np.clip(axis[2], -1.0, 1.0))
    lon = math.atan2(axis[1], axis[0]) if 1 - abs(axis[2]) > _POLE_EPSILON else 0.0
    spin = Rotation.from_euler("ZY", [lon, lat]).as_matrix().T @ rotation
    theta = math.degrees(math.atan2(spin[1, 0], spin[0, 0]))
    return float(axis[0]), float(axis[1]), float(axis[2]), theta


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def pose_to_matrix(pose) -> np.ndarray:
    """Convert a viam.proto.common.Pose (or anything with its fields) to 4x4."""
    rotation = ov_to_matrix(pose.o_x, pose.o_y, pose.o_z, pose.theta)
    return make_transform(rotation, [pose.x, pose.y, pose.z])


def matrix_to_pose(transform: np.ndarray):
    from viam.proto.common import Pose

    o_x, o_y, o_z, theta = matrix_to_ov(transform[:3, :3])
    x, y, z = transform[:3, 3]
    return Pose(x=x, y=y, z=z, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_z(angle_deg: float) -> np.ndarray:
    return Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()


def look_at(eye, target, roll_deg: float = 0.0) -> np.ndarray:
    """Camera pose at `eye` whose optical axis (+z) points at `target`.

    Uses the OpenCV camera convention (+x right, +y down in the image). With
    roll 0 the top of the image faces world +z, or world +x when the camera
    looks straight down. `roll_deg` spins the camera about its optical axis.
    """
    eye = np.asarray(eye, dtype=float)
    z_axis = np.asarray(target, dtype=float) - eye
    z_axis /= np.linalg.norm(z_axis)
    up = np.array([0.0, 0.0, 1.0])
    if abs(z_axis @ up) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    y_axis = -(up - (up @ z_axis) * z_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis]) @ rotation_z(roll_deg)
    return make_transform(rotation, eye)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))
