import numpy as np
import pytest

from recon.transforms import look_at, matrix_to_ov, ov_to_matrix, pose_to_matrix, matrix_to_pose


def test_matches_viam_frame_system():
    # Camera pose and axes reported by the machine's transform_pose on the xArm6
    # (camera x-axis = transform of (100, 0, 0) in the cam frame, and so on).
    rotation = ov_to_matrix(-0.020992479341378847, -0.014881124706322427, -0.99966887914877922, -52.885791982507435)
    np.testing.assert_allclose(rotation[:, 0], [0.0309, 0.9994, -0.0155], atol=2e-4)
    np.testing.assert_allclose(rotation[:, 1], [0.9993, -0.0313, -0.0205], atol=2e-4)


@pytest.mark.parametrize("ov", [(0, 0, 1, 0), (0, 0, -1, 45), (0.3, -0.5, 0.8, 120), (-1, 0, 0, -90), (0.01, 0, -1, 10)])
def test_orientation_vector_round_trip(ov):
    rotation = ov_to_matrix(*ov)
    np.testing.assert_allclose(ov_to_matrix(*matrix_to_ov(rotation)), rotation, atol=1e-9)


def test_pose_round_trip():
    transform = look_at([400, -100, 300], [450, -150, 0], roll_deg=30)
    np.testing.assert_allclose(pose_to_matrix(matrix_to_pose(transform)), transform, atol=1e-9)


def test_look_at_points_optical_axis_at_target_with_image_up():
    eye, target = np.array([300.0, 0, 200]), np.array([500.0, 0, 0])
    pose = look_at(eye, target)
    forward = (target - eye) / np.linalg.norm(target - eye)
    np.testing.assert_allclose(pose[:3, 2], forward, atol=1e-12)
    assert pose[:3, 1] @ [0, 0, 1] < 0  # image +y (down) points away from world up
    np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0)


def test_look_at_straight_down_is_well_defined():
    pose = look_at([400, 0, 500], [400, 0, 0])
    np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0)
