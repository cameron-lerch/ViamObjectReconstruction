import numpy as np

from recon.rgbd import Intrinsics, View, back_project, decode_depth, edge_mask, load_view, median_depth, save_view

INTRINSICS = Intrinsics(8, 6, 10.0, 10.0, 4.0, 3.0)


def test_decode_viam_depth_map():
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
    data = b"DEPTHMAP" + (4).to_bytes(8, "big") + (3).to_bytes(8, "big") + depth.astype(">u2").tobytes()
    np.testing.assert_array_equal(decode_depth(data), depth)


def test_median_depth_ignores_dropouts():
    frames = [np.full((2, 2), 300, np.uint16), np.full((2, 2), 310, np.uint16), np.full((2, 2), 302, np.uint16)]
    frames[0][0, 0] = 0
    frames[1][1, 1] = 0
    frames[2][1, 1] = 0
    result = median_depth(frames)
    assert result[0, 1] == 302
    assert result[0, 0] == 306  # median of the two valid readings
    assert result[1, 1] == 0  # valid in only one of three frames


def test_edge_mask_flags_both_sides_of_a_jump():
    depth = np.full((5, 6), 300, np.uint16)
    depth[:, 3:] = 500
    mask = edge_mask(depth)
    assert mask[:, 2:4].all() and not mask[:, [0, 5]].any()


def test_back_project_center_pixel_lies_on_optical_axis():
    depth = np.zeros((6, 8), np.uint16)
    depth[3, 4] = 400
    view = View(np.zeros((6, 8, 3), np.uint8), depth, np.eye(4), INTRINSICS)
    points, _, pixels = back_project(view)
    np.testing.assert_allclose(points, [[0, 0, 400]])
    np.testing.assert_array_equal(pixels, [[3, 4]])


def test_save_and_load_view(tmp_path):
    color = np.random.default_rng(0).integers(0, 255, (6, 8, 3), np.uint8)
    depth = np.arange(48, dtype=np.uint16).reshape(6, 8) + 300
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    save_view(tmp_path / "v", color, depth, pose, INTRINSICS, {"joints_deg": [1, 2]})
    view = load_view(tmp_path / "v")
    np.testing.assert_array_equal(view.depth, depth)
    np.testing.assert_allclose(view.T_world_cam, pose)
    assert view.intrinsics == INTRINSICS and view.meta["joints_deg"] == [1, 2]
