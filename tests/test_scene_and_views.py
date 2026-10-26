import numpy as np
import open3d as o3d
import pytest

from recon import synthetic
from recon.config import Config, load_config
from recon.relocate import grasp_options
from recon.rgbd import back_project
from recon.scene import DetectedObject, detect_objects, fit_table_plane, merge_fragments, min_area_rectangle
from recon.transforms import look_at
from recon.views import aim_point, coverage, generate_candidates, order_for_travel


@pytest.fixture(scope="module")
def survey():
    o3d.utility.random.seed(0)
    config = Config()
    view = synthetic.render_views(synthetic.demo_scene(), [look_at([380, -150, 550], [380, -150, 0])], frames=3)[0]
    points = back_project(view, 2, config.camera.min_depth, config.camera.max_depth)[0]
    plane = fit_table_plane(points, config.workspace)
    return plane, detect_objects(points, plane, config.workspace)


def test_min_area_rectangle_recovers_rotated_box():
    corners = np.array([[-35, -20], [35, -20], [35, 20], [-35, 20]], float)
    angle = np.radians(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center, extents, yaw = min_area_rectangle(corners @ rotation.T + [100, 50])
    np.testing.assert_allclose(center, [100, 50], atol=1e-9)
    np.testing.assert_allclose(extents, [70, 40], atol=1e-9)
    assert yaw == pytest.approx(30)


def test_table_plane_is_found(survey):
    plane, _ = survey
    assert plane.z_at(400, -150) == pytest.approx(synthetic.TABLE_Z, abs=3)


def test_detects_both_objects_where_they_are(survey):
    # The simulated camera-mount error (0.8deg) shifts the survey by up to ~7mm at
    # this height; scan views and crop margins only need the object roughly located.
    _, objects = survey
    assert len(objects) == 2
    cylinder = min(objects, key=lambda o: np.hypot(o.center[0] - 430, o.center[1] + 170))
    box = min(objects, key=lambda o: np.hypot(o.center[0] - 340, o.center[1] + 40))
    assert np.hypot(cylinder.center[0] - 430, cylinder.center[1] + 170) < 8
    assert cylinder.size[2] == pytest.approx(90, abs=8)
    assert np.hypot(box.center[0] - 340, box.center[1] + 40) < 8
    assert box.size[:2] == pytest.approx([70, 40], abs=8)
    assert box.yaw_deg == pytest.approx(30, abs=5)


def test_overlapping_fragments_merge_and_separate_objects_do_not():
    grid = np.stack(np.meshgrid(np.arange(10.0), np.arange(10.0)), -1).reshape(-1, 2)
    lift = lambda xy: np.column_stack([xy, np.ones(len(xy))])
    body, sliver, other = lift(grid * 4), lift(grid[:20] * 2 + [5, 5]), lift(grid * 4 + [100, 0])
    merged = merge_fragments([body, other, sliver])
    assert sorted(len(c) for c in merged) == [100, 120]


def make_object():
    return DetectedObject(0, np.array([400.0, -150.0, 37.0]), np.array([50.0, 50.0, 90.0]), 0.0, 1000)


def test_full_candidate_set_covers_the_object():
    config = Config().views
    obj = make_object()
    candidates = generate_candidates(obj, config)
    assert coverage(aim_point(obj, config), [c.eye for c in candidates], config) >= config.min_coverage
    for candidate in candidates:
        distance = np.linalg.norm(candidate.eye - candidate.target)
        assert distance - obj.radius >= config.standoff - 1e-6


def test_one_sided_views_fall_short_of_coverage():
    config = Config().views
    obj = make_object()
    near_side = [c.eye for c in generate_candidates(obj, config) if np.cos(np.radians(c.azimuth_deg)) < 0]
    assert coverage(aim_point(obj, config), near_side, config) < config.min_coverage


def test_travel_order_starts_on_top_and_visits_every_view_once():
    candidates = generate_candidates(make_object(), Config().views)
    ordered = order_for_travel(candidates)
    assert ordered[0].elevation_deg == 90
    assert sorted(id(c) for c in ordered) == sorted(id(c) for c in candidates)
    assert [c.elevation_deg for c in ordered] == sorted([c.elevation_deg for c in ordered], reverse=True)


def test_grasp_closes_across_the_narrow_side_pointing_down():
    config = Config()
    obj = DetectedObject(0, np.array([400.0, 0, 20]), np.array([70.0, 40.0, 50.0]), 30.0, 100)
    for grasp in grasp_options(obj, config):
        np.testing.assert_allclose(grasp[:3, 2], [0, 0, -1], atol=1e-12)
        across = np.array([np.cos(np.radians(120)), np.sin(np.radians(120)), 0])
        assert abs(grasp[:3, 1] @ across) == pytest.approx(1)
        assert grasp[2, 3] == pytest.approx(obj.top_z - config.relocate.grasp_depth)
    assert grasp_options(DetectedObject(0, obj.center, np.array([120.0, 100.0, 50.0]), 0.0, 100), config) == []


def test_config_rejects_unknown_settings(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[views]\nstandof = 300\n")
    with pytest.raises(ValueError, match="views.standof"):
        load_config(path)
    path.write_text("[views]\nstandoff = 300\n")
    assert load_config(path).views.standoff == 300
