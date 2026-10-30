''' tools/compare.py: variants of a saved scan rebuilt and measured the same way. '''

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from recon import synthetic
from recon.config import Config
from recon.experiment import Variant, run, set_setting
from recon.rgbd import save_frames, save_view
from recon.scene import DetectedObject, TablePlane
from recon.session import Session
from recon.views import generate_candidates, order_for_travel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import compare  # noqa: E402

OBJ = DetectedObject(0, np.array([431.0, -168.0, 37.0]), np.array([52.0, 51.0, 92.0]), 0.0, 1000)
TRUTH = {"test_cylinder": {"shape": "cylinder", "diameter": 50.0, "height": 90.0}}


def saved_scan(tmp_path, frames=3) -> Path:
    ''' A synthetic 13-view scan of a 50 x 90mm cylinder, saved like a real full scan. '''
    config = Config()
    config.views.azimuth_step_deg = 60
    config.views.elevations_deg = [25.0, 50.0]
    scene = synthetic.SyntheticScene([synthetic.SceneObject(synthetic.cylinder(430, -170, 25, 90), (0.1, 0.6, 0.25))])
    poses = [c.pose() for c in order_for_travel(generate_candidates(OBJ, config.views))]
    root = tmp_path / "scan"
    root.mkdir()
    session = Session(root)
    session.save_scene([OBJ], TablePlane(np.array([0.0, 0.0, 1.0]), -synthetic.TABLE_Z))
    session.write_json(session.object_dir(0) / "scanned_object.json", OBJ.to_dict())
    rng = np.random.default_rng(0)
    for index, view in enumerate(synthetic.render_views(scene, poses, seed=4)):
        folder = session.views_dir(0) / f"view_{index:03d}"
        save_view(folder, view.color, view.depth, view.T_world_cam, view.intrinsics,
                  {"frames": frames, "camera_mount": np.eye(4).tolist()})
        noisy = [np.where(view.depth > 0, view.depth + rng.integers(-1, 2, view.depth.shape), 0).astype(np.uint16)
                 for _ in range(frames)]
        save_frames(folder / "frames", [b"jpeg"] * frames, noisy)
    return root


def test_settings_are_set_by_name_and_typed():
    config = Config()
    set_setting(config, "reconstruct.voxel", "2.0")
    set_setting(config, "reconstruct.align", "false")
    set_setting(config, "reconstruct.closing", "none")
    assert config.reconstruct.voxel == 2.0 and config.reconstruct.align is False and config.reconstruct.closing == "none"
    with pytest.raises(ValueError):
        set_setting(config, "reconstruct.no_such_setting", "1")


def test_variants_cross_every_value_and_label_themselves():
    variants = compare.build_variants([Path("data/a")], ["a"], [("frames", ["1", "5"]), ("reconstruct.closing", ["none", "poisson"])], 0)
    assert [v.label for v in variants] == ["frames=1 closing=none", "frames=1 closing=poisson",
                                          "frames=5 closing=none", "frames=5 closing=poisson"]
    assert variants[3].frames == 5 and variants[3].settings == {"reconstruct.closing": "poisson"}
    two = compare.build_variants([Path("data/a"), Path("data/b")], ["realsense", "orbbec"], [], 0)
    assert [v.label for v in two] == ["realsense", "orbbec"]


def test_rebuilds_variants_of_a_saved_scan_and_writes_results(tmp_path):
    scan = saved_scan(tmp_path)
    variants = [Variant("frames=1", scan, frames=1), Variant("frames=3 open", scan, frames=3, settings={"reconstruct.closing": "none"}),
                Variant("half the views", scan, views=6)]
    out = tmp_path / "results"
    rows = run(variants, Config(), out, TRUTH, "test_cylinder")
    assert [r["label"] for r in rows] == ["frames=1", "frames=3 open", "half the views"]
    assert rows[0]["closed"] and not rows[1]["closed"] and rows[2]["views"] == 6
    assert abs(rows[0]["diameter_error_mm"]) < 3 and rows[0]["mean_size_error_pct"] < 6
    for name in ("results.csv", "results.json", "summary.md", "figure.svg"):
        assert (out / name).exists()
    assert (out / "frames_1" / "mesh.ply").exists() or (out / "frames=1" / "mesh.ply").exists()
    assert len(json.loads((out / "results.json").read_text())) == 3


def test_a_different_mount_moves_every_view(tmp_path):
    scan = saved_scan(tmp_path, frames=1)
    shifted = np.eye(4)
    shifted[0, 3] = 5.0
    mount = tmp_path / "mount.json"
    mount.write_text(json.dumps({"mount": shifted.tolist()}))
    from recon.experiment import variant_views
    plain, _, _ = variant_views(Variant("a", scan))
    moved, _, _ = variant_views(Variant("b", scan, mount=mount))
    offsets = [np.linalg.norm(m.T_world_cam[:3, 3] - p.T_world_cam[:3, 3]) for p, m in zip(plain, moved)]
    assert np.allclose(offsets, 5.0)


def test_incidence_is_zero_head_on_and_large_on_a_steep_plane():
    from recon.rgbd import Intrinsics, incidence_deg
    lens = Intrinsics(width=200, height=100, fx=200.0, fy=200.0, cx=100.0, cy=50.0)
    flat = np.full((100, 200), 400, np.uint16)
    assert np.median(incidence_deg(flat, lens)[10:-10, 10:-10]) < 15  # off-axis pixels see it slightly tilted
    v, u = np.mgrid[0:100, 0:200]
    x = (u - 100) / 200.0
    steep = (400 / (1 - 3.0 * x)).clip(0, 3000).astype(np.uint16)  # a plane at atan(3) = 71.6deg to the optical axis
    # Down the image's center column the line of sight is the optical axis.
    assert np.median(incidence_deg(steep, lens)[20:-20, 98:102]) == pytest.approx(71.6, abs=2.0)
