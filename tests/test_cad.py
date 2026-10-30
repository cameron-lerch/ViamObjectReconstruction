''' The CAD comparison: the fit finds a moved model, and a too-big scan shows a positive bias. '''

import numpy as np
import pytest

from recon.cad import compare_with_cad, load_cad, signed_distances
from recon.scene import DetectedObject, TablePlane

BLOCK = "calibration/printed_objects/calibration_block/calibration_block.stl"
TABLE = TablePlane(np.array([0.0, 0.0, 1.0]), 0.0)


def scanned(scale=1.0, yaw_deg=33.0, shift=(400.0, -300.0)):
    ''' What a scan of the block sees: its sides and top (never the bottom), as a mesh. '''
    import open3d as o3d
    cad = load_cad(BLOCK)
    mesh = o3d.geometry.TriangleMesh(cad).subdivide_midpoint(4)
    v = np.asarray(mesh.vertices) - [20.0, 20.0, 0.0]
    v *= scale
    a = np.radians(yaw_deg)
    v = v @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]).T + [*shift, 0.0]
    mesh.vertices = o3d.utility.Vector3dVector(v)
    keep = v[:, 2] > 2.0
    mesh.remove_vertices_by_mask(~keep)
    return mesh


def test_fit_finds_the_moved_block_and_measures_no_error():
    mesh = scanned()
    obj = DetectedObject(0, np.array([403.0, -298.0, 30.0]), np.array([56.0, 56.0, 60.0]), 0.0, 100)
    metrics, errors, _ = compare_with_cad(load_cad(BLOCK), mesh, obj, TABLE)
    assert metrics["cad_median_mm"] < 0.2 and abs(metrics["cad_bias_mm"]) < 0.2


def test_a_scan_five_percent_too_big_has_a_positive_bias():
    mesh = scanned(scale=1.05)
    obj = DetectedObject(0, np.array([400.0, -300.0, 31.0]), np.array([59.0, 59.0, 63.0]), 0.0, 100)
    metrics, _, _ = compare_with_cad(load_cad(BLOCK), mesh, obj, TABLE)
    assert metrics["cad_bias_mm"] > 0.5


def test_signed_distance_is_positive_outside():
    cad = load_cad(BLOCK)
    d = signed_distances(cad, np.array([[20.0, 20.0, 70.0], [20.0, 20.0, 30.0]]), np.eye(4))
    assert d[0] == pytest.approx(10.0, abs=0.01) and d[1] < 0
