"""Measure a reconstructed mesh like calipers would, and compare it with the truth.

The truth is in ground_truth.toml (caliper measurements of the test objects).
Blocks are measured by width, length and height; cylinders by diameter and
height. Widths come from the middle half of the object's height, away from the
rounded top edge and the filled-in base.
"""

import tomllib
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

from .scene import min_area_rectangle


def load_truth(path: Path) -> dict:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def match_truth(truth: dict, color: str) -> str | None:
    """The ground-truth entry with this color name, if exactly one has it."""
    names = [name for name, entry in truth.items() if entry.get("color") == color]
    return names[0] if len(names) == 1 else None


def measure(mesh: o3d.geometry.TriangleMesh, shape: str) -> dict:
    """Height, plus width and length (box) or diameter (cylinder), of an upright object.

    The mesh's lowest point is taken as the table (closed meshes have a flat bottom there).
    """
    points = np.asarray(mesh.sample_points_uniformly(60000).points)
    bottom, top = points[:, 2].min(), np.percentile(points[:, 2], 99.9)
    height = float(top - bottom)
    band = points[(points[:, 2] > bottom + 0.25 * height) & (points[:, 2] < bottom + 0.75 * height)]
    result = {"height": height}
    if shape == "cylinder":
        def gaps(c):
            return np.hypot(band[:, 0] - c[0], band[:, 1] - c[1]) - c[2]
        fit = least_squares(gaps, [*band[:, :2].mean(axis=0), 15.0], loss="soft_l1", f_scale=0.5)
        result |= {"diameter": float(2 * fit.x[2]), "roundness": float(np.median(np.abs(gaps(fit.x))))}
    else:
        center, _, yaw = min_area_rectangle(band[:, :2])
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        local = (band[:, :2] - center) @ np.array([[c, -s], [s, c]])
        extents = np.percentile(local, 99, axis=0) - np.percentile(local, 1, axis=0)
        result |= {"length": float(extents.max()), "width": float(extents.min())}
    return result


def compare(measured: dict, truth: dict) -> list[tuple[str, float, float, float]]:
    """(dimension, measured, true, error) for each dimension the truth has.

    A block's horizontal sides are compared shorter with shorter ("width") and
    longer with longer ("length"), whichever way they were labeled when measured.
    """
    truth = dict(truth)
    if "width" in truth and "length" in truth:
        truth["width"], truth["length"] = sorted((truth["width"], truth["length"]))
    rows = []
    for key in ("diameter", "width", "length", "height"):
        if key in truth and key in measured:
            rows.append((key, measured[key], truth[key], measured[key] - truth[key]))
    return rows
