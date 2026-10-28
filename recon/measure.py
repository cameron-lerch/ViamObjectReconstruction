''' Measure a reconstructed mesh like calipers would, and compare it with the truth.

The truth is in ground_truth.toml (caliper measurements of the test objects).
Blocks are measured by width, length and height; cylinders by diameter and
height. Widths come from the middle half of the object's height, away from the
rounded top edge and the filled-in base.
'''

import tomllib
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

from .scene import min_area_rectangle


def load_truth(path: Path) -> dict:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def measure(mesh: o3d.geometry.TriangleMesh, shape: str) -> dict:
    ''' Height, plus width and length (box) or diameter (cylinder), of an upright object.

    The mesh's lowest point is taken as the table (closed meshes have a flat bottom there).
    '''
    # 60000 surface samples: dense enough for fits to a few hundredths of a mm on these
    # objects. The top is the 99.9th percentile so a stray triangle can't raise it.
    points = np.asarray(mesh.sample_points_uniformly(60000).points)
    bottom, top = points[:, 2].min(), np.percentile(points[:, 2], 99.9)
    height = float(top - bottom)
    band = points[(points[:, 2] > bottom + 0.25 * height) & (points[:, 2] < bottom + 0.75 * height)]
    result = {"height": height}
    if shape == "cylinder":
        def gaps(c):
            return np.hypot(band[:, 0] - c[0], band[:, 1] - c[1]) - c[2]
        # Start from a 15mm radius (the test cylinder's); the robust loss down-weights
        # points more than about 0.5mm off the circle.
        fit = least_squares(gaps, [*band[:, :2].mean(axis=0), 15.0], loss="soft_l1", f_scale=0.5)
        result |= {"diameter": float(2 * fit.x[2]), "fit_mm": float(np.median(np.abs(gaps(fit.x))))}
    else:
        center, _, yaw = min_area_rectangle(band[:, :2])
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        local = (band[:, :2] - center) @ np.array([[c, -s], [s, c]])
        # Like calipers on the faces, ignoring the outermost 1% of points on each side (stray points).
        low, high = np.percentile(local, 1, axis=0), np.percentile(local, 99, axis=0)
        extents = high - low
        middle, half = (high + low) / 2, extents / 2
        # Distance from each point to the nearest side of the rectangle.
        to_side = np.min(np.abs(half - np.abs(local - middle)), axis=1)
        result |= {"length": float(extents.max()), "width": float(extents.min()), "fit_mm": float(np.median(to_side))}
    return result


def identify(mesh: o3d.geometry.TriangleMesh, truth: dict, tolerance_mm: float = 5.0,
             tolerance_share: float = 0.15) -> tuple[str | None, dict, str]:
    ''' Which ground-truth object (if any) this mesh looks like: (name, measurements, why).

    The shape is whichever of block and cylinder fits the mesh's middle band
    more closely; then every measured dimension of a truth entry with that
    shape must agree within `tolerance_mm` or `tolerance_share`, whichever is
    larger (loose: the real meshes come out about 3mm wider than the calipers
    say, which is what the comparison is meant to show). Of several matches (two blocks of nearly the same size) the closest
    wins; anything else (a pumpkin, a bottle with no entry) gets None.
    '''
    fits = {shape: measure(mesh, shape) for shape in ("box", "cylinder")}
    shape = min(fits, key=lambda name: fits[name]["fit_mm"])
    measured = fits[shape]
    best, best_error = None, np.inf
    for name, entry in truth.items():
        if entry.get("shape") != shape:
            continue
        rows = compare(measured, entry)
        if rows and all(abs(error) <= max(tolerance_mm, tolerance_share * true) for _, _, true, error in rows):
            total = sum(abs(error) for *_, error in rows)
            if total < best_error:
                best, best_error = name, total
    size = " x ".join(f"{measured[k]:.1f}" for k in ("diameter", "width", "length", "height") if k in measured)
    why = f"looks like a {'cylinder' if shape == 'cylinder' else 'block'}, {size}mm"
    return best, measured, why


def compare(measured: dict, truth: dict) -> list[tuple[str, float, float, float]]:
    ''' (dimension, measured, true, error) for each dimension the truth has.

    A block's horizontal sides are compared shorter with shorter ("width") and
    longer with longer ("length"), whichever way they were labeled when measured.
    '''
    truth = dict(truth)
    if "width" in truth and "length" in truth:
        truth["width"], truth["length"] = sorted((truth["width"], truth["length"]))
    rows = []
    for key in ("diameter", "width", "length", "height"):
        if key in truth and key in measured:
            rows.append((key, measured[key], truth[key], measured[key] - truth[key]))
    return rows
