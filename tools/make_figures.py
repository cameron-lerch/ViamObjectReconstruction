''' Make the README's result figures from saved sessions (offscreen; no windows).

    python tools/make_figures.py [--real data/<session>] [--real-object 1] [--synthetic data/demo-<...>]
    python tools/make_figures.py --only-latest   # just the figures of the latest real scans (fast)

Figures are SVGs (rendered panels plus labels) in media/images/; when cairosvg
is installed (pip install cairosvg) a PNG copy is written next to each.
Run `python scan.py progression ...` and `python scan.py demo --progression`
first: the progression meshes are read from disk.
'''

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon.config import Config, load_config  # noqa: E402
from recon.figures import PANEL, color_measured, comparison_figure, framing, progression_figure, row_figure  # noqa: E402
from recon.measure import compare, measure  # noqa: E402
from recon.pipeline import scanned_object  # noqa: E402
from recon.progression import measured_share  # noqa: E402
from recon.reconstruct import reconstruct  # noqa: E402
from recon.render import render  # noqa: E402
from recon.rgbd import load_views  # noqa: E402
from recon.session import Session  # noqa: E402

OUT = ROOT / "media" / "images"


def figure(name: str, title: str, panels: list[tuple[np.ndarray, str, str]], note: str = "") -> None:
    row_figure(OUT / f"{name}.svg", title, panels, note)
    print("wrote", OUT / f"{name}.svg")


def measured_figure(folder: Path, counts: list[int], elevation: float, azimuth: float) -> None:
    rows = {str(r["views"]): r for r in json.loads((folder / "summary.json").read_text())}
    center, radius = framing(o3d.io.read_triangle_mesh(str(folder / f"views_{max(counts):02d}" / "mesh.ply")))
    panels = []
    for k in counts:
        closed = o3d.io.read_triangle_mesh(str(folder / f"views_{k:02d}" / "mesh.ply"))
        observed = np.asarray(o3d.io.read_triangle_mesh(str(folder / f"views_{k:02d}" / "observed.ply")).vertices)
        if len(closed.triangles) == 0 or len(observed) == 0:
            continue
        for el, az, side in ((elevation, azimuth, "side"), (-55, azimuth, "underside")):
            panels.append((render(color_measured(closed, observed), PANEL, el, az, center, radius, use_colors=True, wire_boundary=False),
                           f"{k} view{'s' if k > 1 else ''}, {side}", f"{rows[str(k)]['measured_share']:.0%} measured, {rows[str(k)]['volume_cm3']:.1f}cm³"))
    figure("measured_vs_filled", "Closed meshes: measured surface (gray-blue) vs. filled in (orange)", panels,
           "The flat bottom is always filled in; with few views, so are most walls (straight down from the outline seen from above).")


def observed_vs_closed_figure(session: Session, object_id: int, name: str, title: str) -> None:
    ''' observed.obj and mesh.obj of one object, from the side and from below. '''
    folder = session.recon_dir(object_id)
    observed = o3d.io.read_triangle_mesh(str(folder / "observed.ply"))
    closed = o3d.io.read_triangle_mesh(str(folder / "mesh.ply"))
    center, radius = framing(closed)
    measured = np.asarray(observed.vertices)
    colored = color_measured(closed, measured)
    share = measured_share(closed, measured)  # of the surface area, as everywhere in the README
    panels = []
    for el, side in ((20, "side"), (-55, "below")):
        panels.append((render(observed, PANEL, el, -60, center, radius), f"observed.obj, {side}",
                       f"{len(observed.triangles):,} triangles, open (red edges)"))
    for el, side in ((20, "side"), (-55, "below")):
        panels.append((render(colored, PANEL, el, -60, center, radius, use_colors=True, wire_boundary=False), f"mesh.obj, {side}",
                       f"{len(closed.triangles):,} triangles, closed, {share:.0%} on measured data"))
    figure(name, title, panels, "observed.obj keeps only surface the camera measured; mesh.obj closes it (orange = filled in: the flat bottom and anything unseen).")


def setup_metrics(result, truth: dict) -> tuple[list[str], float]:
    ''' Metric lines for one setup's observed surface, and its mean size error (% of the
    calipers). Size error measures accuracy; how far single views lie from the fused
    surface measures consistency (do the views agree with each other?). '''
    # Measured on the observed surface (what the camera saw), from the table up: the
    # closed mesh of badly misaligned views can't always be built.
    obj = result.report["object"]
    table_z = obj["center"][2] - obj["size"][2] / 2
    rows = compare(measure(result.observed, truth["shape"], table_z), truth)
    lines = [f"{key}: {got:.1f}mm vs. {want:.1f}mm calipers ({error:+.1f}mm)" for key, got, want, error in rows]
    size_error = float(np.mean([abs(error) / want for _, _, want, error in rows]))
    lines.append(f"mean size error: {size_error:.1%}")
    lines.append(f"views disagree by {result.report['median_single_view_deviation_mm']:.2f}mm (median)")
    return lines, size_error


def calibration_figure(session: Session, object_id: int, truth_name: str = "green_cylinder") -> None:
    ''' The same captures reconstructed with the hand-entered camera mount and with the
    calibrated one, side by side, against the calipers. The same layout can take more
    columns, such as another camera's scan of the same object. '''
    logging.disable(logging.WARNING)
    config = load_config(ROOT / "config.toml")
    truth = tomllib.loads((ROOT / "ground_truth.toml").read_text())[truth_name]
    obj = scanned_object(session, object_id)
    _, plane = session.load_scene()
    views = load_views(session.views_dir(object_id))
    calibration = json.loads((session.root / "mount_calibration.json").read_text())
    swap = np.linalg.inv(np.array(calibration["original"])) @ np.array(calibration["calibrated"])
    calibrated = [type(v)(v.color, v.depth, v.T_world_cam @ swap, v.intrinsics, v.meta, v.name) for v in views]
    results = {"Uncalibrated camera": reconstruct(views, obj, plane, config),
               "Calibrated camera": reconstruct(calibrated, obj, plane, config)}
    # One framing for both, so they are drawn at the same scale.
    both = np.vstack([np.asarray(r.observed.vertices) for r in results.values()])
    center = both.mean(axis=0)
    radius = float(np.linalg.norm(both - center, axis=1).max())
    columns, errors = [], []
    for label, result in results.items():
        lines, size_error = setup_metrics(result, truth)
        errors.append(size_error)
        columns.append((label, [render(result.observed, PANEL, 20, az, center, radius) for az in (-60, 120)], lines))
    uncalibrated, calibrated_error = errors
    takeaway = (f"Uncalibrated, the mesh's size was off by {uncalibrated:.0%} on average; calibrated, by {calibrated_error:.0%}: "
                f"{uncalibrated / calibrated_error:.0f}x the error without calibration.")
    comparison_figure(OUT / "calibration_before_after.svg", f"Real cylinder, {len(views)} views: uncalibrated vs. calibrated camera",
                      columns, takeaway,
                      "Observed surfaces (what the camera measured). Same captures, alignment and fusion; only the camera mount differs "
                      "(21mm / 2.2deg apart). Uncalibrated is the mount entered by hand in the Viam app.")
    print("wrote", OUT / "calibration_before_after.svg")


def alignment_figure(old_session: Session) -> None:
    logging.disable(logging.WARNING)
    objects, plane = old_session.load_scene()
    obj = objects[1]
    old = o3d.io.read_triangle_mesh(str(old_session.recon_dir(1) / "mesh.ply"))
    new = reconstruct(load_views(old_session.views_dir(1)), obj, plane, Config())
    center, radius = framing(new.mesh)
    panels = [(render(old, PANEL, 15, az, center, radius, wire_boundary=False), "pairwise ICP + pose graph", "first approach")
              for az in (-60, 60)]
    panels += [(render(new.observed, PANEL, 15, az, center, radius, wire_boundary=False), "joint ICP with arm prior", "current approach")
               for az in (-60, 60)]
    figure("alignment_before_after", "Same 37 synthetic views, two ways of aligning them", panels,
           "Misaligned views average into ripples and double walls; well-aligned ones fuse into a smooth surface.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", default="data/20261026-143006")
    parser.add_argument("--real-object", type=int, default=1)
    parser.add_argument("--synthetic", default="data/demo-20261027-113147")
    parser.add_argument("--old-synthetic", default="data/demo-20261026-130458", help="a demo built with the first alignment approach")
    parser.add_argument("--latest", default="data/20261028-093620", help="the scans in README section 3.1 (block 0, cylinder 1)")
    parser.add_argument("--only-latest", action="store_true", help="only the figures of --latest (no reconstruction runs)")
    parser.add_argument("--only-calibration", action="store_true", help="only the uncalibrated vs. calibrated figure")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only_calibration:
        calibration_figure(Session(ROOT / args.real), args.real_object)
        return
    latest = Session(ROOT / args.latest)
    observed_vs_closed_figure(latest, 0, "observed_vs_closed_block", "Orange block, 34 views: observed.obj vs. mesh.obj")
    observed_vs_closed_figure(latest, 1, "observed_vs_closed_cylinder", "Green cylinder, 34 views: observed.obj vs. mesh.obj")
    if args.only_latest:
        return
    real = Session(ROOT / args.real)
    real_folder = real.recon_dir(args.real_object) / "progression"
    progression_figure(real_folder, OUT / "progression_real.svg", "Real cylinder: one view vs. many", [1, 3, 8, 25])
    measured_figure(real_folder, [1, 8, 25], 20, -60)
    synthetic = Session(ROOT / args.synthetic)
    progression_figure(synthetic.recon_dir(1) / "progression", OUT / "progression_synthetic.svg",
                       "Synthetic cylinder (known shape): one view vs. many", [1, 3, 8, 37])
    calibration_figure(real, args.real_object)
    alignment_figure(Session(ROOT / args.old_synthetic))


if __name__ == "__main__":
    main()
