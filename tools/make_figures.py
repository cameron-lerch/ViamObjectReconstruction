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
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon.config import Config, load_config  # noqa: E402
from recon.figures import PANEL, color_measured, framing, progression_figure, row_figure  # noqa: E402
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


def calibration_figure(session: Session, object_id: int) -> None:
    logging.disable(logging.WARNING)
    config = load_config(ROOT / "config.toml")
    obj = scanned_object(session, object_id)
    _, plane = session.load_scene()
    views = load_views(session.views_dir(object_id))
    calibration = json.loads((session.root / "mount_calibration.json").read_text())
    swap = np.linalg.inv(np.array(calibration["original"])) @ np.array(calibration["calibrated"])
    calibrated = [type(v)(v.color, v.depth, v.T_world_cam @ swap, v.intrinsics, v.meta, v.name) for v in views]
    before = reconstruct(views, obj, plane, config)
    after = reconstruct(calibrated, obj, plane, config)
    center, radius = framing(after.mesh)
    panels = []
    for label, result in (("configured camera mount", before), ("calibrated camera mount", after)):
        deviation = result.report["median_single_view_deviation_mm"]
        for az in (-60, 120):
            panels.append((render(result.observed, PANEL, 20, az, center, radius), label,
                           f"views deviate {deviation:.2f}mm (median)"))
    figure("calibration_before_after", "Real cylinder, 25 views: camera mount as configured vs. calibrated", panels,
           "Same 25 captures, same alignment and fusion; only the camera mount differs (21mm / 2.2° apart).")


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
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
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
