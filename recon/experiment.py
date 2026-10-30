''' Compare reconstructions of saved scans: the same views rebuilt in several variants
(frames per view, camera mount, number of views, any setting), or scans of the same
object from different rigs, each measured the same way. Used by tools/compare.py.

A variant changes one or more of:
- `frames`: rebuild each view's depth from its first k raw frames (scans made with
  `scan.py --full-scan` keep all of them);
- `mount`: re-place every view with another camera mount (a calibration to compare);
- `views`: use only k views, spread over all directions (as `--max-views` would);
- any setting, by its name in config.toml, e.g. `reconstruct.voxel` or `reconstruct.closing`.

Each result is measured by: how far single views lie from the fused surface (views
disagree), roughness, volume and the share measured, size against the calipers, and,
for the 3D-printed objects, the surface error against their CAD model.
'''

import copy
import csv
import json
import logging
import time
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from .cad import compare_with_cad, error_colors, load_cad
from .calibrate import apply_mount
from .config import Config
from .measure import compare, identify, measure
from .pipeline import apply_session_calibration, scanned_object
from .progression import measured_share, roughness, spread_order
from .reconstruct import reconstruct
from .rgbd import load_views
from .scene import DetectedObject, TablePlane
from .session import Session

log = logging.getLogger(__name__)

SPECIAL = ("frames", "mount", "views")  # variant keys that aren't settings


@dataclass
class Variant:
    label: str
    session: Path
    object_id: int = 0
    frames: int | None = None
    mount: Path | None = None
    views: int | None = None
    settings: dict[str, object] = field(default_factory=dict)


def parse_value(text: str, current):
    ''' A command-line value, as the type of the setting it replaces. '''
    if isinstance(current, bool):
        if text.lower() in ("true", "on", "yes", "1"):
            return True
        if text.lower() in ("false", "off", "no", "0"):
            return False
        raise ValueError(f"Expected true or false, got {text!r}")
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, list):
        return json.loads(text)
    return text


def set_setting(config: Config, key: str, value) -> None:
    ''' Set a dotted setting, e.g. "reconstruct.voxel", checking that it exists. '''
    target, *path = config, *key.split(".")
    for name in path[:-1]:
        target = getattr(target, name, None)
        if not is_dataclass(target):
            raise ValueError(f"Unknown setting {key!r}")
    name = path[-1]
    if name not in {f.name for f in fields(target)}:
        raise ValueError(f"Unknown setting {key!r}")
    current = getattr(target, name)
    setattr(target, name, parse_value(value, current) if isinstance(value, str) else value)


def variant_config(config: Config, variant: Variant) -> Config:
    variant_config = copy.deepcopy(config)
    for key, value in variant.settings.items():
        set_setting(variant_config, key, value)
    return variant_config


def read_mount(path) -> np.ndarray:
    return np.array(json.loads(Path(path).read_text())["mount"])


def variant_views(variant: Variant):
    ''' The variant's views, object and table. '''
    session = Session(Path(variant.session))
    _, plane = session.load_scene()
    obj = scanned_object(session, variant.object_id)
    views = load_views(session.views_dir(variant.object_id), variant.frames)
    if not views:
        raise ValueError(f"No views of object {variant.object_id} in {session.root}")
    if variant.mount is not None:
        # From the poses as captured (with the mount each view records), not on top of a
        # session's own mount correction.
        new = read_mount(variant.mount)
        views = [apply_mount([v], np.array(v.meta["camera_mount"]), new)[0] for v in views]
    else:
        views = apply_session_calibration(session, views)
    if variant.views is not None and variant.views < len(views):
        keep = sorted(spread_order(views, obj.center)[:variant.views])
        views = [views[i] for i in keep]
    return views, obj, plane


def evaluate(result, truth_name: str | None, truth: dict, root: Path) -> tuple[dict, np.ndarray | None]:
    ''' The measurements of one rebuilt scan (and the CAD error per observed vertex, if any). '''
    report, mesh, observed = result.report, result.mesh, result.observed
    registration = report.get("registration", {}).get("views", [])
    row = {
        "views": report["views"],
        "views_disagree_mm": report["median_single_view_deviation_mm"],
        "alignment_rejected": sum(v["rejected"] for v in registration),
        "roughness_mm": roughness(np.asarray(observed.vertices)) if len(observed.vertices) else None,
        "observed_triangles": len(observed.triangles),
        "closed": bool(report["mesh"]["closed"]),
        "volume_cm3": report["mesh"]["volume_mm3"] / 1000 if report["mesh"]["volume_mm3"] else None,
        # Of a closed mesh only: an open (observed) mesh is all measured by definition.
        "measured_share": (measured_share(mesh, np.asarray(result.cloud.points))
                           if report["mesh"]["closed"] and len(result.cloud.points) else None),
    }
    obj = report["object"]
    table_z = obj["center"][2] - obj["size"][2] / 2
    if truth_name is None and len(mesh.triangles):
        truth_name, _, _ = identify(mesh, truth)
    row["truth"] = truth_name
    errors = None
    if truth_name:
        entry = truth[truth_name]
        if entry.get("shape") in ("box", "cylinder", "sphere", "bowl"):
            # The closed mesh when there is one (as calipers see it); else the measured
            # surface, from the table up.
            measured = (measure(mesh, entry["shape"]) if row["closed"]
                        else measure(observed, entry["shape"], table_z))
            rows = compare(measured, entry)
            for key, got, _, error in rows:
                row[f"{key}_mm"], row[f"{key}_error_mm"] = got, error
            if rows:
                row["mean_size_error_pct"] = 100 * float(np.mean([abs(e) / w for _, _, w, e in rows]))
        if entry.get("cad") and len(observed.vertices):
            cad = load_cad(root / entry["cad"])
            plane = TablePlane(np.array([0.0, 0.0, 1.0]), -table_z)
            metrics, errors, _ = compare_with_cad(cad, observed, DetectedObject.from_dict(obj), plane)
            row |= metrics
    return row, errors


# The figure's columns show these, in order, when present.
FIGURE_METRICS = [("cad_median_mm", "CAD error {:.2f}mm (median)"), ("cad_bias_mm", "CAD bias {:+.2f}mm"),
                  ("mean_size_error_pct", "size error {:.1f}%"), ("views_disagree_mm", "views disagree {:.2f}mm"),
                  ("roughness_mm", "roughness {:.3f}mm"), ("measured_share", "{:.0%} measured")]
# The takeaway ranks variants by the first of these that every row has (lower is better).
RANKING = [("cad_median_mm", "error against the CAD model", "{:.2f}mm"), ("mean_size_error_pct", "size error", "{:.1f}%"),
           ("views_disagree_mm", "how much the views disagree", "{:.2f}mm")]


def run(variants: list[Variant], config: Config, out: Path, truth: dict, truth_name: str | None = None,
        root: Path = Path(".")) -> list[dict]:
    ''' Rebuild and measure every variant; write meshes, results.csv/json, summary.md and
    figure.svg/png into `out`. Every variant is measured against the same ground truth:
    `truth_name`, or else the entry the first session's saved mesh resembles, or else the
    one the first variant's mesh resembles (per variant, a poor variant could match a
    different object). '''
    out.mkdir(parents=True, exist_ok=True)
    if truth_name is None and variants:
        truth_name = saved_truth(variants[0], truth)
        if truth_name:
            log.info("Comparing every variant with %s (what %s's saved mesh resembles)", truth_name, variants[0].session)
    rows, renders = [], []
    for variant in variants:
        log.info("\n== %s ==", variant.label)
        started = time.monotonic()
        views, obj, plane = variant_views(variant)
        result = reconstruct(views, obj, plane, variant_config(config, variant))
        row, errors = evaluate(result, truth_name, truth, root)
        if truth_name is None and row.get("truth"):
            truth_name = row["truth"]
            log.info("Comparing every variant with %s (what the first variant resembles)", truth_name)
        row = {"label": variant.label, "session": str(variant.session), "object": variant.object_id,
               "frames": variant.frames, "mount": str(variant.mount) if variant.mount else None,
               "settings": json.dumps(variant.settings) if variant.settings else "", **row,
               "seconds": round(time.monotonic() - started, 1)}
        rows.append(row)
        folder = out / safe_name(variant.label)
        folder.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(folder / "mesh.ply"), result.mesh, write_ascii=False)
        o3d.io.write_triangle_mesh(str(folder / "observed.ply"), result.observed, write_ascii=False)
        (folder / "report.json").write_text(json.dumps(result.report, indent=2))
        renders.append((variant.label, result.mesh, result.observed, errors))
        log.info("  %s", ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                   for k, v in row.items() if k not in ("label", "session", "settings")))
    write_results(rows, out)
    write_figure(rows, renders, out)
    return rows


def saved_truth(variant: Variant, truth: dict) -> str | None:
    ''' The ground-truth entry the session's own final mesh resembles, if it has one. '''
    path = Session(Path(variant.session)).recon_dir(variant.object_id) / "mesh.ply"
    if not path.exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(path))
    return identify(mesh, truth)[0] if len(mesh.triangles) else None


def safe_name(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.=" else "_" for c in label)[:80]


def write_results(rows: list[dict], out: Path) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    shown = ["label", "views", "views_disagree_mm", "roughness_mm", "measured_share", "mean_size_error_pct",
             "cad_median_mm", "cad_bias_mm", "seconds"]
    shown = [k for k in shown if any(row.get(k) is not None for row in rows)]
    lines = ["| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(k)) for k in shown) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def format_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def takeaway(rows: list[dict]) -> str:
    for key, name, form in RANKING:
        if all(row.get(key) is not None for row in rows):
            best = min(rows, key=lambda row: row[key])
            return f"Best by {name}: {best['label']} ({form.format(best[key])})"
    return "No measurement every variant has in common"


def write_figure(rows: list[dict], renders: list, out: Path) -> None:
    ''' One column per variant: its final mesh, plus (with a CAD model) its measured surface
    colored by the error against it (red outside, blue inside), and its main measurements. '''
    from .figures import PANEL, comparison_figure
    from .render import render
    meshes = [m for _, mesh, observed, _ in renders for m in (mesh, observed) if len(m.vertices)]
    if not meshes:
        return
    both = np.vstack([np.asarray(m.vertices) for m in meshes])
    center = both.mean(axis=0)
    radius = float(np.linalg.norm(both - center, axis=1).max())
    columns = []
    for row, (label, mesh, observed, errors) in zip(rows, renders):
        images = [render(mesh if len(mesh.triangles) else observed, PANEL, 20, -60, center, radius)]
        if errors is not None:
            colored = o3d.geometry.TriangleMesh(observed)
            colored.vertex_colors = o3d.utility.Vector3dVector(error_colors(errors))
            images.append(render(colored, PANEL, 20, -60, center, radius, use_colors=True))
        lines = [template.format(row[key]) for key, template in FIGURE_METRICS if row.get(key) is not None]
        columns.append((label, images, lines[:4]))
    note = "Each variant's final mesh" + ("; beside it, its measured surface against the CAD model: red outside, "
                                          "blue inside (3mm = full color)" if any(r[3] is not None for r in renders) else "")
    comparison_figure(out / "figure.svg", f"Comparison: {out.name}", columns, takeaway(rows), note)
