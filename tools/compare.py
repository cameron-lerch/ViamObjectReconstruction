''' Compare reconstructions of saved scans side by side.

Rebuild one scan several ways, or scans of the same object from different rigs, and
measure each the same way. Nothing talks to a robot.

    # Frames per view (scans made with scan.py --full-scan keep every raw frame)
    python tools/compare.py data/20261029-155544 --vary frames=1,5,10,50

    # Meshing and smoothing methods
    python tools/compare.py data/20261028-093620 --vary reconstruct.closing=flat_bottom,poisson,none
    python tools/compare.py data/20261028-093620 --vary reconstruct.smoothing=taubin,laplacian,none

    # Any setting, by its name in config.toml; several --vary give every combination
    python tools/compare.py data/20261028-093620 --vary reconstruct.align=true,false --vary reconstruct.edge_grow=0,2,4
    python tools/compare.py data/20261028-093620 --vary reconstruct.voxel=1.0,1.5,2.0

    # Number of views (spread over all directions), and camera mounts (calibrations)
    python tools/compare.py data/20261028-093620 --vary views=8,16,all
    python tools/compare.py data/20261029-155544 --vary mount=calibration/a.json,calibration/b.json

    # Two rigs: the same object scanned with each (labels from each session's rig name)
    python tools/compare.py data/<realsense scan> data/<orbbec scan>

Each scan's object 0 unless --object; ground truth found by size unless --truth (needed
for the printed sphere and bowl: printed_sphere, printed_bowl). Results go to
data/experiments/<time>-<name>/: results.csv and .json, summary.md (a table), figure.png,
and each variant's meshes.
'''

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon.config import load_config  # noqa: E402
from recon.experiment import SPECIAL, Variant, run  # noqa: E402
from recon.measure import load_truth  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rig_name(session: Path) -> str:
    ''' The rig a session came from (saved with it), or the folder's name for older sessions. '''
    record = session / "session.json"
    if record.exists():
        rig = json.loads(record.read_text()).get("config", {}).get("machine", {}).get("rig")
        if rig:
            return rig
    return session.name


def parse_vary(text: str) -> tuple[str, list[str]]:
    key, _, values = text.partition("=")
    if not key or not values:
        raise argparse.ArgumentTypeError(f"--vary needs KEY=VALUE,VALUE..., got {text!r}")
    return key.strip(), [v.strip() for v in values.split(",")]


def build_variants(sessions: list[Path], labels: list[str], varies: list[tuple[str, list[str]]],
                   object_id: int) -> list[Variant]:
    ''' Every session crossed with every combination of the varied values. '''
    variants = []
    combos = list(itertools.product(*[[(key, value) for value in values] for key, values in varies])) or [()]
    for session, session_label in zip(sessions, labels):
        for combo in combos:
            variant = Variant(label=session_label if len(sessions) > 1 else "", session=session, object_id=object_id)
            parts = [variant.label] if variant.label else []
            for key, value in combo:
                if key == "frames":
                    variant.frames = int(value)
                elif key == "views":
                    variant.views = None if value == "all" else int(value)
                elif key == "mount":
                    variant.mount = Path(value)
                    value = Path(value).stem
                else:
                    variant.settings[key] = value
                parts.append(f"{key.split('.')[-1]}={value}")
            variant.label = " ".join(parts) or session_label
            variants.append(variant)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=Path, help="session folders (data/<session>)")
    parser.add_argument("--vary", action="append", type=parse_vary, default=[], metavar="KEY=V1,V2",
                        help=f"what to compare: {', '.join(SPECIAL)}, or a setting such as reconstruct.voxel")
    parser.add_argument("--labels", help="names for the sessions, comma-separated (default: each session's rig)")
    parser.add_argument("--object", type=int, default=0, help="object id in each session (default 0)")
    parser.add_argument("--truth", help="ground_truth.toml entry (default: found by shape and size)")
    parser.add_argument("--name", default="", help="a name for the results folder")
    parser.add_argument("--config", default=str(ROOT / "config.toml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("viam").setLevel(logging.WARNING)

    labels = args.labels.split(",") if args.labels else [rig_name(s) for s in args.sessions]
    if len(labels) != len(args.sessions):
        parser.error("--labels needs one name per session")
    if len(set(labels)) < len(labels):  # same rig twice: tell them apart by session
        labels = [f"{label} {s.name}" for label, s in zip(labels, args.sessions)]
    variants = build_variants(args.sessions, labels, args.vary, args.object)
    name = args.name or "-".join(key.split(".")[-1] for key, _ in args.vary) or "rigs"
    out = ROOT / "data" / "experiments" / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    config = load_config(args.config)
    truth = load_truth(ROOT / "ground_truth.toml")
    if args.truth and args.truth not in truth:
        parser.error(f"No entry {args.truth!r} in ground_truth.toml")
    print(f"{len(variants)} variant(s); results in {out}")
    run(variants, config, out, truth, args.truth, ROOT)
    print("\n" + (out / "summary.md").read_text())
    print(f"Figure: {out / 'figure.png'}")


if __name__ == "__main__":
    main()
