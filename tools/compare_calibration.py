''' Compare hand-eye calibration methods on saved board runs (no robot needed).

    python tools/compare_calibration.py                          # every data/board-* run
    python tools/compare_calibration.py data/board-20261029-150236

Our solver and the five OpenCV methods (Tsai is what Viam's camera-calibration module
uses) each solve the camera mount from the same board views, and are scored by leave-
one-out reprojection, Viam's consistency residuals, and distance from our mount (see
recon/handeye.py). Each method's mount is saved, so a scan can be rebuilt with it:

    python tools/compare.py data/<scan> --vary mount=data/experiments/<this>/mounts/<run>/ours.json,...

Results: data/experiments/<time>-calibration/results.csv, summary.md, mounts/.
'''

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon.handeye import METHODS, compare_methods  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["run", "method", "views", "fit_px", "held_out_px_median", "held_out_px_max", "board_spread_mm",
           "board_turn_deg", "from_ours_mm", "from_ours_deg"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="board runs (default: every data/board-* with board views)")
    parser.add_argument("--methods", default=",".join(METHODS), help=f"comma-separated, from {', '.join(METHODS)}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    runs = args.runs or sorted(p.parent.parent for p in (ROOT / "data").glob("board-*/board/board_calibration.json"))
    out = ROOT / "data" / "experiments" / f"{time.strftime('%Y%m%d-%H%M%S')}-calibration"
    rows = []
    for run in runs:
        print(f"{run.name} ...", flush=True)
        try:
            run_rows, mounts = compare_methods(run / "board", args.methods.split(","))
        except ValueError as problem:
            print(f"  skipped: {problem}")
            continue
        (out / "mounts" / run.name).mkdir(parents=True, exist_ok=True)
        for method, mount in mounts.items():
            (out / "mounts" / run.name / f"{method}.json").write_text(json.dumps(
                {"mount": mount.tolist(), "source": f"{method} on {run.name}"}, indent=2))
        rows += [{"run": run.name, **row} for row in run_rows]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[k]:.2f}" if isinstance(row.get(k), float) else str(row.get(k, "-"))
                                       for k in COLUMNS) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nResults and each method's mount: {out}")


if __name__ == "__main__":
    main()
