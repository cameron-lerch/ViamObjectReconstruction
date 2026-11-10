# Changelog

How ViamObjectReconstruction got to where it is: notable changes and the history behind them, moved out of the README so that the README describes the system as it is now. Newest first.

## 2026-11-09

- **Mesh names:** `observed.obj` is now `open_mesh.obj` and `mesh.obj` is `watertight_mesh.obj` (and the same for `.ply`), in the code, the figures, the viewer and every saved scan.
- **Measured share:** now measured against the open mesh, as the figures color it. It used the raw fused point cloud, which still holds small fragments the open mesh drops, so it overstated the measured share with few views (the 8-view cylinder: 76% → 63%). The full scans changed by 1-2 points.
- **Progression figure and video:** the figure has three rows (cumulative point cloud, open mesh, watertight mesh) across 1, 8, 16, 32 and all views, and every scan also saves `progression_vid.mp4` (and `.gif`): the color image, the growing point cloud and the growing mesh, view by view.
- **README:** restructured. The analyses moved into §6 "Tuning our Approach", the synthetic results were removed, closing the mesh is in one place (§5.8), and the viewer has its own section (§9.5). This changelog took the dated history.
- **The 850 again,** now with a gripper, for the 320mm full scans.

## 2026-11-05

- **The xArm6 (armfarm12):** the first runs on a second arm, which led to most of the changes below.
- **Calibration starts fresh every time** (`calibration.py`): from wherever the arm is, with nothing saved by an earlier calibration.
- **A home position found above the mat** replaced the typed-in ready pose (`[machine] ready_camera_mm`, the camera at (505, -400, 322) on armfarm14), which was out of the xArm6's reach and in another cell's coordinates.
- **The survey follows the table from where the camera is:** a first look in place, a turn and rise over the same spot, then outward in steps, stopping at the table's edges, the arm's reach and Viam's obstacles. It replaced a fixed grid over a typed-in region (`x_range`, `y_range`), and its height went from 600mm to 500mm (spacing 340mm → 280mm), since 600mm was out of the xArm6's reach.
- **Rough calibration in place:** when the mount in use can't be trusted (a new arm), the camera first turns a little over the mat, 8-10° each way, to solve a rough mount before any larger move. The first attempt started from a copy of armfarm14's mount, which was about 180° off on the xArm6.
- **Camera collision boxes** only come from a calibrated mount; boxes placed by a guessed mount overlapped the arm's wrist and blocked every move.
- **The touch-off's travel height** went from 600mm to 300mm, out of the xArm6's reach with the gripper pointing down.
- **`align_color_depth: true`** was missing from the xArm6's camera: its depth image was 60-130 pixels off the color image, so objects were found beside where they were.
- **The low-view question removed:** the scan no longer waits for the operator before the 10° ring.

## 2026-11-04

- **Rings:** staggered as before, but a ring is turned by half a step when the arm's base blocks one of its views and that reaches more views.
- **The bowl reprint** (a less shiny filament) replaced the shiny bowl in the README, and its caliper measurements went into `ground_truth.toml`.
- **The camera is a D435,** not a D435i, everywhere it was named.
- **Datasets** are named `<session>-<object>-<standoff>`.

## 2026-11-03

- **The scan distance (standoff)** went from 280mm to 320mm, so every surface stays beyond the D435's 0.3m minimum.
- **HC Laplacian smoothing** on both meshes became the default, after comparing seven methods.
- **Deadlines on every call to the robot,** after `viam-server` restarted mid-scan and the scan waited forever.

## 2026-10-30

- **The ready pose became one planned move** through Viam's motion service, without a prompt. Until then it went one joint group at a time (forearm roll, wrist, shoulder and elbow, then base), each step confirmed at the terminal.

## 2026-10-29

- **The gripper came off** the 850, so that the camera could reach more views.
- **The fast move:** after the gripper came off, the arm had been moved by hand into a pose with joint 4 at 290°, and the calibration's first planned move turned it 290° at once, very fast, until the xArm controller stopped it. From then on, every run that moves the arm starts at a ready pose (the camera about 322mm over the mat, looking down).

## Text replaced in the README

*Earlier README text that code changes replaced, kept for reference (until 2026-11-09 the README's "Temp Section - Review"). Newest first; each entry says what replaced it.*

### 2026-10-30: the live preview's table (§9.5)

Replaced by cropping each view at its own table fit:

> | Where the table is (pixels just above it are left out) | The survey's table | Each view's own fit of the table nearby |

### 2026-10-30: the ready pose (§10, §8.1, §9)

Replaced by one planned move through Viam's motion service, without a prompt, so a `scan.py` run only asks which objects to scan; the start-up "Start? [y/N]" question was removed too. Old text:

> - **Every run that moves the arm starts at a ready pose.** It reads the arm's joints, and if they aren't already at the ready pose (camera about 320mm over the mat, looking down; `[machine] ready_joints_deg`), steps there one joint group at a time: forearm roll, wrist, shoulder and elbow, then base, each shown and confirmed at the terminal. See the free-move incident in [§8.1](#81-hardware-and-platform).

### 2026-10-30: candidates near the arm's base, and the box (§5.1)

Replaced after two pumpkin surveys in a row rejected the pumpkin as "not in open table": the base cut-off at 350mm removed its near third, and the leftover part counted against it in the ring. Old text:

> 3. **Candidates:** keep points 8-500mm above the table, inside the search region, and away from the arm's base; average them into 2mm voxels; remove statistical outliers.

> 5. **Boxes:** for each cluster, the smallest rectangle around its footprint (rotating calipers on its convex hull) gives the center, length, width and heading; the height runs from the table to the 98th-percentile top.

### 2026-10-29: uncalibrated vs. calibrated table (now §6.4)

Replaced after the closing changes of 2026-10-28 (walls from the lowest seen surface, table scraps, the balloon check): the numbers below were measured on closed meshes, which the uncalibrated views can't reliably produce any more. The new table measures the observed surface, from the table up. Old text:

> | Camera mount | Alignment | Views disagree by (median) | Diameter (calipers 29.53mm) | Roundness |
> |---|---|---|---|---|
> | Uncalibrated | off | 3.66mm | 57.05mm | 2.13mm |
> | Uncalibrated | on | 2.47mm | 52.87mm | 3.55mm |
> | Calibrated | off | 0.76mm | 30.87mm | 1.22mm |
> | Calibrated | on | 0.53mm | 32.22mm | 0.29mm |
>
> One open question: with the calibrated mount, turning alignment *off* gives a diameter 1.35mm closer to the calipers, though a far less round one.

### 2026-10-28: the straight-wall prior (§5.8)

Replaced by walls that hang from the lowest place the surface was seen, after the pumpkin's closed mesh came out as a cylinder below its widest point even where the 25° views had measured the surface curving in. Old text:

> - a **wall that was never seen** most likely drops straight down from that outline.
>
> The final `mesh.obj` is a screened Poisson surface fit through the observed surface points (with their normals), plus points filling the bottom outline (normals pointing down), plus straight-wall points only where no observed surface lies within 3mm.

### 2026-10-28: the open-table rule (§5.1)

Replaced by the seen-only rule with a cap on unseen table, after the survey rejected a 200mm-tall pumpkin: its own shadow hid 11% of its ring and counted as "not table" (87% < 90%). Old text:

> - not in open table: less than 90% of a ring 5-60mm around its footprint is bare table (within 5mm of the plane). That rules out cords at the table's edge, things in a gap between tables, and things beside a power strip. It also means two objects closer than about 60mm reject each other.

> On the armfarm14 table, the survey found 10 things above the table: the three blocks, the arm's base and its cables, a clamp-on power strip at the table's edge, and cords. The rules above keep exactly the three blocks.
