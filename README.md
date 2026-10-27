<p align="center">
  <img src="media/images/scan_view_50deg.jpg" alt="The wrist camera's view of a green cylinder, a yellow block and an orange block on a white table, from the 50° ring" width="520">
  <br>
  <em>One of the 25 views the arm took of the green cylinder (50° ring), straight from the wrist-mounted RealSense.</em>
</p>

<h1 align="center">ViamObjectReconstruction by Cameron Lerch</h1>

ViamObjectReconstruction finds the objects sitting on a table, drives a [Viam](https://www.viam.com/)-controlled robotic arm with a wrist-mounted Intel RealSense depth camera around each one, and stitches the views into a separate 3D mesh per object. A single RealSense view is noisy and sees only one side; fusing many views from known arm poses averages the noise away and fills in the sides, and the result is closed with a flat bottom where the object rests on the table.

It runs on a UFactory xArm6 in a remote lab cell (armfarm14) and includes a synthetic RGB-D simulator, so the whole reconstruction pipeline can be developed and scored against known shapes without a robot.

---

## Quick Start

> [!NOTE]
> You need a Viam machine with an arm, a depth camera mounted on the arm (in the frame system), and the builtin motion service. See [Machine Configuration](#102-machine-configuration).

**1. Install**

```bash
git clone git@github.com:cameron-lerch/ViamObjectReconstruction.git
cd ViamObjectReconstruction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the machine's API key in `apikey.json` (it is git-ignored) and the machine address and resource names in `config.toml`:

```json
{"key": "<api key>", "key_id": "<api key id>"}
```

**2. Try it without a robot**

```bash
python -m pytest                          # 36 tests, about 80s
python scan.py demo --progression         # synthetic table scene: survey, plan, render noisy views, reconstruct, score
```

**3. Run on the arm, one careful step at a time** (nothing moves without `--execute`)

```bash
python scan.py plan --new-survey                      # look from where the arm is, plan every object's views (no motion)
python scan.py survey --execute --confirm-moves       # overhead survey; every move is previewed and waits for Enter
python scan.py scan --execute --confirm-above 45 --reuse-survey --objects 1
python scan.py calibrate --objects 1                  # camera-mount calibration from that scan (offline)
python scan.py calibrate-board                        # ChArUco mat on the table: plan board views (add --execute to capture)
python scan.py reconstruct --objects 1                # mesh.obj (closed, flat bottom) and observed.obj
python scan.py progression --objects 1                # meshes and metrics from 1, 2, ... n views
```

Open `data/<session>/object_NN/recon/mesh.ply` in Blender (File > Import > Stanford PLY). Units are millimeters, so scale the import by 0.001.

---

## Table of Contents

1. [Problem Statement and Goals](#1-problem-statement-and-goals) (including [Breaking Down the Problem](#11-breaking-down-the-problem))
2. [Approach](#2-approach)
3. [Results](#3-results)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
6. [Measured vs. Filled In](#6-measured-vs-filled-in)
7. [Critical Design Decisions](#7-critical-design-decisions)
8. [What I Tried That Didn't Work](#8-what-i-tried-that-didnt-work)
9. [Mesh Files and Their Sizes](#9-mesh-files-and-their-sizes)
10. [Setup and Usage](#10-setup-and-usage)
11. [Working Safely on a Shared Real Arm](#11-working-safely-on-a-shared-real-arm)
12. [Repository Structure](#12-repository-structure)
13. [Shortcomings and Proposed Solutions](#13-shortcomings-and-proposed-solutions)
14. [Lessons for Other Developers](#14-lessons-for-other-developers)
15. [Future Work](#15-future-work)
16. [Feedback on Viam](#16-feedback-on-viam)

---

## 1. Problem Statement and Goals

Use a robotic arm with a wrist-mounted RealSense camera to:

|  | Requirement |
|---|---|
| 1. | Automatically find the objects sitting on the table |
| 2. | Plan a path of camera viewpoints around each object that the arm can actually reach |
| 3. | If the arm can't reach enough of the way around an object, grasp it and move it somewhere it can be scanned |
| 4. | Stitch the point clouds from many views of the same object into a mesh that is much more accurate than any single view |
| 5. | Produce an individual 3D model of each object, separated from the table and the other objects |

**Additional goals:** treat the bottoms as flat (the objects rest on the table), quantify how much each additional view improves the result, and never surprise anyone on a shared lab arm.

One lesson carried over from an earlier attempt shaped the capture design from the start: point clouds were noticeably better when the arm **moved to a pose, stopped, and captured**, than when capturing on the fly while the arm moved.

### 1.1 Breaking Down the Problem

When I started, I assumed the main job was to stitch the views together and average their point clouds. It turned out to be one part of three, and the other two (knowing exactly where each view was taken, and how the averaged measurements become a mesh) were at least as important.

**1. Where to look: choosing the viewpoints**

- **Coverage:** every part of the surface has to be seen by some view, or it can only be guessed.
- **Repetition:** the same surface seen from several views is what averages the sensor noise away. Views that don't overlap add coverage but no noise reduction.
- **Overlap for alignment:** views are lined up with each other through the surface they share, so neighboring views must overlap.
- **Reachability and cost:** every view has to be reachable without collisions, and each one costs about 20s on this setup.

**2. Knowing exactly where each view was taken**

- **The camera's mount on the wrist** must be calibrated; here it was 21mm and 2.2° off ([§5.6](#56-camera-mount-calibration)).
- **Each view's pose** is refined by aligning the views to each other ([§5.7](#57-stitching-aligning-the-views)).
- **The depth readings themselves** must be accurate. The real cylinder came out 3mm too wide while every view agreed with every other ([§3.4](#34-the-real-cylinder-against-calipers)).

No smoothing can fix a view that is in the wrong place, and this part has caused the largest errors in the project.

**3. Turning the measurements into a surface**

- **Averaging:** a TSDF averages every view's depth in 1.5mm voxels ([§5.8](#58-fusing-the-views-tsdf)); its zero crossing is the surface. Most of the "smoothing" happens here, as averaging, rather than as smoothing a rough mesh afterward.
- **Resolution:** the voxel size sets the mesh's resolution. Smaller voxels keep more detail but average fewer readings each.
- **What to keep:** surface seen by fewer than 2 views is dropped as untrustworthy.
- **Light smoothing** (5 Taubin steps) removes leftover ripple without shrinking the mesh.
- **Sharp edges** get rounded by about one voxel: the synthetic box's error is 0.25mm against the cylinder's 0.07mm.
- **Filling what was never seen** needs a best guess. Here that is a flat bottom and straight walls, joined to the measured surface by Poisson reconstruction, which continues it smoothly across gaps ([§5.9](#59-closing-the-mesh-with-a-flat-bottom)).

**Assumption: objects have flat bottoms.** The objects rest flat on the table, and the camera can never see underneath them, so the bottom is always filled in: the object's outline seen from above, laid flat on the table plane. This assumption is only true for objects like these. The [measured share](#6-measured-vs-filled-in) says how much of each final mesh comes from measurement and how much from this and the straight-wall guess.

**What "watertight" means.** A watertight (closed) mesh has no holes: every edge of every triangle is shared by exactly two triangles, so the surface completely encloses a volume, with a well-defined inside and outside. That is what makes a volume measurable and the mesh 3D-printable or usable in a physics simulator. The observed surface alone is never watertight: its underside is open and it has gaps wherever no view reached. Adding the flat bottom (and filling unseen walls) is what makes the final `mesh.obj` watertight. Strictly, a mesh with tiny self-intersections, which fusion always leaves, is still closed in this sense; [§8.4](#84-closing-the-mesh) explains why that distinction mattered.

**4. Measuring the result.** Consistency between views measures precision, not accuracy. The synthetic objects have a known true shape, and the real cylinder now has caliper measurements; both are needed to know whether a change actually helped.

---

## 2. Approach

I split the work into stages and tested each before moving on:

1. **Find and verify the hardware model.** Probe the machine read-only: arm kinematics, frame system, camera intrinsics, depth format, and whether the motion planner can check a pose without moving.
2. **Build the reconstruction offline first.** A synthetic RGB-D renderer ([`recon/synthetic.py`](recon/synthetic.py)) produces views of known shapes with RealSense-like flaws (depth noise growing with distance², flying pixels at edges, dropouts) and a deliberate camera-mount error. Every reconstruction idea was scored against the true shapes before it touched real data.
3. **Plan and capture on the real arm, carefully.** Plan-only checks first, then one move at a time with previews, then small scans (3 views), then larger ones (25 views).
4. **Let the real data drive the fixes.** Each real run exposed something the simulator hadn't: a 21mm camera-mount error, a table 7mm higher than measured, walls the RealSense can't see from steep angles, slow cloud downloads. Each one is documented in [§8](#8-what-i-tried-that-didnt-work).

---

## 3. Results

### 3.1 Real cylinder: one view vs. many

25 views of a green cylinder (about 33mm × 60mm): the top view, the 72° ring and the 50° ring (the lowest, 25° ring hasn't been captured yet).

<p align="center">
  <img src="media/images/progression_real.png" alt="Real cylinder: one view's raw points form a thin noisy sliver; 2 views give a small patch of the top; 5 and 12 views add partial walls with holes; 25 views give the whole object; the closed 25-view mesh is a full cylinder with the filled-in parts in orange" width="1000">
</p>

| Views | Seen surface (completeness) | Roughness | Deviation from final (median / p90) | Closed volume | Measured share |
|---|---|---|---|---|---|
| raw points, 1 view | – | 0.281mm | 0.33 / 1.07mm | – | – |
| 1 (best single-view mesh) | 11% | 0.131mm | 0.30 / 0.89mm | 32.0cm³ | 12% |
| 2 | 11% | 0.097mm | 0.19 / 0.58mm | 30.0cm³ | 11% |
| 5 | 23% | 0.096mm | 0.22 / 0.77mm | 44.9cm³ | 32% |
| 12 | 85% | 0.121mm | 0.17 / 0.62mm | 50.6cm³ | 82% |
| 18 | 100% | 0.113mm | 0.08 / 0.49mm | 50.3cm³ | 87% |
| 25 | 100% | **0.103mm** | 0.05 / 0.15mm | **50.1cm³** | **87%** |

- **Coverage:** the best single-view mesh sees 11% of the object (essentially its top); its closed model is 88% assumption and holds 32.0cm³, 36% less than the 25-view result. It takes about 17 views to see all of this cylinder, and from 18 views the closed mesh's volume stops changing.
- **Noise:** the same patch of surface goes from 0.281mm roughness in one raw view, to 0.131mm once that one view is fused and lightly smoothed, to about 0.10mm with 2 or more views (2.7× smoother than raw), and then levels off. Real RealSense noise has a slowly varying, spatially correlated part that averaging removes much more slowly than the independent noise in the simulator (where roughness keeps falling, 38× over 37 views).
- **Consistency:** the 90th-percentile distance from a subset's surface to the final result falls from 1.07mm (one raw view) to 0.15mm, so each added view still tightens the shape after the coverage is complete.
- **Shape:** the reconstruction is round to 0.35mm and the right height, but about 1.5mm too large all the way around. [§3.4](#34-the-real-cylinder-against-calipers) compares it with caliper measurements.

The column definitions are in [§5.10](#510-measuring-quality).

### 3.2 Synthetic objects: scored against the true shape

The same pipeline on rendered views of known shapes (`python scan.py demo --progression`), with RealSense-like noise and a 2.5mm / 0.8° camera-mount error:

<p align="center">
  <img src="media/images/progression_synthetic.png" alt="Synthetic cylinder: one view's raw points show the top disk plus flying pixels; 2 views give only the top; 4 views add strips of wall; 8 and 37 views give the whole smooth cylinder" width="1000">
</p>

| Object | Views | Roughness | Closed mesh error vs. truth (median / p90) | Volume (true) |
|---|---|---|---|---|
| Cylinder Ø50 × 90mm | raw, 1 view | 0.621mm | – | – |
| | 1 (best single-view mesh) | 0.324mm | 1.60 / 2.39mm | 152.6cm³ |
| | 2 | 0.049mm | 1.52 / 8.54mm | 127.6cm³ |
| | 8 | 0.035mm | 0.12 / 0.45mm | 177.8cm³ |
| | 37 | **0.017mm** | **0.07 / 0.25mm** | **177.8cm³** (176.7) |
| Box 70 × 40 × 55mm | raw, 1 view | 0.662mm | – | – |
| | 1 (best single-view mesh) | 0.417mm | 1.45 / 1.93mm | 135.8cm³ |
| | 2 | 0.113mm | 2.33 / 3.59mm | 122.3cm³ |
| | 10 | 0.076mm | 0.29 / 0.63mm | 151.4cm³ |
| | 37 | **0.025mm** | **0.25 / 0.41mm** | **151.4cm³** (154.0) |

- **Many views vs. one:** the cylinder's closed mesh goes from 1.60mm median error with the best single view to 0.07mm with 37 views (23× better), within 0.6% of its true volume. The box goes from 1.45mm to 0.25mm (6× better).
- **Edges:** the box is less accurate than the cylinder because fusion rounds its sharp edges by about one voxel (1.5mm), which also removes 1.7% of its volume ([§13](#13-shortcomings-and-proposed-solutions)).
- **2 views can be worse than 1.** Two views keep only the surface both of them saw, which here is less than one view keeps on its own, so the closed shape is a worse guess (the box: 2.33mm vs. 1.45mm). From 3 views on, the result improves steadily.

### 3.3 The two fixes that mattered most

**Calibrating the camera mount.** The same 25 real captures, aligned and fused identically; only the camera mount differs:

<p align="center">
  <img src="media/images/calibration_before_after.png" alt="With the configured camera mount the 25 views smear into a T-shaped mess; with the calibrated mount they form a recognizable cylinder" width="900">
</p>

**Aligning all views together.** The same 37 synthetic views, aligned two ways:

<p align="center">
  <img src="media/images/alignment_before_after.png" alt="Pairwise ICP with a pose graph leaves ripples and double walls; joint ICP with an arm-pose prior fuses into a smooth surface" width="900">
</p>

### 3.4 The real cylinder against calipers

Caliper measurements of the green cylinder, against the 25-view reconstruction:

| | Calipers | Reconstruction | Error |
|---|---|---|---|
| Diameter | 29.53mm (29.51 top, 29.55 middle, 29.52 bottom) | 32.57mm (circle fit to the walls, round to 0.35mm) | **+3.0mm (+10%)** |
| Height | 59.93mm | 60.53mm | +0.6mm (+1.0%) |
| Volume | 41.0cm³ (π × 14.77² × 59.93) | 50.1cm³ (closed mesh) | +22% |

The shape is right (round, straight, the right height) but the walls sit about 1.5mm too far out on every side. Breaking the error down over the 25 views:

- **Edge fattening, near the silhouette.** Wall points within 3 pixels of the object's outline in the image are 2.3mm too far out; 20+ pixels in, 1.0mm. Stereo depth blends foreground and background near edges, so objects look fatter. Dropping pixels within 5px of a depth edge (not yet the default) brings the diameter to 31.6mm.
- **A uniform part, about 1.0-1.2mm**, even where the wall faces the camera head-on, far from any edge. Because it inflates the object equally from every direction, the views still agree with each other, so neither the mount calibration nor the alignment can see it: only an absolute reference can. Adding a flat 1mm to all depth readings doesn't remove it, so it isn't a simple depth offset.
- **A first absolute check from the ChArUco mat**, from one partial view at the edge of the image: the depth there reads 3.8mm (about 1%) *farther* than the board's printed geometry says. One edge-of-image view is weak evidence; the board calibration in [§5.6](#56-camera-mount-calibration) is built to measure this properly.

The synthetic objects, whose simulated noise has no edge fattening or depth bias, come out within 0.6-1.7% of their true volume, so the error comes from the sensor and calibration, not from the reconstruction method. Closing this gap is the first item in [§15](#15-future-work).

---

## 4. System Architecture

### 4.1 Pipeline

<p align="center">
  <img src="media/diagrams/pipeline.png" alt="Pipeline: on the robot, survey, plan views, relocate if needed, capture stop-and-go, saving views; offline, calibrate the camera mount, align views, fuse with a TSDF, close with a flat bottom" width="1000">
</p>

The robot half moves the arm and saves raw views to disk; the offline half never touches the robot. Keeping the raw views means every reconstruction idea can be re-run and compared on the same captures, which is how most of [§8](#8-what-i-tried-that-didnt-work) was found.

### 4.2 Where each view's pixels go

<p align="center">
  <img src="media/diagrams/stitching.png" alt="A depth pixel is back-projected with the camera intrinsics, then placed in the world by the per-view correction, the arm kinematics and the camera mount. Three error sources: camera-mount error fixed by calibration, residual per-view error fixed by joint ICP, sensor noise fixed by TSDF averaging" width="1000">
</p>

---

## 5. Methodology

### 5.1 Survey and object detection

The camera looks straight down from fixed survey positions (600mm up). For each view separately:

1. Back-project the depth image to 3D points in the world ([§5.5](#55-placing-each-view-in-the-world)), dropping "flying pixels" next to depth jumps.
2. Fit the table plane with RANSAC (must be within 10° of horizontal).
3. Keep points 8-250mm above the table inside the search region, and away from the arm's own base.
4. Cluster them (DBSCAN, 10mm), join fragments whose footprints overlap, and drop clusters under 20mm tall (table noise).
5. Fit each object's smallest footprint rectangle, giving a box: center, length × width × height, and heading.

Each object is then taken from the view that saw it most nearly straight down, where it is least distorted. Detecting per view, instead of merging the views' point clouds first, matters when the camera calibration is off ([§8](#8-what-i-tried-that-didnt-work)).

### 5.2 View planning and reachability

<p align="center">
  <img src="media/diagrams/view_planning.png" alt="Side view: a top view plus rings at 72°, 50° and 25° elevation around the object, all aimed at a point 40% up the object, at a standoff of 280mm plus the object's radius" width="900">
</p>

Every candidate pose is sent to Viam's builtin motion service as a **plan-only** request (`do_command({"plan": ...})`). That runs the real planner (inverse kinematics, joint limits, and collisions with the configured table and walls, every detected object, and the measured table) without moving anything. The planner targets the **camera frame** directly, so no camera-to-gripper math is needed. Coverage is scored over about 350 surface directions (the bottom excluded); below 85%, the object would be relocated.

### 5.3 Relocation

If coverage is too low, the planner looks for a free scan station where the object would be fully coverable, finds a top-down grasp across the object's narrow side, checks the approach, grasp and place poses, and carries the object there (attached to the gripper in the planner's world state, so the carried object avoids obstacles too). After scanning, it puts the object back. **This is implemented but disabled** (`[relocate] enabled = false`) until the gripper's opening, fingertip offset and jaw direction are measured; on armfarm14 every object so far was fully coverable in place.

### 5.4 Stop-and-go capture

At each view: move, wait until the arm reports it has stopped, wait 0.6s more, then take 5 frames and keep the per-pixel median depth. Frames count only if the camera **stamped them after the settle time, by the robot's own clock**, so nothing captured while moving can slip in. The camera pose is read after settling and checked against the arm controller ([§11](#11-working-safely-on-a-shared-real-arm)). Each view is saved as `color.jpg`, `depth.png` (16-bit millimeters) and `view.json` (pose, joints, timestamps, camera mount).

### 5.5 Placing each view in the world

A depth pixel (u, v) with depth d becomes a camera-frame point with the camera's intrinsics (focal lengths fx, fy and center cx, cy), then a world point through three transforms:

```math
p_\text{world} = C_i \; T_{\text{world} \leftarrow \text{arm end}}(\text{joints}) \; M_{\text{arm end} \leftarrow \text{camera}} \; p_\text{camera}
```

- **T** comes from the arm's joint angles and kinematics. The xArm6 repeats to about 0.1mm, so this part is trustworthy.
- **M** is the camera's mounting pose on the wrist, from the machine's frame configuration. If it's wrong, every view is wrong by a different amount, because the error rotates with the wrist.
- **Cᵢ** is a small correction for view *i*, found by the alignment step.

These three terms map onto the three error sources and the three fixes below.

### 5.6 Camera-mount calibration

The configured mount on armfarm14 was off by 21mm and 2.2°, which showed up as the table appearing at heights up to 20mm apart from one view to the next. `scan.py calibrate` estimates **M** from any scan of a still scene: it finds the one mount correction (6 numbers, shared by all views) that makes the views agree, by minimizing point-to-plane distances between overlapping views over the object and a 60mm ring of table, with correspondences re-matched as the estimate improves (coarse 25mm to fine 3mm). On the real cylinder:

| | Spread of table height across 25 views | Spread of the cylinder's top | Single views vs. fused surface |
|---|---|---|---|
| Configured mount | 20mm (std 6.0mm) | 21mm (std 6.1mm) | 2.48mm median |
| Calibrated mount | 4.4mm (std 1.0mm) | 5.0mm (std 1.1mm) | 0.52mm median |

On synthetic data it recovers a known 12mm / 2° mount error to within 1.5mm / 0.3°. The result is saved with the session (later reconstructions use it) and printed in the Viam frame format, to be entered once in the app so future captures start out right.

**What scan-based calibration can't see.** Anything that changes the object's size equally from every direction (a depth bias, or a camera that's really a little farther along its own axis than calibrated) leaves all the views in agreement, so self-calibration absorbs it. The caliper check ([§3.4](#34-the-real-cylinder-against-calipers)) shows about 1mm of exactly that.

**Board calibration** (`scan.py calibrate-board`) uses a ChArUco mat on the table (calib.io, 12 × 9 squares of 15mm, 11mm markers, ArUco DICT_5X5) as an absolute reference. From the color image alone, each view gives the camera's pose relative to the board in millimeters, independent of the depth sensor. The command:

1. Finds the board from where the camera is, and checks it looks **level**. A mat on the table must be level, so if it isn't, the camera frame in the machine config is wrong and the command stops. Through the camera frame entered with θ 129.489°, the mat looked 10.5° from level; through the calibrated θ 89.233°, 0.6°. That was an independent confirmation of the scan-based calibration's rotation, from one frame and no motion.
2. Plans 16 views of the board from 300-460mm and from different directions (plan-only), and with `--execute` captures them.
3. Solves for the camera mount and the board's pose together, minimizing the corners' reprojection error (hand-eye calibration), and fits the depth sensor's error as an offset plus a scale against the board's known geometry.

On synthetic corners with 0.3px noise it recovers the mount to within 1mm / 0.1° from a start 20mm / 3° off. It hasn't been run on the arm yet.

### 5.7 Stitching: aligning the views

After calibration, views still disagree by about a millimeter (arm accuracy, calibration leftovers). Fusing them as-is would average slightly shifted copies of each surface into ripples. The alignment step solves for all the per-view corrections **Cᵢ** at once:

1. Each view's points near the object (plus a 10mm ring of table) are downsampled to 2mm, cleaned, and given surface normals.
2. Every pair of views pointing within 60° of each other is an overlapping pair. For 1500 points per pair, the nearest point in the other view with a similar normal is a match.
3. Each match contributes a **point-to-plane** residual: the distance from a point to the plane of its match. Point-to-plane lets flat surfaces slide along themselves, which is correct: a flat patch says nothing about sliding.
4. Each correction is also pulled toward zero with a **prior**: 3mm, 1° of tilt and 0.3° of spin about the vertical axis, the arm's expected accuracy. Where the geometry pins a correction down, the thousands of matches outweigh the prior; where it can't (a cylinder spinning about its own axis looks identical), the prior keeps the arm's pose. Spin gets the tighter prior because it is exactly the motion a round object can't constrain, and noise will otherwise pretend to.
5. Gauss-Newton solves all corrections together (6 numbers per view), re-matching each iteration while the match distance shrinks from 6mm to 3mm, for up to 25 iterations. Rotations are measured about the object's center, not the robot's base 500mm away, so they mean something physically.
6. Any correction over 15mm or 5° is rejected and that view keeps its arm pose. On the real, calibrated cylinder, 1 of 25 views was rejected.

On synthetic views this cuts the relative pose error from 1.87mm to 0.48mm, and it is the difference between the two meshes in [§3.3](#33-the-two-fixes-that-mattered-most).

### 5.8 Fusing the views (TSDF)

First, the object's box is **refit from the scan itself**: every view's points above its own table, within 40mm of the survey's box, are clustered and a new box is fit around the main cluster (before alignment with the arm poses, and again after). The survey's box can be wrong by more than the crop margin, and a box in the wrong place cuts the same slice off the object in every view ([§8.2](#82-detection)).

Each view's depth map is then cropped to the object: inside its box (plus 12mm) and at least 2.5mm above **that view's own fit of the table** in a 30mm ring around the object (the survey's table can be several mm off at the object). The cropped depth maps go into a truncated signed distance field (TSDF, Open3D's `VoxelBlockGrid`, on the GPU when available):

<p align="center">
  <img src="media/diagrams/tsdf_fusion.png" alt="Three panels. 1: three cameras measure an object's wall with noise; flying pixels at the silhouette are crossed out. 2: one view writes signed distances into 1.5mm voxels: positive in front of its reading, negative behind it, within a ±6mm band, free space clamped, voxels more than 6mm behind left alone. 3: averaging three views' signed distances; the fused surface is where the average crosses zero, and a region seen by only one view is not kept. Below: what is thrown out per view, during fusion, and at extraction." width="1000">
</p>

- **Space is divided into 1.5mm voxels.** Each view, along every pixel's ray, writes the signed distance from each voxel to the surface that pixel measured: positive in front of it, negative behind it.
- **Only a ±6mm band is updated.** Signed distances are truncated to ±6mm; voxels farther in front are marked as free space, and voxels more than 6mm *behind* the reading are left alone, since that region is hidden from this view and another view may see it.
- **Readings are averaged, not thrown out.** Each voxel keeps a weighted average of every view's signed distance, so random noise cancels. The surface is where the average crosses zero (marching cubes).
- **Data is thrown out before and after fusion, not inside it:** flying pixels at depth edges and anything outside the object's box or below its table before; surface seen by fewer than 2 views (or fragments) after.

That is `observed.obj`: everything in it was measured, and it is open wherever nothing was seen.

### 5.9 Closing the mesh with a flat bottom

The objects rest flat on the table, which gives two priors:

- the **bottom** is the object's outline seen from above, lying on the table plane;
- a **wall that was never seen** most likely drops straight down from that outline.

The final `mesh.obj` is a screened Poisson surface fit through the observed surface points (with their normals), plus points filling the bottom outline (normals pointing down), plus straight-wall points only where no observed surface lies within 3mm. The lowest 5mm of observed surface is replaced by the straight-wall prior (depth pixels where wall meets table flare outward), anything below the table is flattened onto it, and the outline is shrunk by 1mm because edge noise pads it. The result is closed (no open edges), so it has a volume. [§6](#6-measured-vs-filled-in) shows which parts are measured and which are filled in.

### 5.10 Measuring quality

`scan.py progression` aligns all views once, then fuses the first *k* views for every *k* (adding each next view from the most different direction, so 3 views means 3 different sides). Each subset keeps surface seen by at least min(2, *k*) of its views: the single view keeps everything it saw, so the 1-view mesh is the best one view can give, and 2 or more views keep only what at least 2 of them agree on. `--min-views` overrides this for every subset.

| Metric | What it measures | Needs ground truth? |
|---|---|---|
| **Roughness** | Median RMS distance of 3mm surface patches from their best-fit plane. Measured only on the patch the first view saw, which every subset includes, so rows compare the same surface. | No |
| **Deviation from final** | Distance from this subset's observed surface to the all-views closed mesh (median and 90th percentile). | No |
| **Completeness** | Share of the all-views observed surface that this subset saw (within 2mm). It is relative: a region no view ever saw can't count against it, which is why the measured share matters too. | No |
| **Measured share** | Share of the closed mesh's surface that lies within 2mm of measured data ([§6](#6-measured-vs-filled-in)). | No |
| **Volume** | Enclosed volume of the closed mesh. | No |
| **Error vs. truth** | Distance from the mesh to the true shape after the best rigid fit, so it scores shape rather than placement. | Yes (synthetic) |

Results go to `recon/progression/summary.csv`, with `mesh.obj` and `observed.obj` for each *k* and the first view's raw points.

---

## 6. Measured vs. Filled In

Every closed mesh is part measurement and part assumption. The **measured share** is the fraction of the closed mesh's surface area that lies within 2mm of a point the camera actually measured (fused from at least 2 views). The rest, shown in orange below, comes from the two priors in [§5.9](#59-closing-the-mesh-with-a-flat-bottom): the flat bottom and straight walls where nothing was seen.

<p align="center">
  <img src="media/images/measured_vs_filled.png" alt="Closed meshes of the real cylinder from 3, 8 and 25 views, from the side and from below. With 3 views almost all of the walls and the bottom are orange (filled in); with 8, part of the walls is measured; with 25, most walls are measured and the bottom remains filled in" width="1000">
</p>

| Real cylinder | 1 view | 8 views | 25 views |
|---|---|---|---|
| Measured share | 12% | 65% | 87% |
| Closed volume | 32.0cm³ | 50.9cm³ | 50.1cm³ |

How to read it:

- **The bottom is always filled in.** The camera can never see under the object. For these cylinders the bottom is 10-11% of the surface, which is why the real 25-view scan (87%) and the synthetic 37-view scan (88%) both top out just under 90%.
- **With few views, the walls are mostly guessed.** The best single-view closed mesh looks like a plausible cylinder, but 88% of it is assumption, not data, and its volume (32.0cm³) is 36% below the 25-view result and 22% below the caliper volume. This is why the closed meshes alone would overstate what few views achieve, and why `observed.obj` is saved next to every `mesh.obj`.
- **The measured share caught a real bug.** An earlier version of this scan was only 64% measured, with one side of the cylinder orange in every closed mesh, even with 25 views. The crop box was cutting that side off ([§8.2](#82-detection)); completeness alone couldn't show it.
- **The priors are only right for objects shaped like these:** flat bottom, walls roughly vertical where unseen. An overhang or a hollow would be filled in wrong; the measured share says how much of the model depends on that.

---

## 7. Critical Design Decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| **Start fresh instead of extending my earlier script** | Build on the earlier ViamOpenLab script | It was a single 636-line script that moved the arm but saved nothing and had no reconstruction. Its hand-written camera-to-gripper math and reach limits became unnecessary once the planner could target the camera frame and report reachability. I kept its good ideas: the credentials file, detected objects as obstacles, and previewing before moving. |
| **Stop, settle, then capture** | Capture while moving | Earlier experience showed cleaner clouds when stopped. Frames are accepted only if stamped after settling, by the robot's clock. |
| **Depth images + intrinsics, not the point-cloud API** | `camera.get_point_cloud()` | The full point cloud dropped the cloud connection; depth images are smaller, and owning the back-projection allows per-pixel filtering and median stacking. |
| **Reachability from Viam's planner, plan-only** | Own IK from the arm's kinematics | The planner already knows the joint limits and the cell's collision geometry; asking it guarantees that "reachable" means the same thing at plan time and at move time. |
| **Save raw views, reconstruct offline** | Reconstruct live | Every method in this README was compared on the same captures. Robot time is the scarce resource. |
| **Simulator with deliberate errors** | Develop on the real arm only | Ground truth makes it possible to tell "better" from "different", and the arm is shared. |
| **Median of 5 frames per view** | 1 frame | Cuts per-pixel noise cheaply while the arm is still anyway. |
| **Calibrate the camera mount from scan data** | ChArUco hand-eye calibration | No board needed and it uses exactly the data being fused. It found a 21mm error; a board-based calibration would be a good independent check. |
| **Joint ICP with an arm-pose prior** | Pairwise ICP + pose graph (tried first) | The pose graph trusted directions the geometry can't observe and spun symmetric objects; the joint solve with a prior handles that by construction. |
| **TSDF fusion, keeping surface seen by 2+ views** | Merge point clouds; keep everything | Averaging signed distances removes random noise; the 2-view rule removes surface no second view confirms. |
| **Refit the object's box from the scan** | Trust the survey's box | The survey sees the object from far away and before calibration; the scan sees it from 25 sides. |
| **Two meshes per object: observed and closed** | Only a closed mesh | The closed mesh is what you'd 3D-print or simulate with; the observed mesh is the honest record. The measured share connects them. |
| **Flat bottom + straight-wall prior, via Poisson** | Leave meshes open; cap only the bottom rim | Real scans have holes in the walls, so a bottom cap alone can't close them ([§8](#8-what-i-tried-that-didnt-work)). |

---

## 8. What I Tried That Didn't Work

In roughly the order I ran into them. Each was found by measuring, and the numbers are from those measurements.

### 8.1 Hardware and platform

| What I tried | What went wrong | What I did instead |
|---|---|---|
| `camera.get_point_cloud()` | The full 1280×720 cloud dropped the connection to the cloud-hosted machine. | `get_images()` returns color plus raw depth (`image/vnd.viam.dep`: header, then big-endian uint16 mm); I back-project it myself. |
| The SDK's `RotationMatrix.elements` reshaped row by row | It's column-major, so that gives the transpose. It went unnoticed at first because a camera looking straight down has a nearly symmetric rotation. | Own orientation math ([`recon/transforms.py`](recon/transforms.py)), tested against the numbers the machine's `transform_pose` reports. |
| Open3D's legacy `ScalableTSDFVolume` | On Open3D 0.20 / Python 3.14 it integrated nothing, even for a flat wall straight in front of the camera. | The tensor `VoxelBlockGrid`, which also runs on the GPU. |
| A view-count threshold of 2.0 in the `VoxelBlockGrid` | Extraction keeps weights strictly *above* the threshold, so "at least 2 views" really meant "at least 3". The real meshes had more holes than they should have. | `min_views` (an integer) passed as `min_views − 0.5`. The real 25-view surface grew from 4,237 to 7,386 triangles. |
| Capturing 5 frames within 10s | Each `get_images()` takes 2-4s over the cloud connection (depth alone is 1.8MB). | The timeout is per frame. A view now takes about 20s. |
| The SDK's default connection check | It allows 1s for a reply; during a 1.8MB download the reply is late, and the SDK closes the channel mid-download. | Turned off for this script. A real disconnect still raises on the next call, and viam-server's session heartbeat (which stops the arm if the client disappears) is unaffected. |
| Entering the calibrated mount as an orientation vector `(0.00678, -0.00574, 0.99996)`, θ 129.5° | Within 1e-4 of the pole, Viam reads θ as the *total* turn about z, ignoring the axis's longitude. The camera ended up 40.3° off, and the pose guard stopped the run before any motion. | The pose math implements Viam's pole rule; the correct θ is 89.233°. |
| The frame system, once | During one survey, Viam's frame system reported the camera exactly at its planned target while the joint readings (and the image) put it rolled 180° away. It hasn't recurred. | A pose guard ([§11](#11-working-safely-on-a-shared-real-arm)) checks the frame system against the arm controller after every move and at every capture. |
| The first roll that worked for each pose | The first survey move would have turned the wrist 140°, and a 25-view scan wound it to −280°, tangling the camera cable. | Plan every roll, keep the one needing the least joint motion, and keep the wrist within ±200°. |
| Open3D's interactive viewer | It crashes on GNOME 50 / Wayland (a GTK settings lookup), even with a hidden window. | Meshes are viewed in Blender; README figures are rendered offscreen with numpy ([`tools/render_mesh.py`](tools/render_mesh.py)). |

### 8.2 Detection

| What I tried | What went wrong | What I did instead |
|---|---|---|
| Clustering every point above the table | Flying pixels between an object's top edge and the table shifted the synthetic cylinder's center by 36mm; noise on the table formed flat fake objects. | Drop pixels next to depth jumps; require objects to stand at least 20mm tall. |
| Merging the survey views' clouds, then detecting | With the camera-mount error, the two views placed the same block 7-26mm apart; merged, the blocks smeared into 121 × 92mm boxes (and once, into one 493 × 303mm blob). | Detect in each view; keep each object from the view that saw it most directly from above. |
| Cropping each view to the survey's box | The survey (taken before the mount was calibrated) placed the real cylinder 17mm off. With a 12mm crop margin, every view lost the same 60-90° slice of wall: the 25-view closed mesh was only 64-73% measured. The fitted diameter came out 29-30mm, close to the true 29.5mm only by coincidence: the missing slice hid the inflation of the rest. | Refit the box from the scan's own points ([§5.8](#58-fusing-the-views-tsdf)): every 30° sector of the wall now has data and 87% is measured. That exposed the real 1.5mm-per-side size error ([§3.4](#34-the-real-cylinder-against-calipers)). |
| Using the survey's table plane to separate object from table | Near the object the real table was about 7mm higher than the survey measured, so table points leaked into the object. | Fit the table per view from a ring around the object, in that view's own (aligned) pose. |

### 8.3 Stitching

| What I tried | What went wrong | What I did instead |
|---|---|---|
| **Pairwise ICP + Open3D pose graph** (multiway registration) | Open3D's information matrix assumes point-to-point matching, so it claimed certainty about motions the surface can't constrain, such as a cylinder's spin. Low views of the cylinder were "corrected" by up to 125°. | Point-to-plane information matrices: better, but pairwise ICP still drifted 27° along the spin before the graph ever saw it. |
| Projecting each pairwise result onto its well-constrained directions (eigenvalue cutoff) | Fragile: the table ring dominated the eigenvalues, so the box's real in-plane constraints fell below the cutoff and its 90th-percentile error jumped to 8mm. | Dropped the pose graph entirely for a joint solve with a prior ([§5.7](#57-stitching-aligning-the-views)). |
| The joint solve, first version | Every view shared a ~10° rotation. The target-side Jacobian ignored that the target's normal rotates with its view; done right, the two sides cancel exactly for a shared motion. A finite-difference check found it. | Fixed (J_target = −J_source). Relative pose error 1.87 → 1.02mm. |
| Default ICP settings | Not converged: wide match distance, a large table ring. | Match distance 6 → 3mm over 25 iterations, a 10mm table ring, 1500 points per pair: 0.48mm. |
| One rotation prior (1°) for all three axes | On the real, calibrated cylinder 21 of 25 views wanted 5-33° of spin about the vertical and were rejected, so the result was effectively arm poses only. | A separate 0.3° prior on spin: rejections 21 → 1, wall roundness 0.99 → 0.35mm, single views vs. the result 0.74 → 0.52mm. The synthetic objects are unchanged. |
| Per-view alignment on the real 25-view scan | It wanted 7-44mm, up to 33° corrections; the 15mm / 5° safety limit rejected them. The views weren't slightly off; the camera mount was wrong. | Mount calibration ([§5.6](#56-camera-mount-calibration)), then alignment for what's left. |

**Why some early demo meshes were smoother than others.** The synthetic demos `demo-20261026-131545`, `-131913`, `-132152` and `-133443` were the first four built with the joint solve; the earlier ones used the pose graph, with up to 26 of 37 views rejected or mis-corrected. Same noise, same smoothing settings; only the alignment differed. Re-fusing the bumpy demos' own raw views with the joint solve cut their wall roughness from 0.14-0.28mm to 0.05-0.06mm and their error against the true shape from 0.19-0.46mm to 0.11mm. Misaligned views don't add noise; they add copies of the surface at slightly different places, which the averaging turns into ripples.

### 8.4 Closing the mesh

| What I tried | What went wrong | What I did instead |
|---|---|---|
| A vertical "skirt" from the observed bottom rim down to the table, capped with a flat polygon | Exact on synthetic meshes, which have a clean bottom rim. Real meshes have large holes in the walls, so the "rim" ran 60mm up the side, and pinch points between holes broke the loop tracing. | Poisson with a flat-bottom prior ([§5.9](#59-closing-the-mesh-with-a-flat-bottom)). |
| Open3D's `is_watertight()` and `get_volume()` | Both refuse meshes with tiny self-intersections, which marching cubes and smoothing always leave. | Closed means "no open edges"; volume from the divergence theorem. |
| Poisson with a tight working box | Few-view surfaces ran into the box and stayed open. | A 3× box. |
| Poisson alone with few views | With nothing to go on for the unseen sides, the 1-view synthetic cylinder ballooned to 2,521cm³ (true 177). | The straight-wall prior, only where nothing was observed: 153cm³ from 1 view, and no effect from 8 views on. |
| Wall prior points on the 1mm outline grid | Vertical ribs from the grid's staircase. | A smoothed outline with sub-cell positions. |
| The observed surface right down to the table | Pixels where wall meets table flare outward, giving the base a lip. | The lowest 5mm is replaced by the straight-wall prior. |

### 8.5 Measuring progress

| What I tried | What went wrong | What I did instead |
|---|---|---|
| Roughness over each subset's whole surface | It *rose* with more views (0.08 → 0.24mm), because few views see mostly the flat top and more views add the curved, noisier walls. | Roughness on the patch the first view saw, which every subset contains. |
| Completeness alone as the coverage measure | It is relative to the all-views surface, so a slice that every view lost (the crop bug above) still scored 100%. | The measured share, which compares against the closed mesh, showed the missing side as a persistent orange band. |
| Judging the real result by self-consistency | Round to 0.35mm, the same volume from 17 views on, views agreeing to 0.5mm: all true, and the cylinder was still 3mm too wide. | Calipers ([§3.4](#34-the-real-cylinder-against-calipers)). Consistency measures precision, not accuracy. |
| Correcting the size error with a flat depth offset | Adding 1mm to all depth made the diameter slightly *larger* (the alignment and table fits move with it). | Measure the depth error against the ChArUco board instead of guessing ([§5.6](#56-camera-mount-calibration)). |
| Applying "seen by 2+ views" to every subset, including 1 view | The 1-view mesh was empty, so there was no single-view result to compare the others against. | Each subset keeps surface seen by min(2, k) views: the 1-view mesh is the best one view can give. |

---

## 9. Mesh Files and Their Sizes

Each reconstruction writes, in `data/<session>/object_NN/recon/`:

| File | What it is |
|---|---|
| `mesh.obj`, `mesh.ply` | Closed mesh with a flat bottom (the final model). PLY keeps vertex colors; OBJ doesn't. |
| `observed.obj`, `observed.ply` | Only the surface seen by 2+ views; open where nothing was seen. |
| `cloud.ply` | The fused surface points with normals (what Poisson closes). |
| `report.json` | Alignment corrections per view, per-view table offsets, how far each raw view lies from the result, mesh statistics. |
| `refined_poses.json` | Each view's camera pose after alignment. |
| `progression/views_NN/` | `mesh.*` and `observed.*` from the first N views; `views_01/raw_points.ply` is the first view's raw points. |
| `progression/summary.csv` | The metrics in [§5.10](#510-measuring-quality) for every N. |

Sizes from the scans in [§3](#3-results):

| | Views | Observed triangles | `observed.obj` | Closed triangles | `mesh.obj` |
|---|---|---|---|---|---|
| Real cylinder (~33 × 60mm) | 1 | 627 | 47KB | 26,434 | 1.7MB |
| | 2 | 705 | 53KB | 26,736 | 1.8MB |
| | 12 | 6,333 | 494KB | 25,986 | 1.7MB |
| | 25 | 8,548 | 647KB | 25,996 | 1.7MB |
| Synthetic cylinder (Ø50 × 90mm) | 1 | 2,781 | 207KB | 129,072 | 9.0MB |
| | 2 | 1,316 | 98KB | 124,320 | 8.6MB |
| | 8 | 19,297 | 1.5MB | 46,186 | 3.1MB |
| | 37 | 18,493 | 1.4MB | 43,360 | 2.9MB |

How to read the sizes:

- **The observed mesh grows with coverage.** Its triangle count tracks how much surface was seen, at a fixed 1.5mm voxel. (One view keeps everything it saw, so the 1-view observed mesh can be bigger than the 2-view one, which keeps only the overlap.) It stops growing once everything is seen (the synthetic cylinder at 8 views) and then shrinks slightly as averaging smooths out small bumps.
- **The closed mesh's size says nothing about quality.** Poisson's resolution is set by its working box (2⁹ cells across), so the triangle count follows surface area and wrinkliness. With few views the guessed surface is larger and lumpier: the 1- and 2-view synthetic cylinders' closed meshes have about 3× the triangles of the 37-view one.
- **The real cylinder's meshes are smaller** than the synthetic one's because the object is smaller (33mm vs. 50mm across, 60mm vs. 90mm tall).
- **Raw data:** the 25 captured views take 4.2MB (each about 170KB: a JPEG and a 16-bit PNG depth map); one raw view's object points are 43KB.

---

## 10. Setup and Usage

### 10.1 Prerequisites

- Python 3.14 with the packages in [`requirements.txt`](requirements.txt) (Viam SDK, Open3D 0.20, NumPy, SciPy). `cairosvg` is optional, for PNG copies of figures.
- A Viam machine with an arm, a depth camera whose frame is attached to the arm, the builtin motion service, and the work area's static obstacles (table, walls) as geometry in the frame system.

### 10.2 Machine Configuration

`config.toml` holds the cell's settings; every setting and its default is documented in [`recon/config.py`](recon/config.py). For armfarm14:

| Setting | Value | Notes |
|---|---|---|
| `[machine] address` | `armfarm14-main.310sld03v2.viam.cloud` | resources `arm`, `cam`, `gripper`, `builtin` |
| `[workspace] x_range, y_range` | −350..700, −650..350mm | clear of the walls (front x=800, side y=425) |
| `[workspace] survey_positions` | (150, −400, 600), (350, −150, 600) | both plan; camera looks straight down |
| `[views] standoff` | 280mm | beyond the RealSense's ~260mm minimum range |
| `[capture]` | settle 0.6s, 5 frames | |
| `[relocate] enabled` | false | until the gripper is measured |

The calibrated camera frame for armfarm14 (parent `arm`), from `scan.py calibrate`:

```json
"translation": {"x": 70.03, "y": -29.50, "z": 24.38},
"orientation": {"type": "ov_degrees", "value": {"x": 0.00678, "y": -0.00574, "z": 0.99996, "th": 89.233}}
```

### 10.3 Commands

| Command | Moves the arm? | What it does |
|---|---|---|
| `scan.py survey [--execute]` | with `--execute` | Overhead views, table plane, objects; without `--execute`, from the current pose |
| `scan.py plan [--new-survey]` | no | Reachable views, coverage and relocation decision per object |
| `scan.py scan --execute [--objects N] [--max-views K] [--reuse-survey]` | yes | Plan, (relocate), capture, reconstruct |
| `scan.py calibrate --objects N` | no | Camera mount from a scan; prints the Viam frame values |
| `scan.py calibrate-board [--execute]` | with `--execute` | ChArUco mat: level check, board views, hand-eye mount and depth error |
| `scan.py reconstruct [--objects N] [--no-refine]` | no | Re-run reconstruction on saved views |
| `scan.py progression --objects N [--counts 1,2,5,25] [--order spread\|capture] [--min-views M]` | no | Meshes and metrics for 1..n views |
| `scan.py demo [--progression]` | no | Everything on synthetic data, scored against the truth |
| `tools/make_figures.py`, `tools/make_tsdf_diagram.py` | no | Regenerate this README's figures and the fusion diagram |

Motion options: `--confirm-moves` previews every move (target, joint changes) and waits for Enter; `--confirm-above 45` previews every move but only waits when a joint would turn 45° or more; `--step` asks before each object and each relocation.

### 10.4 Testing

`python -m pytest` runs 36 tests, about 80s. Besides unit tests (pose math against the machine's own numbers, depth decoding, edge filtering, detection, view geometry, grasp poses, the pose guard, ChArUco detection and hand-eye recovery of a known mount), several run the pipeline on synthetic scenes and check it against the truth: fused meshes beat single views, mount calibration recovers a known error, closed meshes have the right volume, and the progression improves with views.

---

## 11. Working Safely on a Shared Real Arm

The armfarm cells are shared and remote, so the code assumes that nobody may be watching the arm closely:

- **Nothing moves without `--execute`.** Every command without it only looks and plans.
- **Every move can be previewed**, with its target and the joint changes from a plan-only check, and waits for Enter. Typing `n` stops cleanly before the move.
- **Pose guard.** At startup, after every move and at every capture, the camera pose from Viam's frame system is compared with the pose computed from the arm controller's own end position and the camera mount. If they disagree by more than 2mm or 1°, the run stops before any further motion. It caught a 40° camera-configuration error before the first move.
- **The measured table is an obstacle.** The configured table obstacle sat 15-30mm below the real table, so a slab of the table as perception measured it is added around each object for every plan.
- **Detected objects are obstacles**, with a 15mm margin.
- **Wrist limit.** Camera rolls that would take the last joint past ±200° are never chosen.
- **Ctrl+C sends `arm.stop()`** before disconnecting. The e-stop remains the real safety.
- **Relocation is off** until the gripper is measured; it hasn't been tested on hardware.
- **Board level check.** `calibrate-board` refuses to plan views if the mat doesn't look level through the configured camera frame, which is what a wrong frame in the machine config looks like.

---

## 12. Repository Structure

```
scan.py                  command line: survey, plan, scan, calibrate, calibrate-board, reconstruct, progression, demo
config.toml              this cell's settings (address, workspace, survey positions, stations)
recon/
  config.py              every setting, with defaults and explanations
  robot.py               everything that talks to Viam: plan-only checks, moves, previews, pose guard, capture
  pipeline.py            survey → plan → (relocate) → capture → reconstruct, and calibration I/O
  scene.py               table plane, object detection, merging detections across views
  views.py, planner.py   candidate viewpoints, coverage, reachability and roll choice
  relocate.py            grasps, scan stations, pick and place
  calibrate.py           camera-mount calibration from a scan
  board.py               ChArUco board: detection, hand-eye calibration, depth error model
  reconstruct.py         per-view table, joint alignment, TSDF fusion, observed mesh, reports
  bottom.py              closed mesh: flat bottom, straight-wall prior, Poisson
  progression.py         1..n view comparison and its metrics
  rgbd.py, transforms.py depth decoding and filtering, view files, pose math (Viam conventions)
  session.py             data/<session>/ layout
  synthetic.py           noisy synthetic RGB-D renderer with a camera-mount error
tools/
  render_mesh.py         offscreen mesh and point renders (PNG)
  make_figures.py        this README's result figures
  make_tsdf_diagram.py   the fusion diagram
tests/                   36 tests, mostly against synthetic ground truth
media/                   diagrams (SVG + PNG) and figures
```

---

## 13. Shortcomings and Proposed Solutions

| Shortcoming | Impact | Proposed solution |
|---|---|---|
| **Real objects come out about 1.5mm too large on every side** (the cylinder: 32.6mm vs. 29.5mm). | Volumes are about 20% high for an object this size; heights are right to 1%. | Run the board calibration to measure depth error against an absolute reference; drop pixels near depth edges (31.6mm alone); weight each reading by how squarely it sees the surface; re-check against the calipers. |
| **Real noise levels off at about 0.10mm roughness.** | Beyond a few views, more views tighten the shape but don't make the surface smoother; the RealSense's noise is partly correlated across pixels and views. | Vary the views more (distances, rolls), use the RealSense's own spatial/temporal filters, or a camera better suited to close range (such as the D405). |
| **The lowest ring (25°) hasn't been captured on the real arm.** | The base of the walls is seen only obliquely (and the lowest 5mm is replaced by the straight-wall prior anyway). | Scan all 37 views; confirm the low views' clearance with `--confirm-moves` first. |
| **The calibrated camera frame is not yet in the Viam app correctly.** | The app currently has θ 129.489° (40° off); scans use the session's calibration file, but planning sees the wrong camera pose. | Enter θ 89.233° ([§10.2](#102-machine-configuration)), then re-survey and check the two survey views agree. |
| **Calibration from one object, and blind to uniform scale.** | A good fit to one scene could hide a degenerate direction, and no scan-based calibration can see an error that inflates everything equally. | Run `calibrate-board` and compare; calibrate from two or three objects. |
| **Sharp edges are rounded** by about one voxel. | The box's error is 0.25mm vs. the cylinder's 0.07mm, and it loses 1.7% of its volume. | Smaller voxels near edges, less smoothing, or fitting planes and edges for box-like objects. |
| **Priors assume flat bottoms and vertical unseen walls.** | Overhangs, hollows or tilted objects would be filled in wrong. | Use the measured share to flag models that lean on the priors; capture more views where it's low; a visual hull from many silhouettes for unseen regions. |
| **Capture is slow over the cloud connection** (about 20s per view). | A full 37-view scan takes about 15 minutes. | Run the scanner next to the arm (a Viam module), stream depth only, or use 3 frames per view. |
| **Relocation untested on hardware.** | Objects the arm can't reach around wouldn't be scanned well. | Measure the gripper (opening, fingertip offset, jaw axis) and test pick and place on its own first. |
| **A symmetric object's spin is unobservable.** | Fine for shape; texture would be smeared on a textured cylinder. | Colored ICP when the object has texture. |
| **The intermittent frame-system flip** has no known cause. | Caught by the pose guard, but a run would stop. | Log joints and frame-system poses at high rate around moves to reproduce it, then report it. |
| **Stop-and-go vs. continuous capture** hasn't been measured. | Stop-and-go is the slower, assumed-better choice. | Capture the same object both ways and compare with the progression metrics; capture already records timestamps and joints per frame. |

---

## 14. Lessons for Other Developers

- **Build a simulator with your real errors in it.** Noise alone isn't enough; add a camera-mount error, and every alignment idea can be scored against the truth before touching the robot.
- **Calibrate the camera mount before anything else.** A 21mm / 2° mount error looked like noisy, unalignable views; no amount of per-view alignment fixes a systematic error.
- **Keep the raw views.** Nearly every improvement here came from re-running reconstruction on captures that already existed.
- **Point-to-plane, and know what your geometry can't see.** A cylinder's spin, a plane's sliding: an optimizer that doesn't know these are unobservable will invent motion along them. A weak prior toward the measured pose is the cleanest fix.
- **Check derivatives numerically.** A missing term in one Jacobian made every view rotate together by 10°; a finite-difference check found it in a minute.
- **Check every threshold's exact meaning.** "Weight > 2" vs. "≥ 2" and Viam's pole rule for orientation vectors each produced results that looked plausible and were wrong.
- **Self-consistency isn't accuracy.** Views agreeing to half a millimeter, a round cylinder, a stable volume: all true of a reconstruction that was 3mm too wide. One caliper measurement found it.
- **Report what's measured and what's assumed.** A closed mesh always looks complete; the measured share tells the truth about it.
- **On a shared arm, verify the world model, not just the commands.** The pose guard compares two independent sources of where the camera is; it stopped a run with a 40°-wrong camera configuration before the first move.
- **Make sure every crop contains the whole object.** A crop box from a coarser, earlier step quietly removed the same slice of the object from every view; only a metric that compared against an independent closed surface exposed it.

---

## 15. Future Work

### 15.1 Closing the size gap

The real cylinder is 3mm too wide ([§3.4](#34-the-real-cylinder-against-calipers)). Next: run `scan.py calibrate-board --execute` to measure the depth error and the mount against the ChArUco mat; apply the depth model and the board mount; drop or down-weight pixels near depth edges and readings that see the surface at a grazing angle; then re-scan and compare with the calipers again. The yellow and orange blocks should be measured too, since flat faces separate edge fattening from depth bias better than a cylinder.

### 15.2 Choosing the next best view

The scan now visits fixed rings of viewpoints. A next-best-view planner would choose each next pose for what it is expected to add:

- **Information gain:** cast rays from each candidate pose into the current TSDF and count voxels it would see for the first time (unknown) or improve (low weight, high variance). This is the classic next-best-view formulation, and the voxel grid already holds what it needs.
- **Objectives to compare:** fastest to a watertight, mostly measured mesh (measured share per second of robot time, including travel); smoothest surface (largest expected drop in roughness or per-voxel variance); most accurate where it matters (edges, or a region a task cares about); or least arm motion for a given coverage.
- **Constraints already in place:** every candidate can be checked with plan-only motion requests, and travel cost can come from the planned joint change.
- **How to judge it:** the progression metrics already measure what each added view buys. A next-best-view planner should reach, say, 85% measured or a given roughness in fewer views (and seconds) than the fixed rings, on the same objects.

### 15.3 Learning which objects to scan, and how

- **Recognize and separate objects** with a vision model (open-vocabulary detection and segmentation, served through Viam's ML model and vision services) instead of clustering points above the table. That also separates objects that touch, which clustering merges.
- **Prioritize** with a vision-language model (or a VLA policy): which objects matter for the task, which are likely to be hard (shiny, transparent, thin, hollow), and which sides are probably still unseen. A model's uncertainty, or the mesh's measured share, can say which object needs more views next.
- **Use semantic priors** where the camera can't see: a mug has a handle and a hollow inside; a box has flat, square faces. These could replace the generic straight-wall prior ([§6](#6-measured-vs-filled-in)) for known object types.

### 15.4 Calibration that just works

- **Board-based:** run `calibrate-board` routinely (and on a schedule), and keep its mount, depth model and intrinsics check with each session.
- **Known objects:** a caliper-measured object, such as this cylinder, is itself a calibration target: fit the depth model so the scanned object matches its known size.
- **Compare with Viam's camera-calibration module.** A technical interviewer suggested looking at `viam:camera-calibration`, which Viam's team is using to calibrate cameras for its surface-finishing project. It provides a ChArUco pose tracker (`viam:camera-calibration:charuco`) and a hand-eye calibration service (`viam:camera-calibration:handeye`). My earlier lab setup had both configured but disabled. Its methods should be compared with this project's scan-based and board-based calibrations, on the same arm and board.
- **Self-calibration at startup:** turn the camera on, look at whatever is on the table from a few poses, and solve for the mount and depth error together. The table plane (flat and level) and the board's level check already show that a few frames carry this information.
- **Watch for drift:** RealSense depth changes with temperature and time, so re-check (the level test takes one frame) and warn when it changes.

### 15.5 Two arms

Use a second arm to hold and turn the object while the camera arm scans it, or hold the camera still and let one arm present every side of the object. That would see the bottom (removing the flat-bottom assumption), reach sides a single arm can't, and avoid moving the object to a scan station. It needs:

- both arms in one frame system, with their relative pose calibrated (the ChArUco mat again);
- collision checking between the arms and the carried object;
- the gripper masked out of the depth images (its geometry is known from the frame system);
- the object's pose tracked as it moves, either from the gripper's pose or by registering each view to the growing model.

### 15.6 Also

- Full 37-view scans (including the 25° ring) of all three objects.
- Relocation on hardware, with the gripper measured.
- The stop-and-go vs. continuous capture comparison.
- Colored meshes (vertex colors are already fused; texture maps would need colored ICP and better color calibration).
- Package the scanner as a Viam module running next to the arm.

---

## 16. Feedback on Viam

### What Worked Well

- **Plan-only motion requests** (`do_command({"plan": ...})`) made reachability checks exact and fast (0.03-0.6s per pose) without writing any IK.
- **Targeting any frame** (here, the camera) in `motion.move` removed a whole class of transform bugs.
- **The frame system's collision geometry** for the table and walls was enforced automatically, and `WorldState` transforms let a carried object be attached to the gripper.
- **`transform_pose`** gave an authoritative check for my own pose math.

### Issues and Suggestions

- **Orientation vectors near the pole** read θ as the total turn about z. That's sensible but surprising, and nothing warns when a configured value falls inside that zone. A warning in the app, or accepting quaternions everywhere, would prevent a silent 40° error.
- **`RotationMatrix.elements` is column-major**; a note in the docstring would help.
- **The Python SDK's connection check** uses a fixed 1s timeout, which closes healthy connections during large transfers. It should be configurable or skip while a call is streaming.
- **`get_point_cloud()` over a cloud connection** dropped the connection at 1280×720; documenting a size limit, or a decimation option, would help.
- **Once, the frame system reported a pose 180° off the arm's joints.** I couldn't reproduce it, but a built-in consistency check (frame system vs. controller) would be valuable on shared arms.
