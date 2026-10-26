# ViamObjectReconstruction

Find the objects on a table, scan each one from many viewpoints with a wrist-mounted
Intel RealSense on a Viam-controlled arm, and fuse the views into a 3D mesh of each
object, separated from the table and the other objects. A single RealSense view is
noisy (about 1.5mm at 300mm); stitching many views averages most of that away.

## How it works

1. **Survey.** The camera looks straight down from one or more positions. The table
   plane is fit with RANSAC, and points standing above it are clustered into objects,
   each with a box (center, size, heading).
2. **Plan views.** Around each object, candidate camera poses sit on three rings
   (25°, 50° and 72° elevation, every 30°) plus one view straight down, all aimed at the
   object from a 280mm standoff. Each pose is checked with Viam's motion planner
   *without moving*: IK, joint limits, and collisions with the configured table and
   walls, the detected objects, and the measured table. The planner aims the camera
   frame directly. If a pose fails, other camera rolls are tried.
3. **Relocate if needed.** Coverage is the fraction of the object's surface directions
   that some reachable camera faces within 55°. Below 85% coverage, the object is picked
   up (top-down, jaws across its narrow side) and moved to the first free scan station
   that gives better coverage. It is put back afterward. While it is carried, the object
   is attached to the gripper in the planner's world state.
4. **Capture, stop-and-go.** At each view the arm stops, waits 0.6s, and takes the
   per-pixel median of 5 frames. Frames are accepted only if the camera stamped them
   after the settle time, by the robot's own clock, so frames buffered during motion are
   never used. The camera pose is read from the frame system after settling.
5. **Reconstruct.** Each view is placed in the world using its arm pose. One joint
   point-to-plane ICP over all overlapping view pairs then refines the poses, using the
   arm poses as a prior, much like a small bundle adjustment. The cropped depth maps are
   fused into a TSDF volume (Open3D `VoxelBlockGrid`, on CUDA when available). The mesh
   is extracted and cleaned, keeping only surfaces seen by at least 2 views.

Results on synthetic data (`python scan.py demo`), which simulates RealSense noise,
flying pixels, dropouts and a 2.5mm / 0.8° camera-mount calibration error:

| object | single view (median) | fused mesh shape (median / p90) |
|---|---|---|
| 70 x 40 x 55mm box | 1.43mm | 0.32mm / 0.45mm |
| 50mm x 90mm cylinder | 1.31mm | 0.07mm / 0.19mm |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the machine's API key in `apikey.json` as `{"key": "...", "key_id": "..."}`. It is
git-ignored. Machine address, resource names, workspace limits, survey positions, scan
stations and gripper settings live in `config.toml`. Every setting and its default is
documented in [recon/config.py](recon/config.py).

## Usage

The arm only moves with `--execute`. Without it, every command works from wherever the
arm is now and only plans.

```bash
python scan.py survey                    # find the table and objects (current pose)
python scan.py plan                      # reachable views, coverage and relocation per object
python scan.py plan --new-survey         # survey again first
python scan.py scan --execute --step     # full run, confirming each object and each relocation
python scan.py scan --execute --objects 0,2
python scan.py reconstruct               # rebuild meshes from the latest session's saved views
python scan.py reconstruct --no-refine   # compare: arm poses only
python scan.py demo                      # whole pipeline on synthetic data, no robot
python -m pytest                         # 24 tests, about 30s
```

Each session is saved under `data/<timestamp>/`. The raw views are kept, so
reconstruction can be re-run and tuned offline:

```
session.json                table plane, detected objects, intrinsics, config used
survey/view_000/            color.jpg, depth.png (uint16 mm), view.json (pose, joints, timestamps)
object_00/plan.json         reachable and unreachable views (with reasons), coverage, relocation
object_00/views/view_000/   captured views
object_00/recon/            mesh.ply, mesh.obj, cloud.ply, report.json, refined_poses.json
```

`report.json` lists how far ICP moved each view from its arm pose, and how far each
raw view lies from the fused surface. Large or one-sided corrections point to a
camera-mount calibration error.

## Code

| file | role |
|---|---|
| [scan.py](scan.py) | command line |
| [recon/pipeline.py](recon/pipeline.py) | survey, plan, relocate, capture, reconstruct workflow |
| [recon/robot.py](recon/robot.py) | everything that talks to Viam: plan-only checks, moves, still capture, world state |
| [recon/scene.py](recon/scene.py) | table plane and object detection |
| [recon/views.py](recon/views.py), [recon/planner.py](recon/planner.py) | candidate views, coverage, reachability |
| [recon/relocate.py](recon/relocate.py) | grasps, scan stations, pick and place |
| [recon/reconstruct.py](recon/reconstruct.py) | joint ICP with arm prior, TSDF fusion, mesh cleanup, metrics |
| [recon/rgbd.py](recon/rgbd.py), [recon/transforms.py](recon/transforms.py) | depth decoding and filtering, view files, pose math |
| [recon/synthetic.py](recon/synthetic.py) | noisy synthetic RGB-D renderer for tests and the demo |

## Things learned about the platform

- **Plan without moving:** the builtin motion service accepts
  `do_command({"plan": <MoveRequest as JSON>})`. It returns the joint trajectory, or
  the reason it failed (`out of reach`, `collision: gripper:claws and wall-front`).
  Planner geometry configured on the machine is enforced automatically.
- **Avoid `camera.get_point_cloud()` over a cloud connection:** the full 1280x720
  cloud dropped the connection on armfarm11. `get_images()` returns color plus raw depth
  (`image/vnd.viam.dep`: header, then big-endian uint16 mm), and back-projecting it with
  the intrinsics is faster and keeps pixel-level control.
- **`viam.spatialmath.RotationMatrix.elements` is column-major.** Reshaping it row-wise
  gives the transpose. That bug stays hidden for a camera looking straight down. This
  project uses its own orientation-vector math, tested against the machine's
  `transform_pose`.
- **Open3D 0.20 on Python 3.14:** the legacy `ScalableTSDFVolume` integrates nothing.
  The tensor `VoxelBlockGrid` works, and only extracts surfaces with weight at or above
  the threshold you give it.
- **Flying pixels** at depth edges shifted an object's detected center by 36mm until
  pixels next to depth jumps were dropped.

## Status on armfarm14 (2026-10-26)

Probed read-only and planned, **no motion yet**. It is an xArm6 with the camera on the
wrist. The survey from the current pose found all three objects (yellow block, orange
block, green cylinder), and 34-37 of 37 views plan for each, so none needs relocating.

Before running with `--execute`:

- **Camera calibration.** The measured table is tilted 2° and sits 15-30mm above the
  configured table obstacle (top at z=-23mm). The `cam` frame on this machine looks
  hand-entered. A hand-eye calibration (for example the `viam:camera-calibration`
  module) will tighten both the survey and the starting poses for stitching. Until
  then, the measured table is added as an extra obstacle around each object.
- **Gripper numbers** in `config.toml [relocate]`: maximum opening, fingertip offset
  from the `gripper` frame, and jaw axis. They are placeholders and only matter for
  relocation.

## Next steps

- First hardware run: `scan --execute --step --objects <one>`, then compare
  `reconstruct` with and without `--no-refine`.
- Relocation code has not run on hardware yet. Test pick and place on its own first.
- Test continuous capture against stop-and-go (earlier experience: pausing at each pose
  gives better clouds). The capture code records timestamps and joints per frame to
  support that comparison.
- Use the ICP corrections across many views to estimate the camera-mount error
  (self-calibration).
- Close the open bottom of each mesh (Poisson reconstruction) when a watertight model
  is needed.
