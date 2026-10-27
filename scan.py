"""Scan the objects on the table and build a 3D mesh of each.

The arm only moves when you pass --execute. Without it, commands look and plan
from wherever the arm is now.

    python scan.py survey                 # find the table and objects from the current pose
    python scan.py plan                   # reachable views and coverage per object (no motion)
    python scan.py scan --execute --step  # the full run, confirming each object
    python scan.py scan --execute --confirm-moves --reuse-survey --objects 2 --max-views 3
    python scan.py calibrate --objects 1  # estimate the camera mount from a scanned object
    python scan.py reconstruct            # rebuild meshes from the latest session's saved views
    python scan.py progression --objects 1  # meshes and metrics from 1, 2, ... n views
    python scan.py demo                   # the whole pipeline on synthetic data, no robot needed
"""

import argparse
import asyncio
import logging
import time

import numpy as np

from recon import pipeline, synthetic
from recon.config import Config, load_config
from recon.progression import progression, shape_error
from recon.progression import table as progression_table
from recon.reconstruct import object_mask, surface_distances
from recon.rgbd import back_project, load_views, save_view
from recon.robot import MoveDeclined, PoseMismatch, Robot, connect
from recon.scene import detect_objects, fit_table_plane
from recon.session import Session
from recon.transforms import look_at, matrix_to_ov
from recon.views import generate_candidates, order_for_travel

log = logging.getLogger("scan")


def selected(objects, ids: list[int] | None):
    return [obj for obj in objects if ids is None or obj.id in ids]


async def run_robot(args, config: Config) -> None:
    machine = await connect(config)
    async with machine:
        robot = Robot(machine, config)
        robot.confirm_moves = args.confirm_moves or args.confirm_above is not None
        robot.confirm_above_deg = args.confirm_above
        try:
            await robot.checked_camera_pose()
            await run_steps(robot, args, config)
        except MoveDeclined as declined:
            log.info("Stopped: %s. The arm is where it was before that move.", declined)
        except PoseMismatch as mismatch:
            log.error("STOPPED, no further motion. %s", mismatch)
        except asyncio.CancelledError:
            # Ctrl+C: halt the arm before disconnecting. The e-stop remains the real safety.
            log.warning("Interrupted; stopping the arm")
            await robot.arm.stop()
            raise


async def run_steps(robot: Robot, args, config: Config) -> None:
    reuse = (args.command == "plan" and not args.new_survey) or (args.command == "scan" and args.reuse_survey)
    if reuse:
        session = Session.latest(config.resolve(config.data_dir))
        objects, plane = session.load_scene()
        robot.set_measured_table(plane)
        log.info("Using survey from %s", session.root)
    else:
        session = Session.create(config.resolve(config.data_dir))
        log.info("Session %s", session.root)
        objects, plane = await pipeline.survey(robot, session, config, args.execute)
    if args.command == "survey":
        return
    for obj in selected(objects, args.objects):
        others = [o for o in objects if o.id != obj.id]
        if args.command == "plan":
            plan, job, station_plan = await pipeline.plan_object(robot, obj, others, config)
            pipeline.save_plan(session, obj, plan, job, station_plan)
        else:
            await pipeline.scan_object(robot, session, obj, others, plane, config, args.execute, args.step,
                                       args.max_views)
    if args.command == "scan" and not args.execute:
        log.info("Planning only. Add --execute to move the arm (and --step to confirm each object).")


def run_calibrate(args, config: Config) -> None:
    """Estimate the camera mount from a scanned object and print it in Viam's frame format."""
    session = Session(args.session) if args.session else Session.latest(config.resolve(config.data_dir))
    if not args.objects or len(args.objects) != 1:
        raise SystemExit("calibrate needs one scanned object: --objects N (a scan of 20+ views works best)")
    calibration = pipeline.calibrate_session(session, args.objects[0], config)
    log.info("%s", calibration.describe())
    translation = calibration.mount[:3, 3]
    o_x, o_y, o_z, theta = matrix_to_ov(calibration.mount[:3, :3])
    log.info("Saved to %s; reconstructions of this session now use it.", session.root / pipeline.CALIBRATION_FILE)
    log.info("To use it for future captures, set the camera's frame (parent: arm) in the Viam app to:\n"
             '  "translation": {"x": %.2f, "y": %.2f, "z": %.2f},\n'
             '  "orientation": {"type": "ov_degrees", "value": {"x": %.5f, "y": %.5f, "z": %.5f, "th": %.3f}}',
             *translation, o_x, o_y, o_z, theta)


def run_reconstruct(args, config: Config) -> None:
    data_dir = config.resolve(config.data_dir)
    session = Session(args.session) if args.session else Session.latest(data_dir)
    objects, plane = session.load_scene()
    for obj in selected(objects, args.objects):
        if session.views_dir(obj.id).exists():
            pipeline.reconstruct_object(session, obj.id, plane, config, refine=not args.no_refine)


def run_progression(args, config: Config) -> None:
    """Meshes and metrics from 1, 2, ... n views of each selected object."""
    session = Session(args.session) if args.session else Session.latest(config.resolve(config.data_dir))
    objects, plane = session.load_scene()
    for obj in selected(objects, args.objects):
        if not session.views_dir(obj.id).exists():
            continue
        scanned = pipeline.scanned_object(session, obj.id)
        views = pipeline.apply_session_calibration(session, load_views(session.views_dir(obj.id)))
        folder = session.recon_dir(obj.id) / "progression"
        rows = progression(views, scanned, plane, config, folder, args.counts, args.order, not args.no_refine,
                           min_views=args.min_views)
        log.info("Object %d, 1 to %d views (%s order):\n%s\nMeshes and summary.csv in %s",
                 obj.id, len(views), args.order, progression_table(rows), folder)


def run_demo(config: Config, with_progression: bool = False) -> None:
    """Synthetic table scene: survey, plan views (all treated as reachable), render
    noisy views with a hand-eye error, save them, reconstruct, and score against
    the true shapes."""
    scene = synthetic.demo_scene()
    session = Session.create(config.resolve(config.data_dir), "demo-" + time.strftime("%Y%m%d-%H%M%S"))
    log.info("Demo session %s", session.root)
    # Survey from above the synthetic objects (the configured survey positions belong to the real cell).
    x, y = np.mean([o.mesh.get_center()[:2] for o in scene.objects], axis=0)
    survey = synthetic.render_views(scene, [look_at([x, y, 550], [x, y, 0])], seed=1, frames=config.capture.frames_per_view)[0]
    save_view(session.survey_dir / "view_000", survey.color, survey.depth, survey.T_world_cam, survey.intrinsics)
    points = pipeline.scene_points([session.survey_dir / "view_000"], config)
    plane = fit_table_plane(points, config.workspace)
    objects = detect_objects(points, plane, config.workspace)
    session.save_scene(objects, plane)
    for obj in objects:
        log.info("  %s", obj.describe())

    for obj in objects:
        truth_index = int(np.argmin([np.linalg.norm(o.mesh.get_center()[:2] - obj.center[:2]) for o in scene.objects]))
        poses = [c.pose() for c in order_for_travel(generate_candidates(obj, config.views))]
        log.info("Rendering %d views of object %d", len(poses), obj.id)
        views = synthetic.render_views(scene, poses, seed=10 + obj.id, frames=config.capture.frames_per_view)
        for index, view in enumerate(views):
            save_view(session.views_dir(obj.id) / f"view_{index:03d}", view.color, view.depth,
                      view.T_world_cam, view.intrinsics)
        result = pipeline.reconstruct_object(session, obj.id, plane, config)
        truth = scene.objects[truth_index].mesh
        single = []
        for view in views:
            pts, _, _ = back_project(view, 4, config.camera.min_depth, config.camera.max_depth)
            pts = pts[object_mask(pts, obj, plane, config.reconstruct)]
            if len(pts):
                single.append(np.median(surface_distances(truth, pts)))
        vertices = np.asarray(result.mesh.vertices)
        placed = surface_distances(truth, vertices)
        shape = shape_error(vertices, truth)
        log.info("  error vs. true shape: single view %.2fmm median -> fused mesh %.2fmm median, %.2fmm p90 "
                 "(%.2fmm median as placed; the rest is a shared offset from camera-mount error)",
                 np.median(single), np.median(shape), np.percentile(shape, 90), np.median(placed))
        if with_progression:
            rows = progression(views, obj, plane, config, session.recon_dir(obj.id) / "progression", truth=truth)
            log.info("Object %d, 1 to %d views:\n%s", obj.id, len(views), progression_table(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["survey", "plan", "scan", "calibrate", "reconstruct", "progression", "demo"])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--execute", action="store_true", help="allow the arm to move")
    parser.add_argument("--step", action="store_true", help="confirm before each object and each relocation")
    parser.add_argument("--confirm-moves", action="store_true", help="show and confirm every single arm move")
    parser.add_argument("--confirm-above", type=float, metavar="DEG",
                        help="show every move, but only ask when some joint turns at least DEG degrees")
    parser.add_argument("--max-views", type=int, help="scan: capture only the first N views of each object (top first)")
    parser.add_argument("--reuse-survey", action="store_true", help="scan: use the latest survey instead of surveying again")
    parser.add_argument("--objects", type=lambda s: [int(v) for v in s.split(",")], help="object ids, e.g. 0,2")
    parser.add_argument("--new-survey", action="store_true", help="plan: survey again instead of using the latest")
    parser.add_argument("--session", help="reconstruct: session folder (default: latest)")
    parser.add_argument("--no-refine", action="store_true", help="reconstruct: use arm poses without ICP refinement")
    parser.add_argument("--counts", type=lambda s: [int(v) for v in s.split(",")],
                        help="progression: view counts to build, e.g. 1,2,3,5,10,25 (default: every count)")
    parser.add_argument("--order", choices=["spread", "capture"], default="spread",
                        help="progression: add views most-different-first (spread) or in capture order")
    parser.add_argument("--progression", action="store_true", help="demo: also build the 1..n view progression")
    parser.add_argument("--min-views", type=int,
                        help="progression: keep surfaces seen by at least this many views (default: the config's, 2)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    for noisy in ("viam", "grpclib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config = load_config(args.config)

    if args.command == "demo":
        run_demo(config, args.progression)
    elif args.command == "progression":
        run_progression(args, config)
    elif args.command == "calibrate":
        run_calibrate(args, config)
    elif args.command == "reconstruct":
        run_reconstruct(args, config)
    else:
        asyncio.run(run_robot(args, config))


if __name__ == "__main__":
    main()
