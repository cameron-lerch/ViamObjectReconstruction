''' Scan the objects on the table and build a 3D mesh of each.

    python scan.py                  # the whole scan: asks once before the arm moves, then before each object
    python scan.py --plan-only      # plan every object on the latest survey (no motion)
    python scan.py --max-views 16   # about half the time: 16 views per object, spread over all directions
    python scan.py --full-scan      # 50 frames per view, every raw frame saved (a dataset; about 2 min per view)
    python scan.py --full-scan --resume data/20261029-155544   # finish an interrupted scan (same options)

The steps on their own, for troubleshooting (the arm only moves with --execute;
without it, they look and plan from wherever the arm is now):

    python scan.py survey                 # find the table and objects from the current pose
    python scan.py plan                   # reachable views and coverage per object (no motion)
    python scan.py scan --execute --step  # the full run, confirming each object
    python scan.py scan --execute --confirm-moves --reuse-survey --objects 2 --max-views 3
    python scan.py calibrate --objects 1  # estimate the camera mount from a scanned object
    python scan.py calibrate-board        # ChArUco board: plan views (add --execute to capture and solve)
    python scan.py reconstruct            # rebuild meshes from the latest session's saved views
    python scan.py progression --objects 1  # meshes and metrics from 1, 2, ... n views
    python scan.py compare --objects 1    # measure the mesh and compare with ground_truth.toml
    python scan.py demo                   # the whole pipeline on synthetic data, no robot needed
'''

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from recon import pipeline, synthetic
from recon.config import Config, load_config
from recon.measure import compare, identify, load_truth, measure
from recon.figures import doubling_counts, facing_the_arm, progression_figure
from recon.progression import progression, shape_error
from recon.progression import table as progression_table
from recon.reconstruct import object_mask, surface_distances
from recon.rgbd import back_project, load_views, save_view
from recon.live import Live
from recon.robot import MoveDeclined, PoseMismatch, Robot, connect
from recon.scene import detect_objects, fit_table_plane
from recon.session import Session
from recon.touchoff import touch_off
from recon.transforms import look_at, matrix_to_ov
from recon.views import generate_candidates, order_for_travel

log = logging.getLogger("scan")


def selected(objects, wanted: list[int] | None):
    ''' Objects by the ids the survey gave them (all of them without `--objects`). '''
    return [obj for obj in objects if wanted is None or obj.id in wanted]


def object_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]


async def with_robot(config: Config, steps, confirm_moves: bool = False, confirm_above: float | None = None,
                     go_ready: bool = False) -> None:
    ''' Connect, check the arm's pose, run `steps(robot)`, and stop safely on any interruption.
    With `go_ready` (any run that moves the arm), first step to the ready pose (Robot.go_ready). '''
    machine = await connect(config)
    async with machine:
        robot = Robot(machine, config)
        robot.confirm_moves = confirm_moves or confirm_above is not None
        robot.confirm_above_deg = confirm_above
        if config.live.enabled:
            robot.live = Live(config, config.resolve(config.data_dir) / "live")
            robot.live.start_server()
            robot.live.attach(robot, asyncio.get_running_loop())
        try:
            await robot.checked_camera_pose()
            if go_ready:
                await robot.go_ready()
            await steps(robot)
        except MoveDeclined as declined:
            log.info("Stopped: %s. The arm is where it was before that move.", declined)
        except PoseMismatch as mismatch:
            log.error("STOPPED, no further motion. %s", mismatch)
        except asyncio.CancelledError:
            # Ctrl+C: halt the arm before disconnecting. The e-stop remains the real safety.
            log.warning("Interrupted; stopping the arm")
            await robot.arm.stop()
            raise
        finally:
            if robot.live:
                robot.live.message("Finished; the arm is parked or stopped", "done")
                robot.live.close()


async def run_robot(args, config: Config) -> None:
    await with_robot(config, lambda robot: run_steps(robot, args, config), args.confirm_moves, args.confirm_above,
                     go_ready=args.execute)


async def touch_off_step(robot: Robot, config: Config, execute: bool, session: Session | None = None) -> dict | None:
    ''' Touch-off on a survey's table (the latest survey unless `session` is given); saves the result. '''
    if not config.machine.gripper:
        log.info("Touch-off is off: no gripper is mounted ([machine] gripper = \"\" in config.toml).")
        return None
    session = session or Session.latest(config.resolve(config.data_dir), surveyed=True)
    objects, plane = session.load_scene()
    robot.set_scene(plane, session.load_clutter())
    board = session.read().get("board")
    log.info("Using the survey in %s", session.root)
    survey_points = pipeline.scene_points(sorted(session.survey_dir.glob("view_*")), config)
    result = await touch_off(robot, objects, plane, config, execute, board["center"] if board else None, survey_points)
    if result:
        path = config.resolve(config.machine.touch_off_file or "calibration/touch_off.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result | {"session": session.root.name}, indent=2) + "\n")
        log.info("Fingertips reach %.1fmm past the gripper frame (spread %.1fmm over %d spots); the collision model says %.1fmm.",
                 result["reach_mm"], result["reach_spread_mm"], len(result["points"]), result["modeled_reach_mm"])
        if result["extra_reach_mm"] > 0:
            log.info("The real fingertips reach %.1fmm further than the model: every plan now adds that to the table margin.",
                     result["extra_reach_mm"])
        else:
            log.info("The model covers the real fingertips; the table margin stays at %.0fmm.", config.workspace.table_safety_margin)
        tilt = result["table_vs_camera"]
        if tilt:
            log.info("The table the arm touched vs the camera's table: tilted %.2fdeg (%.2fmm per 100mm); the reach spread above includes this.",
                     tilt["tilt_deg"], tilt["tilt_mm_per_100mm"])
        log.info("Saved to %s", path)
    return result


async def board_step(robot: Robot, config: Config, execute: bool, save_mount: bool) -> dict | None:
    ''' Calibrate the camera with the ChArUco board (starting from where the last survey saw it, if any). '''
    known = None
    try:
        board = Session.latest(config.resolve(config.data_dir)).read().get("board")
        known = np.array(board["center"]) if board else None
    except FileNotFoundError:
        pass
    session = Session.create(config.resolve(config.data_dir), "board-" + time.strftime("%Y%m%d-%H%M%S"))
    result = await pipeline.board_calibration(robot, session, config, execute, known_center=known)
    if result:
        report_board(result, session)
        if save_mount:
            path = pipeline.save_mount(config, np.array(result["board_mount"]), f"board calibration, {session.root.name}")
            log.info("Saved as the camera mount for all moves and captures: %s", path)
    return result


async def survey_step(robot: Robot, config: Config, execute: bool):
    ''' A new survey session: the table, the objects on it and the board. Returns (session, objects, plane). '''
    session = Session.create(config.resolve(config.data_dir))
    log.info("Session %s", session.root)
    objects, plane = await pipeline.survey(robot, session, config, execute)
    return session, objects, plane


async def run_steps(robot: Robot, args, config: Config) -> None:
    if args.command == "touch-off":
        await touch_off_step(robot, config, args.execute)
        return
    if args.command == "calibrate-board":
        await board_step(robot, config, args.execute, args.save_mount)
        return
    reuse = (args.command == "plan" and not args.new_survey) or (args.command == "scan" and args.reuse_survey)
    if reuse:
        session = Session.latest(config.resolve(config.data_dir), surveyed=True)
        objects, plane = session.load_scene()
        robot.set_scene(plane, session.load_clutter())
        log.info("Using survey from %s", session.root)
    else:
        session, objects, plane = await survey_step(robot, config, args.execute)
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
    if args.command == "scan" and args.execute:
        await pipeline.go_home(robot, session, config)
        # Meshes and figures only after the arm is parked: they take a while and need no robot.
        for obj in selected(objects, args.objects):
            if session.views_dir(obj.id).exists():
                pipeline.finish_object(session, obj.id, plane, config)
    if args.command == "scan" and not args.execute:
        log.info("Planning only. Add --execute to move the arm (and --step to confirm each object).")


def viam_frame(mount: np.ndarray) -> str:
    o_x, o_y, o_z, theta = matrix_to_ov(mount[:3, :3])
    x, y, z = mount[:3, 3]
    return ('  "translation": {"x": %.2f, "y": %.2f, "z": %.2f},\n'
            '  "orientation": {"type": "ov_degrees", "value": {"x": %.5f, "y": %.5f, "z": %.5f, "th": %.3f}}'
            % (x, y, z, o_x, o_y, o_z, theta))


def describe_for_person(obj) -> str:
    x, y, _ = obj.center
    length, width, height = obj.size
    color = f"{obj.color_name}, " if obj.color_name else ""
    return f"object {obj.id}: {color}{length:.0f} x {width:.0f} x {height:.0f}mm, centered at ({x:.0f}, {y:.0f})"


def ask_object(obj) -> str:
    ''' 'y' (scan it), 'n' (skip), 'a' (scan this and all the rest without asking) or 'q' (stop). '''
    while True:
        answer = input(f"Scan {describe_for_person(obj)}? [Enter = yes, n = skip, a = all the rest, q = stop] ").strip().lower()
        if answer in ("", "y", "yes"):
            return "y"
        if answer in ("n", "a", "q"):
            return answer
        print("  Please type Enter, n, a or q.")


def latest_survey(robot: Robot, config: Config):
    ''' The latest saved survey, for planning only: wherever the camera happens to be now
    (a fresh look from a pose too close to the table finds nothing). '''
    session = Session.latest(config.resolve(config.data_dir), surveyed=True)
    objects, plane = session.load_scene()
    robot.set_scene(plane, session.load_clutter())
    log.info("Planning on the survey in %s (run calibration.py or scan.py to survey again)", session.root)
    for obj in objects:
        log.info("  %s", obj.describe())
    return session, objects, plane


async def scan_everything(robot: Robot, config: Config, execute: bool, max_views: int | None,
                          truth_file: str = "ground_truth.toml", resume: str | None = None) -> None:
    ''' Survey, list what's on the table, scan each object (asking first), park the arm,
    then build every mesh and figure and compare with the calipers where possible.

    With `resume` (a session folder), continue that session instead: its survey is reused,
    and objects it already started are checked for movement and then finished, skipping
    the views already captured. '''
    if resume:
        session = Session(Path(resume))
        objects, plane = session.load_scene()
        robot.set_scene(plane, session.load_clutter())
        log.info("Resuming %s (its survey; the objects must not have moved)", session.root)
    elif execute:
        session, objects, plane = await survey_step(robot, config, execute)
    else:
        session, objects, plane = latest_survey(robot, config)
    if not objects:
        log.info("No objects to scan on the table. (Things too low, or not standing in open table, are skipped; see the survey output above.)")
        return
    log.info("\nFound %d object(s) to scan:", len(objects))
    started = {obj.id: len(pipeline.captured_views(session, obj.id)) if resume else 0 for obj in objects}
    for obj in objects:
        log.info("  %s%s", describe_for_person(obj), f" ({started[obj.id]} views captured)" if started[obj.id] else "")

    scanned, ask_each = [], True
    for obj in objects:
        others = [o for o in objects if o.id != obj.id]
        if not execute and started[obj.id]:
            plan = await pipeline.plan_scan(robot, pipeline.scanned_object(session, obj.id), others, config)
            left = pipeline.missing_views(plan.views, pipeline.captured_views(session, obj.id))
            log.info("Object %d: %d views captured, %d still to capture", obj.id, started[obj.id], len(left))
            continue
        if not execute:
            plan, job, station_plan = await pipeline.plan_object(robot, obj, others, config)
            pipeline.save_plan(session, obj, plan, job, station_plan)
            continue
        answer = await asyncio.to_thread(ask_object, obj) if ask_each else "y"
        if answer == "q":
            break
        if answer == "n":
            continue
        if answer == "a":
            ask_each = False
        log.info("\nScanning %s", describe_for_person(obj))
        if started[obj.id]:
            try:
                await pipeline.check_unmoved(robot, session, obj, others, plane, config)
            except pipeline.ObjectMoved as moved:
                log.error("  %s", moved)
                continue
        await pipeline.scan_object(robot, session, obj, others, plane, config, execute, False, max_views,
                                   resume=bool(started[obj.id]))
        scanned.append(obj)
    if not execute:
        log.info("\nPlanned only (nothing moved). Run python scan.py to scan.")
        return

    await pipeline.go_home(robot, session, config)
    # Meshes, figures and comparisons after the arm is parked: they take a while and need no robot.
    truth = load_truth(config.resolve(truth_file))
    live = getattr(robot, "live", None)
    for obj in scanned:
        log.info("\nBuilding object %d's meshes and figures", obj.id)
        pipeline.finish_object(session, obj.id, plane, config)
        compare_object(session, obj, truth, truth_file)
        if live:
            live.show_final(session.recon_dir(obj.id) / "mesh.ply")
    log.info("\nDone: %d object(s) scanned. Results in %s/object_NN/ (recon/mesh.obj, recon/observed.obj, progression.png)",
             len(scanned), session.root)
    if live and scanned:
        # The viewer stays open, with the final mesh, until this last prompt is answered.
        views = sum(len(pipeline.captured_views(session, obj.id)) for obj in scanned)
        live.complete(f"{views} views, final mesh ready")
        await asyncio.to_thread(input, f"\nScan completed {views} views with {config.capture.frames_per_view} frames per view. "
                                       f"\nData saved and final mesh generated in {session.root} "
                                       "\n Press ENTER to end the program and close the webapp.")


def report_board(result: dict, session: Session) -> None:
    original, board = np.array(result["original_mount"]), np.array(result["board_mount"])
    change = np.linalg.inv(original) @ board
    eye = result["hand_eye"]
    log.info("Hand-eye from %d views (%d corners): reprojection %.2f -> %.2fpx", eye["views"], eye["corners"],
             eye["reprojection_rms_px_before"], eye["reprojection_rms_px_after"])
    log.info("Board-calibrated mount differs from the configured one by %.1fmm / %.2fdeg:\n%s",
             np.linalg.norm(change[:3, 3]), np.degrees(np.linalg.norm(Rotation.from_matrix(change[:3, :3]).as_rotvec())),
             viam_frame(board))
    depth = result["depth"]
    if depth:
        log.info("Depth vs. board over %d corners at %.0f-%.0fmm: median error %+.2fmm; fit %+.2fmm %+.2f%% of distance (rms %.2fmm)",
                 depth["corners"], *depth["distance_range_mm"], depth["median_error_mm"], depth["offset_mm"],
                 depth["scale_percent"], depth["fit_rms_mm"])
    log.info("Saved to %s", session.root / "board" / "board_calibration.json")


def run_calibrate(args, config: Config) -> None:
    ''' Estimate the camera mount from a scanned object and print it in Viam's frame format. '''
    session = Session(args.session) if args.session else Session.latest(config.resolve(config.data_dir))
    if not args.objects or len(args.objects) != 1:
        raise SystemExit("calibrate needs one scanned object: --objects N (a scan of 20+ views works best)")
    calibration = pipeline.calibrate_session(session, args.objects[0], config)
    log.info("%s", calibration.describe())
    log.info("Saved to %s; reconstructions of this session now use it.", session.root / pipeline.CALIBRATION_FILE)
    if args.save_mount:
        path = pipeline.save_mount(config, calibration.mount, f"scan.py calibrate on {session.root.name}, object {args.objects[0]}")
        log.info("Saved as the camera mount for all moves and captures: %s", path)
    else:
        log.info("Add --save-mount to use it for all moves and captures (%s).", config.machine.camera_mount_file)
    log.info("For the Viam app's 3D scene only (optional), the camera frame (parent: arm) would be:\n%s",
             viam_frame(calibration.mount))


def run_reconstruct(args, config: Config) -> None:
    data_dir = config.resolve(config.data_dir)
    session = Session(args.session) if args.session else Session.latest(data_dir)
    objects, plane = session.load_scene()
    for obj in selected(objects, args.objects):
        if session.views_dir(obj.id).exists():
            pipeline.reconstruct_object(session, obj.id, plane, config, refine=False if args.no_refine else None)


def compare_object(session: Session, obj, truth: dict, truth_file: str, name: str | None = None) -> dict | None:
    ''' Measure an object's mesh like calipers would and compare it with the ground-truth
    entry it resembles (or `name`). Returns the errors, or None if nothing to compare. '''
    folder = session.recon_dir(obj.id)
    if not (folder / "mesh.ply").exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(folder / "mesh.ply"))
    if name is None:
        name, _, why = identify(mesh, truth)
        if name is None:
            log.info("Object %d %s: no similar object in %s, so no caliper comparison", obj.id, why, truth_file)
            return None
        log.info("Object %d %s: comparing with %s", obj.id, why, name)
    entry = truth[name]
    measured = measure(mesh, entry["shape"])
    rows = compare(measured, entry)
    lines = [f"  {key:9s} {got:7.2f}mm  vs {want:6.2f}mm  {error:+6.2f}mm ({100 * error / want:+.1f}%)"
             for key, got, want, error in rows]
    log.info("Object %d vs %s (calipers):\n%s", obj.id, name, "\n".join(lines))
    errors = {key: error for key, _, _, error in rows}
    session.write_json(folder / "accuracy.json", {"truth": name, "measured": measured, "errors": errors})
    return errors


def run_compare(args, config: Config) -> None:
    ''' Measure reconstructed meshes like calipers would and compare them with ground_truth.toml. '''
    truth = load_truth(config.resolve(args.truth_file))
    session = Session(args.session) if args.session else Session.latest(config.resolve(config.data_dir))
    objects, _ = session.load_scene()
    for obj in selected(objects, args.objects):
        compare_object(session, obj, truth, args.truth_file, args.truth)


def run_progression(args, config: Config) -> None:
    ''' Meshes and metrics from 1, 2, ... n views of each selected object. '''
    session = Session(args.session) if args.session else Session.latest(config.resolve(config.data_dir))
    objects, plane = session.load_scene()
    for obj in selected(objects, args.objects):
        if not session.views_dir(obj.id).exists():
            continue
        scanned = pipeline.scanned_object(session, obj.id)
        views = pipeline.apply_session_calibration(session, load_views(session.views_dir(obj.id)))
        folder = session.recon_dir(obj.id) / "progression"
        rows = progression(views, scanned, plane, config, folder, args.counts, args.order, False if args.no_refine else None,
                           min_views=args.min_views)
        log.info("Object %d, 1 to %d views (%s order):\n%s\nMeshes and summary.csv in %s",
                 obj.id, len(views), args.order, progression_table(rows), folder)
        counts = args.counts or doubling_counts(len(views))
        progression_figure(folder, session.object_dir(obj.id) / "progression.svg",
                           f"Object {obj.id}: reconstruction from 1 to {len(views)} views, seen from the arm's side",
                           sorted(c for c in counts if c <= len(views)), azimuth=facing_the_arm(scanned.center))


def run_demo(config: Config, with_progression: bool = False) -> None:
    ''' Synthetic table scene: survey, plan views (all treated as reachable), render
    noisy views with a hand-eye error, save them, reconstruct, and score against
    the true shapes. '''
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
    parser.add_argument("command", nargs="?", choices=["survey", "plan", "scan", "calibrate", "calibrate-board", "touch-off", "reconstruct", "progression", "compare", "demo"],
                        help="one step on its own (leave out for the whole scan)")
    parser.add_argument("--plan-only", action="store_true", help="the whole scan, planned on the latest survey, without moving")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--execute", action="store_true", help="allow the arm to move")
    parser.add_argument("--step", action="store_true", help="confirm before each object and each relocation")
    parser.add_argument("--confirm-moves", action="store_true", help="show and confirm every single arm move")
    parser.add_argument("--confirm-above", type=float, metavar="DEG",
                        help="show every move, but only ask when some joint turns at least DEG degrees")
    parser.add_argument("--resume", metavar="SESSION",
                        help="continue an interrupted scan in that session folder (e.g. data/20261029-155544): "
                             "reuses its survey, checks the objects haven't moved, skips views already captured")
    parser.add_argument("--full-scan", action="store_true",
                        help="50 frames per scan view, and every raw frame saved (a dataset to subsample later; "
                             "about 2 minutes per view)")
    parser.add_argument("--max-views", type=int, help="scan: capture only N of each object's views, spread over all directions (16 measured every side of the test blocks)")
    parser.add_argument("--reuse-survey", action="store_true", help="scan: use the latest survey instead of surveying again")
    parser.add_argument("--objects", type=object_list, help="object ids from the survey, e.g. 0,2 (default: all)")
    parser.add_argument("--new-survey", action="store_true", help="plan: survey again instead of using the latest")
    parser.add_argument("--session", help="reconstruct: session folder (default: latest)")
    parser.add_argument("--no-refine", action="store_true", help="reconstruct: use arm poses without ICP refinement")
    parser.add_argument("--truth", help="compare: the ground_truth.toml entry to compare with")
    parser.add_argument("--truth-file", default="ground_truth.toml", help="compare: caliper measurements")
    parser.add_argument("--save-mount", action="store_true",
                        help="calibrate, calibrate-board: use the result as the camera mount for all moves and captures")
    parser.add_argument("--counts", type=lambda s: [int(v) for v in s.split(",")],
                        help="progression: view counts to build, e.g. 1,2,3,5,10,25 (default: every count)")
    parser.add_argument("--order", choices=["spread", "capture"], default="spread",
                        help="progression: add views most-different-first (spread) or in capture order")
    parser.add_argument("--progression", action="store_true", help="demo: also build the 1..n view progression")
    parser.add_argument("--min-views", type=int,
                        help="progression: keep surfaces seen by at least this many views (default: min(2, number of views))")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    for noisy in ("viam", "grpclib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config = load_config(args.config)
    if args.full_scan:
        config.capture.frames_per_view = config.capture.full_scan_frames
        config.capture.keep_frames = True
        log.info("Full scan: %d frames per view, every raw frame saved (about %.0f minutes per view)",
                 config.capture.frames_per_view, config.capture.frames_per_view * 2.6 / 60)

    if args.command is None:
        execute = not args.plan_only
        # About 45 views per object; each is a move (about 6s) plus its frames (about 2.6s each).
        minutes = 45 * (6 + 2.6 * config.capture.frames_per_view) / 60
        if execute and input("The arm will move to survey the table, then scan each object you choose "
                             f"(about {minutes:.0f} minutes each; Ctrl+C, the live viewer's Stop, or the e-stop stops it). "
                             "Start? [y/N] ").strip().lower() not in ("y", "yes"):
            return
        asyncio.run(with_robot(config, lambda robot: scan_everything(robot, config, execute, args.max_views, args.truth_file,
                                                                     args.resume),
                               args.confirm_moves, args.confirm_above, go_ready=execute))
    elif args.command == "demo":
        run_demo(config, args.progression)
    elif args.command == "progression":
        run_progression(args, config)
    elif args.command == "compare":
        run_compare(args, config)
    elif args.command == "calibrate":
        run_calibrate(args, config)
    elif args.command == "reconstruct":
        run_reconstruct(args, config)
    else:
        asyncio.run(run_robot(args, config))


if __name__ == "__main__":
    main()
