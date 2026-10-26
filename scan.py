"""Scan the objects on the table and build a 3D mesh of each.

The arm only moves when you pass --execute. Without it, commands look and plan
from wherever the arm is now.

    python scan.py survey                 # find the table and objects from the current pose
    python scan.py plan                   # reachable views and coverage per object (no motion)
    python scan.py scan --execute --step  # the full run, confirming each object
    python scan.py reconstruct            # rebuild meshes from the latest session's saved views
    python scan.py demo                   # the whole pipeline on synthetic data, no robot needed
"""

import argparse
import asyncio
import logging
import time

import numpy as np
import open3d as o3d

from recon import pipeline, synthetic
from recon.config import Config, load_config
from recon.reconstruct import object_mask, surface_distances
from recon.rgbd import back_project, save_view
from recon.robot import Robot, connect
from recon.scene import detect_objects, fit_table_plane
from recon.session import Session
from recon.transforms import look_at
from recon.views import generate_candidates, order_for_travel

log = logging.getLogger("scan")


def selected(objects, ids: list[int] | None):
    return [obj for obj in objects if ids is None or obj.id in ids]


async def run_robot(args, config: Config) -> None:
    machine = await connect(config)
    async with machine:
        robot = Robot(machine, config)
        if args.command == "plan" and not args.new_survey:
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
                await pipeline.scan_object(robot, session, obj, others, plane, config, args.execute, args.step)
        if args.command == "scan" and not args.execute:
            log.info("Planning only. Add --execute to move the arm (and --step to confirm each object).")


def run_reconstruct(args, config: Config) -> None:
    data_dir = config.resolve(config.data_dir)
    session = Session(args.session) if args.session else Session.latest(data_dir)
    objects, plane = session.load_scene()
    for obj in selected(objects, args.objects):
        if session.views_dir(obj.id).exists():
            pipeline.reconstruct_object(session, obj.id, plane, config, refine=not args.no_refine)


def run_demo(config: Config) -> None:
    """Synthetic table scene: survey, plan views (all treated as reachable), render
    noisy views with a hand-eye error, save them, reconstruct, and score against
    the true shapes."""
    scene = synthetic.demo_scene()
    session = Session.create(config.resolve(config.data_dir), "demo-" + time.strftime("%Y%m%d-%H%M%S"))
    log.info("Demo session %s", session.root)
    x, y, z = config.workspace.survey_positions[0]
    survey = synthetic.render_views(scene, [look_at([x, y, z], [x, y, 0])], seed=1, frames=config.capture.frames_per_view)[0]
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
        shape = surface_distances(truth, vertices @ (fit := rigid_fit(vertices, truth))[:3, :3].T + fit[:3, 3])
        log.info("  error vs. true shape: single view %.2fmm median -> fused mesh %.2fmm median, %.2fmm p90 "
                 "(%.2fmm median as placed; the rest is a shared offset from camera-mount error)",
                 np.median(single), np.median(shape), np.percentile(shape, 90), np.median(placed))


def rigid_fit(points: np.ndarray, mesh) -> np.ndarray:
    """Best rigid transform of `points` onto `mesh`, to score shape apart from placement."""
    target = mesh.sample_points_uniformly(30000)
    target.estimate_normals()
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    return o3d.pipelines.registration.registration_icp(source, target, 5.0, np.eye(4), method).transformation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["survey", "plan", "scan", "reconstruct", "demo"])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--execute", action="store_true", help="allow the arm to move")
    parser.add_argument("--step", action="store_true", help="confirm before each object and each relocation")
    parser.add_argument("--objects", type=lambda s: [int(v) for v in s.split(",")], help="object ids, e.g. 0,2")
    parser.add_argument("--new-survey", action="store_true", help="plan: survey again instead of using the latest")
    parser.add_argument("--session", help="reconstruct: session folder (default: latest)")
    parser.add_argument("--no-refine", action="store_true", help="reconstruct: use arm poses without ICP refinement")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    for noisy in ("viam", "grpclib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config = load_config(args.config)

    if args.command == "demo":
        run_demo(config)
    elif args.command == "reconstruct":
        run_reconstruct(args, config)
    else:
        asyncio.run(run_robot(args, config))


if __name__ == "__main__":
    main()
