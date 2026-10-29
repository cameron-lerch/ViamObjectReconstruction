''' The live viewer: a web page on this computer that follows a scan as it runs.

    http://localhost:8765  (printed when a scan starts)

After each captured view the page shows the color and depth images, that view's
points, and the mesh fused from every view so far, in 3D (drag to rotate). A
second, monitoring camera (any Viam camera component, `[live] monitor_camera`)
can show the whole cell. Pause holds the arm still before its next move; End Scan
Session ends the run before the next move; STOP halts the arm at once, mid-move,
without asking. The e-stop is still the real safety.

Everything stays on this computer: the scan writes small preview files into
`data/live/`, and a web server in a background thread serves them with the page
(`web/live.html`) to the browser on this computer only.
'''

import asyncio
import json
import logging
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import open3d as o3d

from .config import Config
from .reconstruct import integrate, new_grid
from .rgbd import back_project, load_view
from .robot import MoveDeclined
from .scene import DetectedObject, TablePlane

log = logging.getLogger(__name__)

WEB = Path(__file__).resolve().parent.parent / "web"
POINT_STRIDE = 3  # every 3rd pixel of a view's points: plenty to see, a few hundred KB
POINT_MARGIN = 60.0  # points within this distance (mm) of the object's box
ALL_POINTS_VOXEL = 2.0  # the points from every view so far, thinned to one per 2mm cube


class StopRequested(MoveDeclined):
    ''' The operator stopped the scan from the live viewer. '''


class Live:
    def __init__(self, config: Config, folder: Path, read_only: bool = False):
        self.config = config
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self._lock = threading.Lock()
        self._status = {"state": "starting", "message": "", "session": "", "object": None,
                        "view": 0, "views": 0, "versions": {}, "log": [], "read_only": read_only,
                        "monitor": bool(config.live.monitor_camera)}
        self._log = deque(maxlen=40)
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._halt_now = None  # set by attach(): halts the arm from the server thread
        self._fusion = ThreadPoolExecutor(max_workers=1)  # one view at a time, in order
        self._grid = None
        self._all_points: o3d.geometry.PointCloud | None = None
        self._fused_views = 0
        self._server: ThreadingHTTPServer | None = None
        self._monitor_task: asyncio.Task | None = None

    # ---- the web server ----

    def start_server(self) -> str:
        handler = partial(_Handler, self)
        port = self.config.live.port
        for candidate in range(port, port + 10):  # the next free port if one is taken
            try:
                self._server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"No free port for the live viewer in {port}-{port + 9}")
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        logging.getLogger().addHandler(_Buffer(self._log))
        url = f"http://localhost:{self._server.server_address[1]}"
        log.info("Live viewer: %s", url)
        return url

    def close(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
        self._fusion.shutdown(wait=True)
        if self._server:
            self._server.shutdown()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "log": list(self._log), "paused": self._paused.is_set(),
                    "stopping": self._stop.is_set()}

    def update(self, **fields) -> None:
        with self._lock:
            self._status.update(fields)

    def _bump(self, *names: str) -> None:
        with self._lock:
            versions = dict(self._status["versions"])
            for name in names:
                versions[name] = versions.get(name, 0) + 1
            self._status["versions"] = versions

    # ---- control (called from the server thread) ----

    def pause(self) -> None:
        self._paused.set()
        log.info("Pause requested from the live viewer: the arm holds before its next move")

    def resume(self) -> None:
        self._paused.clear()
        log.info("Resumed from the live viewer")

    def stop(self, now: bool = False) -> None:
        self._stop.set()
        self._paused.clear()
        if now and self._halt_now:
            log.warning("STOP from the live viewer: halting the arm")
            self._halt_now()
        else:
            log.info("End Scan from the live viewer: the run ends before the next move")

    # ---- hooks for the scan (called on the asyncio loop) ----

    def attach(self, robot, loop: asyncio.AbstractEventLoop) -> None:
        ''' Let STOP halt the arm, and start the monitoring camera's feed. '''
        self._halt_now = lambda: asyncio.run_coroutine_threadsafe(robot.arm.stop(), loop)
        name = self.config.live.monitor_camera
        if name:
            from viam.components.camera import Camera
            self._monitor_task = loop.create_task(self._monitor(Camera.from_robot(robot.machine, name)))
        self.update(state="running")

    async def checkpoint(self) -> None:
        ''' Before every arm move: wait while paused, and stop if asked. '''
        if self._paused.is_set():
            state = self.status()["state"]
            self.update(state="paused")
            log.info("Paused (Resume or Stop in the live viewer)")
            while self._paused.is_set() and not self._stop.is_set():
                await asyncio.sleep(0.2)
            self.update(state=state)
        if self._stop.is_set():
            self.update(state="stopped")
            raise StopRequested("stopped from the live viewer")

    def captured(self, folder: Path) -> None:
        ''' Any capture (survey, board, scan): show its color and depth images. '''
        folder = Path(folder)
        if (folder / "view.json").exists():
            # Where the camera was, so the page can draw it in 3D and the world axes on its image.
            record = json.loads((folder / "view.json").read_text())
            lens = record["intrinsics"]
            self.update(camera={"T_world_cam": record["T_world_cam"],
                                **{key: lens[key] for key in ("fx", "fy", "cx", "cy", "width", "height")}})
        if (folder / "color.jpg").exists():
            shutil.copyfile(folder / "color.jpg", self.folder / "color.jpg")
            self._bump("color")
        if (folder / "depth_preview.jpg").exists():
            shutil.copyfile(folder / "depth_preview.jpg", self.folder / "depth.jpg")
            self._bump("depth")

    def start_object(self, session_name: str, obj: DetectedObject, plane: TablePlane, views: int) -> None:
        ''' A new object: start an empty mesh. '''
        self._fusion.submit(self._reset)
        self._object, self._plane = obj, plane
        self.update(state="scanning", session=session_name, view=0, views=views,
                    axes_origin=[float(obj.center[0]), float(obj.center[1]), float(obj.bottom_z)],
                    object={"id": obj.id, "description": obj.describe(), "center": obj.center.tolist(),
                            "size": obj.size.tolist()})

    def view_captured(self, folder: Path, index: int) -> None:
        ''' A scan view: fuse it into the object's mesh in the background (while the arm
        moves on) and refresh the page's 3D views. '''
        self.update(view=index + 1)
        self._fusion.submit(self._fuse, Path(folder), self._object, self._plane)

    def end_object(self) -> None:
        ''' The object's views are all captured: free the preview's grid (GPU memory)
        before the final reconstruction builds its own. '''
        self._fusion.submit(self._release).result()  # done before reconstruction starts

    def message(self, text: str, state: str | None = None) -> None:
        self.update(message=text, **({"state": state} if state else {}))

    # ---- background work ----

    def _release(self) -> None:
        self._grid = None
        self._all_points = None

    def _reset(self) -> None:
        self._grid = new_grid(self.config)
        self._all_points = None
        self._fused_views = 0

    def _fuse(self, folder: Path, obj: DetectedObject, plane: TablePlane) -> None:
        try:
            started = time.monotonic()
            view = load_view(folder)
            integrate(self._grid, view, view.T_world_cam, obj, plane, self.config)
            self._fused_views += 1
            # Surface seen by at least min(2, views) views, as the final mesh keeps.
            weight = min(self.config.reconstruct.min_views, self._fused_views) - 0.5
            mesh = self._grid.extract_triangle_mesh(weight).to_legacy()
            if len(mesh.triangles):
                o3d.io.write_triangle_mesh(str(self.folder / "mesh.ply"), mesh, write_ascii=False)
                self._bump("mesh")
            points, colors, _ = back_project(view, POINT_STRIDE, self.config.camera.min_depth,
                                             self.config.camera.max_depth, view.T_world_cam,
                                             edge_grow=self.config.reconstruct.edge_grow)
            near = obj.contains(points, POINT_MARGIN)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[near]))
            cloud.colors = o3d.utility.Vector3dVector(colors[near])
            o3d.io.write_point_cloud(str(self.folder / "points.ply"), cloud, write_ascii=False)
            self._all_points = cloud if self._all_points is None else self._all_points + cloud
            self._all_points = self._all_points.voxel_down_sample(ALL_POINTS_VOXEL)
            o3d.io.write_point_cloud(str(self.folder / "all_points.ply"), self._all_points, write_ascii=False)
            self._bump("points")
            self.update(fused=self._fused_views, triangles=len(mesh.triangles),
                        fuse_seconds=round(time.monotonic() - started, 1))
        except Exception:  # a preview must never stop the scan
            log.exception("Live preview of %s failed", folder.name)

    async def _monitor(self, camera) -> None:
        ''' The monitoring camera's latest frame, about once per `monitor_interval_s`. '''
        failed = False
        while True:
            try:
                images, _ = await camera.get_images()
                if images:
                    (self.folder / "monitor.jpg").write_bytes(images[0].data)
                    self._bump("monitor")
                failed = False
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not failed:
                    log.warning("Monitoring camera %r: %s (retrying)", self.config.live.monitor_camera, error)
                failed = True
            await asyncio.sleep(self.config.live.monitor_interval_s)


class _Buffer(logging.Handler):
    ''' The last lines of the scan's log, for the page. '''

    def __init__(self, lines: deque):
        super().__init__(logging.INFO)
        self.lines = lines

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(("viam", "grpclib", "asyncio")):
            return
        for line in self.format(record).splitlines():
            if line.strip():
                self.lines.append(line)


class _Handler(SimpleHTTPRequestHandler):
    ''' The page, the preview files, the status, and the Pause/Resume/Stop buttons. '''

    def __init__(self, live: Live, *args, **kwargs):
        self.live = live
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, *args) -> None:  # keep the terminal for the scan's own log
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.path = "/live.html"
            return super().do_GET()
        if path == "/api/status":
            return self._json(self.live.status())
        if path.startswith("/files/"):
            name = Path(path.removeprefix("/files/")).name  # no folders: only the preview files
            file = self.live.folder / name
            if not file.exists():
                return self.send_error(404)
            body = file.read_bytes()
            self.send_response(200)
            kind = {".jpg": "image/jpeg", ".png": "image/png"}.get(file.suffix, "application/octet-stream")
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        return super().do_GET()

    def do_POST(self) -> None:
        if self.live.read_only:
            return self.send_error(403, "This viewer is showing a finished scan")
        actions = {"/api/pause": self.live.pause, "/api/resume": self.live.resume,
                   "/api/stop": self.live.stop, "/api/stop-now": lambda: self.live.stop(now=True)}
        action = actions.get(self.path)
        if action is None:
            return self.send_error(404)
        action()
        return self._json(self.live.status())

    def _json(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
