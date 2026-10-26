"""On-disk layout of one scanning session.

data/<session>/
    session.json            table plane, detected objects, camera intrinsics, config used
    survey/view_000/        overhead views used to find the objects
    object_00/plan.json     reachable views, coverage, relocation decision
    object_00/views/...     the captured views of this object
    object_00/recon/        mesh.ply, mesh.obj, cloud.ply, report.json
"""

import json
import time
from pathlib import Path

from .scene import DetectedObject, TablePlane


class Session:
    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def create(cls, data_dir: Path, name: str | None = None) -> "Session":
        root = Path(data_dir) / (name or time.strftime("%Y%m%d-%H%M%S"))
        root.mkdir(parents=True, exist_ok=False)
        return cls(root)

    @classmethod
    def latest(cls, data_dir: Path) -> "Session":
        files = sorted(Path(data_dir).glob("*/session.json"), key=lambda path: path.stat().st_mtime)
        sessions = [path.parent for path in files]
        if not sessions:
            raise FileNotFoundError(f"No sessions in {data_dir}; run a survey first")
        return cls(sessions[-1])

    @property
    def survey_dir(self) -> Path:
        return self.root / "survey"

    def object_dir(self, object_id: int) -> Path:
        return self.root / f"object_{object_id:02d}"

    def views_dir(self, object_id: int) -> Path:
        return self.object_dir(object_id) / "views"

    def recon_dir(self, object_id: int) -> Path:
        return self.object_dir(object_id) / "recon"

    def read(self) -> dict:
        path = self.root / "session.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def update(self, **values) -> None:
        data = self.read()
        data.update(values)
        (self.root / "session.json").write_text(json.dumps(data, indent=2))

    def save_scene(self, objects: list[DetectedObject], plane: TablePlane) -> None:
        self.update(plane=plane.to_dict(), objects=[obj.to_dict() for obj in objects])

    def load_scene(self) -> tuple[list[DetectedObject], TablePlane]:
        data = self.read()
        if "plane" not in data:
            raise ValueError(f"Session {self.root} has no survey results")
        return [DetectedObject.from_dict(o) for o in data["objects"]], TablePlane.from_dict(data["plane"])

    def write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
