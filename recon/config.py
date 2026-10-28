''' Settings loaded from config.toml. Every value has a default, so the file only
needs the entries that differ. Distances are millimeters in the world frame. '''

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class MachineConfig:
    address: str = "armfarm11-main.310sld03v2.viam.cloud"
    credentials: str = "apikey.json"
    arm: str = "arm"
    camera: str = "cam"
    gripper: str = "gripper"
    # Names of the camera's color and depth images (get_images sources).
    color_source: str = "color"
    depth_source: str = "depth"
    motion: str = "builtin"
    # Our calibrated camera mount (camera pose on the arm's end), used for every move
    # and capture. Empty: use the camera frame configured on the machine instead.
    camera_mount_file: str = ""
    # Touch-off result (`scan.py touch-off`): how far the real fingertips reach past
    # the gripper's collision model, added to the table safety margin.
    touch_off_file: str = ""


@dataclass
class WorkspaceConfig:
    # Table region searched for objects.
    x_range: list[float] = field(default_factory=lambda: [150.0, 650.0])
    y_range: list[float] = field(default_factory=lambda: [-500.0, 180.0])
    # Camera positions for the survey; the camera looks straight down from each.
    survey_positions: list[list[float]] = field(default_factory=list)
    # Without survey_positions: a grid over the search region at this height, spaced so
    # neighboring views overlap (the image's short side covers ~0.79 x height).
    survey_height: float = 600.0
    survey_spacing: float = 340.0
    # Points within this horizontal distance of the arm's base are the arm itself (and,
    # on some cells, the cables around it); nothing that close is a scan target.
    min_base_distance: float = 200.0
    # Every plan treats the table as a slab whose top is this far ABOVE the measured
    # table (the machine's configured table can sit lower than the real one), with a
    # square of this half-size cut out around the arm's base. 15mm is judgment, set
    # after the gripper touched the table; the touch-off's extra reach is added to it.
    table_safety_margin: float = 15.0
    base_clearance: float = 200.0
    # Points count toward objects from this height above the table; an object's
    # top must reach `min_object_top`, which rejects sensor noise on the table.
    # Measured on real surveys from 600mm: noise bumps in bare table reach 10-23mm,
    # most at the edges of the image, so 25mm keeps them out (and also anything
    # flatter, such as cords). 8mm is just above the table's own noise there.
    min_object_height: float = 8.0
    min_object_top: float = 25.0
    # Scan targets, of any size, stand in open table: at least min_table_support
    # of the clear_margin around their footprint is bare table. Cords at the
    # table's edge, power strips and things in gaps between tables aren't; two
    # objects closer than clear_margin would reject each other. Whether a target
    # can be scanned all around is up to the view planner.
    # 60mm and 90% are judgment, checked on the armfarm14 table (they keep the blocks
    # and reject the power strip and cords).
    clear_margin: float = 60.0
    min_table_support: float = 0.9
    # Points higher than this above the table are ignored (taller than any object we scan).
    max_object_height: float = 500.0
    # Clusters with fewer points are noise (at 2mm voxels, 100 points is about 4cm² of surface).
    min_object_points: int = 100
    # Points within this distance belong to the same object (DBSCAN; judgment: wider
    # than gaps from missing depth on an object, narrower than the 60mm between objects).
    cluster_distance: float = 10.0
    # RANSAC table fit: inlier distance (survey noise at 600mm is a few mm) and the
    # steepest plane still accepted as the table.
    table_distance: float = 4.0
    max_table_tilt_deg: float = 10.0


@dataclass
class CameraConfig:
    # Depth readings outside this range (mm) are ignored. Chosen for the RealSense
    # on armfarm14: views are 280-500mm from the object, the survey about 600mm
    # from the table, and the floor beyond the table's edge is farther than 900mm.
    # Judgment, not tuned; check again for another camera (such as the Orbbec Astra 2).
    min_depth: float = 150.0
    max_depth: float = 900.0


@dataclass
class ViewConfig:
    # Distance from the camera to the nearest point of the object's bounding
    # sphere. RealSense D4xx depth fails closer than about 260mm at 1280x720.
    standoff: float = 280.0
    elevations_deg: list[float] = field(default_factory=lambda: [25.0, 50.0, 72.0])
    azimuth_step_deg: float = 30.0
    include_top_view: bool = True
    # Rolls about the optical axis to try; the one needing the least joint motion wins.
    rolls_deg: list[float] = field(default_factory=lambda: [0.0, 180.0, 90.0, -90.0])
    # Keep the last (wrist) joint within this many degrees of zero, so the camera cable doesn't wind up.
    max_wrist_deg: float = 200.0
    # A surface direction counts as seen when a camera lies within this angle of it
    # (judgment: beyond about 55-60deg a stereo camera's readings get sparse and noisy).
    coverage_angle_deg: float = 55.0
    # Below this coverage the object is moved to a scan station first.
    min_coverage: float = 0.85
    # Extra size (mm) on each side of an object's box when it is an obstacle (judgment:
    # covers the survey's few-mm position error and the arm's path tolerance).
    object_margin: float = 15.0
    # The camera aims this fraction of the way up the object (below the middle, so
    # low views also see the wall's base; judgment).
    aim_height_fraction: float = 0.4


@dataclass
class CaptureConfig:
    # Wait after the arm stops so vibration settles and frames taken in motion are flushed.
    settle_s: float = 0.6
    # Frames per view; their per-pixel median suppresses depth noise.
    frames_per_view: int = 5
    # Allowed time per frame.
    frame_timeout_s: float = 10.0
    # Frames per survey view: finding objects needs less precision than scanning them.
    survey_frames: int = 1
    move_timeout_s: float = 60.0


@dataclass
class RelocateConfig:
    enabled: bool = True
    # Candidate scan stations (x, y) with good reach all around. Checked in order.
    stations: list[list[float]] = field(default_factory=lambda: [[380.0, -150.0], [330.0, -250.0], [430.0, -60.0]])
    min_clearance: float = 40.0
    max_grasp_width: float = 80.0
    # How far the fingertips go below the object's top, capped at half its height.
    grasp_depth: float = 25.0
    # Distance from the gripper frame origin to the fingertip center, along the
    # gripper's approach axis. Measure this on the real gripper.
    fingertip_offset: float = 0.0
    approach_height: float = 100.0
    # The jaws are assumed to close along the gripper frame's y-axis; set this if
    # they close along another direction (degrees about the approach axis).
    jaw_axis_offset_deg: float = 0.0
    return_after_scan: bool = True


@dataclass
class ReconstructConfig:
    # TSDF voxel size (mm). A depth pixel covers about 0.33mm at 300mm, but a single
    # RealSense reading is only good to a few tenths of a mm and neighbors are
    # correlated, so finer voxels mostly record noise; 1.5mm keeps 30mm objects
    # recognizable (20 voxels across). Judgment for the RealSense, not tuned.
    voxel: float = 1.5
    # Truncation band (mm): each view updates voxels within this distance of its reading.
    # It must exceed how far the views disagree (about 0.5mm after alignment, several
    # mm before) and is usually 3-5 voxels (here 4). Judgment, not tuned.
    sdf_trunc: float = 6.0
    # A surface must be seen by at least this many views to appear in the mesh.
    min_views: int = 2
    # "auto" uses CUDA when Open3D was built with it, else "CPU:0".
    device: str = "auto"
    # Alignment uses every 2nd pixel in each direction (a quarter of them): plenty of
    # points for ICP, 4x faster. Fusion always uses every pixel.
    pixel_stride: int = 2
    # Also drop pixels this close to a depth edge, where stereo "edge fattening"
    # pushes surfaces outward (real cylinder: +2.3mm within 3px of its outline). 2px
    # takes the real cylinder from 32.62 to 32.22mm (calipers 29.53); wider gains
    # nothing there and costs accuracy on clean data. Survey detection doesn't use it.
    # Measured on RealSense data: test again for another camera (such as the Orbbec Astra 2).
    edge_grow: int = 2
    # Fusion keeps points within this margin around the object's box ...
    crop_margin: float = 12.0
    # ... and at least this far above the table.
    table_clearance: float = 2.5
    # Before fusing, the object's box is refit from the scan's own points found within
    # this far of the survey's box (the survey can be off by more than crop_margin).
    refit_search: float = 40.0
    # Width of the ring of table around the object used to fit the table per view.
    table_ring: float = 30.0
    # Registration also uses a ring of table around the object to lock height and tilt.
    context_margin: float = 10.0
    # Point spacing for alignment (mm): coarser than the fusion voxel, for speed.
    registration_voxel: float = 2.0
    # Joint ICP: correspondence distance (starts at twice this), iterations,
    # robust (Huber) threshold, and how much of a view pair must overlap. Tuned on
    # synthetic scans (relative pose error 1.87 -> 0.48mm; README section 8.3), then used
    # unchanged on real ones. Pairs more than max_pair_angle_deg apart share little surface.
    icp_max_distance: float = 3.0
    icp_iterations: int = 25
    huber_mm: float = 1.5
    points_per_pair: int = 1500
    min_pair_overlap: float = 0.2
    max_pair_angle_deg: float = 60.0
    # The arm's expected pose accuracy at the object, used as a prior: ICP
    # corrections beyond this need strong evidence from the geometry. Judgment from
    # the xArm6's 0.1mm repeatability plus calibration error at a 300mm lever arm.
    prior_translation_mm: float = 3.0
    prior_rotation_deg: float = 1.0
    # Spin about the vertical axis: unobservable for round objects, so held tighter.
    # Measured on the real cylinder: 1deg let 21 of 25 views spin and be rejected; 0.3deg
    # left 1 (README section 8.3).
    prior_spin_deg: float = 0.3
    # Corrections larger than this are treated as registration failures (judgment:
    # several times the prior, but well below the 21mm mount error a bad calibration gives).
    max_correction_mm: float = 15.0
    max_correction_deg: float = 5.0
    # Taubin smoothing steps on the observed mesh (Taubin, unlike plain Laplacian
    # smoothing, doesn't shrink the surface). On the real cylinder 5-15 steps and
    # Poisson depth 7-9 all gave roughness 0.112-0.116mm, so the light end was kept.
    smoothing_iterations: int = 5
    # Objects rest flat on the table, so the final mesh is closed with a flat bottom,
    # and walls never seen are assumed to drop straight down from the outline seen
    # from above (screened Poisson; see bottom.py). The outline is shrunk by
    # footprint_shrink mm because edge noise pads it.
    close_bottom: bool = True
    # Screened Poisson: an octree of depth 9 (2^9 = 512 cells across its box) over a box
    # 3x the object's size; a tighter box left few-view surfaces open (README section 8.4).
    poisson_depth: int = 9
    poisson_scale: float = 3.0
    footprint_shrink: float = 1.0
    # Wall-meets-table pixels flare outward; this lowest band of observed surface (mm)
    # is replaced by the straight-wall prior.
    base_band: float = 5.0
    # Mesh pieces smaller than this share of the largest are dropped as floating noise.
    min_component_fraction: float = 0.05


@dataclass
class BoardConfig:
    # Views captured by `scan.py calibrate-board`.
    views: int = 12
    # Frames per board view (the depth check uses their median).
    frames: int = 1
    # Height above the board for the survey's head-on view: the whole 180 x 135mm
    # board fits in the image's short side from about 250mm up.
    view_height: float = 330.0


@dataclass
class Config:
    machine: MachineConfig = field(default_factory=MachineConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    views: ViewConfig = field(default_factory=ViewConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    relocate: RelocateConfig = field(default_factory=RelocateConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    data_dir: str = "data"
    base_dir: Path = field(default=Path("."), repr=False)

    def resolve(self, path: str) -> Path:
        return (self.base_dir / path).resolve()

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("base_dir")
        return data


def _apply(target, values: dict, section: str) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown setting '{section}{key}' in config")
        current = getattr(target, key)
        if is_dataclass(current):
            _apply(current, value, f"{section}{key}.")
        else:
            setattr(target, key, value)


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    config = Config(base_dir=path.parent.resolve())
    if path.exists():
        with path.open("rb") as file:
            _apply(config, tomllib.load(file), "")
    return config
