''' Candidate camera viewpoints around an object, and how much of it they see. '''

import math
from dataclasses import dataclass

import numpy as np

from .config import ViewConfig
from .scene import DetectedObject
from .transforms import look_at


@dataclass
class ViewCandidate:
    eye: np.ndarray
    target: np.ndarray
    azimuth_deg: float
    elevation_deg: float

    def pose(self, roll_deg: float = 0.0) -> np.ndarray:
        return look_at(self.eye, self.target, roll_deg)

    def describe(self) -> str:
        return f"az {self.azimuth_deg:5.0f} el {self.elevation_deg:4.0f}"


def aim_point(obj: DetectedObject, config: ViewConfig) -> np.ndarray:
    return np.array([obj.center[0], obj.center[1], obj.bottom_z + config.aim_height_fraction * obj.size[2]])


def view_distance(obj: DetectedObject, config: ViewConfig) -> float:
    return config.standoff + obj.radius


def generate_candidates(obj: DetectedObject, config: ViewConfig) -> list[ViewCandidate]:
    ''' Rings of views at each elevation, plus one straight down, all aimed at the object. '''
    target = aim_point(obj, config)
    distance = view_distance(obj, config)
    candidates = []
    if config.include_top_view:
        candidates.append(ViewCandidate(target + [0, 0, distance], target, 0.0, 90.0))
    count = max(3, round(360 / config.azimuth_step_deg))
    for ring, elevation in enumerate(sorted(config.elevations_deg, reverse=True)):
        # Stagger alternate rings so neighboring rings see different seams.
        offset = (ring % 2) * 180 / count
        for k in range(count):
            azimuth = (offset + 360 * k / count) % 360
            direction = _direction(azimuth, elevation)
            candidates.append(ViewCandidate(target + distance * direction, target, azimuth, elevation))
    return candidates


def _direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _sphere_directions(count: int = 600, min_z: float = -0.15) -> np.ndarray:
    ''' Evenly spread unit vectors (Fibonacci sphere), skipping those facing the table.

    Point i has height z = 1 - 2(i + 0.5)/count (equal-area bands) and turns by the
    golden angle, pi(1 + sqrt 5) = 2pi x golden ratio, from the last, which spreads the
    points evenly (Gonzalez, "Measurement of areas on a sphere using Fibonacci and
    latitude-longitude lattices", Mathematical Geosciences, 2010). 600 directions are
    about 8deg apart. Directions down to 0.15 below horizontal (about 9deg) are kept, so
    surface facing slightly downward (the lower half of a rounded object) still counts.
    '''
    i = np.arange(count) + 0.5
    z = 1 - 2 * i / count
    angle = math.pi * (1 + 5**0.5) * i
    r = np.sqrt(1 - z**2)
    directions = np.stack([r * np.cos(angle), r * np.sin(angle), z], axis=1)
    return directions[z >= min_z]


def coverage(target: np.ndarray, eyes: list[np.ndarray], config: ViewConfig) -> float:
    ''' Fraction of the object's outward surface directions that some camera faces closely enough.

    The bottom is excluded since it rests on the table and can never be seen.
    '''
    if not eyes:
        return 0.0
    surface = _sphere_directions()
    looks = np.array([(eye - target) / np.linalg.norm(eye - target) for eye in eyes])
    seen = (surface @ looks.T).max(axis=1) >= math.cos(math.radians(config.coverage_angle_deg))
    return float(seen.mean())


def order_for_travel(candidates: list[ViewCandidate]) -> list[ViewCandidate]:
    ''' Top view first, then ring by ring, reversing direction each ring to keep moves short. '''
    rings: dict[float, list[ViewCandidate]] = {}
    for candidate in candidates:
        rings.setdefault(candidate.elevation_deg, []).append(candidate)
    ordered: list[ViewCandidate] = []
    for index, elevation in enumerate(sorted(rings, reverse=True)):
        ring = sorted(rings[elevation], key=lambda c: c.azimuth_deg)
        if ordered and len(ring) > 1:
            # Start the ring at the view closest to where the previous ring ended.
            last = ordered[-1].eye
            start = int(np.argmin([np.linalg.norm(c.eye - last) for c in ring]))
            ring = ring[start:] + ring[:start]
            if index % 2 == 0:
                ring = ring[:1] + ring[1:][::-1]
        ordered.extend(ring)
    return ordered
