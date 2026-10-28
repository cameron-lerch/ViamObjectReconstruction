"""--max-views: a subset of the planned views spread over all directions."""

import numpy as np

from recon.config import Config
from recon.planner import PlannedView, spread_views
from recon.scene import DetectedObject
from recon.views import generate_candidates, order_for_travel


def planned() -> list[PlannedView]:
    obj = DetectedObject(0, np.array([300.0, -400.0, 18.0]), np.array([30.0, 30.0, 60.0]), 0.0, 100)
    candidates = order_for_travel(generate_candidates(obj, Config().views))
    return [PlannedView(c, 0.0, np.eye(4), np.zeros(6)) for c in candidates]


def test_a_few_views_cover_every_side_not_just_the_top_ring():
    views = planned()
    chosen = spread_views(views, 5)
    assert len(chosen) == 5
    assert chosen[0].candidate.elevation_deg == 90.0  # the top view first
    azimuths = [v.candidate.azimuth_deg for v in chosen if v.candidate.elevation_deg < 90]
    gaps = np.diff(sorted(azimuths + [min(azimuths) + 360]))
    assert gaps.max() <= 135  # four side views spread around the object
    assert min(v.candidate.elevation_deg for v in chosen) < 50  # not only the high ring


def test_subset_keeps_travel_order_and_no_count_keeps_everything():
    views = planned()
    position = {id(v): k for k, v in enumerate(views)}
    order = [position[id(v)] for v in spread_views(views, 16)]
    assert len(order) == 16 and order == sorted(order)
    assert [id(v) for v in spread_views(views, None)] == [id(v) for v in views]
    assert len(spread_views(views, 99)) == len(views)
