''' A survey object must look like it's standing in at least half the views that saw its spot. '''

import numpy as np

from recon.scene import DetectedObject, views_agree


def flat_table(center, radius=60.0, spacing=2.0):
    u = np.arange(-radius, radius, spacing)
    xy = np.array([(x, y) for x in u for y in u]) + center
    return np.column_stack([xy, np.zeros(len(xy))])


def test_a_glitch_in_one_view_is_outvoted_and_a_bowl_counts_by_its_rim():
    center = np.array([400.0, -200.0])
    obj = DetectedObject(0, np.array([*center, 15.0]), np.array([28.0, 14.0, 30.0]), 0.0, 100)
    table = flat_table(center)
    glitch = table.copy()
    near = np.linalg.norm(glitch[:, :2] - center, axis=1) < 6
    glitch[near, 2] = 30.0
    clouds = [(glitch, glitch[:, 2])] + [(table, table[:, 2])] * 3
    assert views_agree(obj, clouds) == (4, 1)

    # A bowl: the floor in the middle is low, the rim near the outline is tall.
    bowl = DetectedObject(1, np.array([*center, 32.0]), np.array([130.0, 130.0, 65.0]), 0.0, 100)
    points = flat_table(center, 70.0)
    r = np.linalg.norm(points[:, :2] - center, axis=1)
    points[:, 2] = np.where(r < 50, 8.0, np.where(r < 65, 65.0, 0.0))
    assert views_agree(bowl, [(points, points[:, 2])] * 2) == (2, 2)
