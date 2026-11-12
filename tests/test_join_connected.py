''' Joining pieces of one object in the survey (a cable's plug and its flat coil), without
merging separate objects. Guards the 2026-11-11 cable survey, where the plug became a
small target and the coil an obstacle, and a coiled cable's mostly empty outline. '''

import numpy as np

from recon.config import WorkspaceConfig
from recon.scene import DetectedObject, join_connected, views_agree

TABLE = 200  # gray table color


def scene(links: list[tuple[int, int]]):
    ''' A 1mm-per-pixel image and points of three dark pieces on a gray table, with dark
    "cable" lines drawn between the pieces in `links`. Piece 0 is a tall target, piece 1
    a flat skipped piece, piece 2 another tall target. '''
    color = np.full((200, 400, 3), TABLE, dtype=np.uint8)
    boxes = [(20, 60, 90, 110), (100, 180, 60, 140), (300, 340, 90, 110)]  # col0, col1, row0, row1
    heights = [40.0, 15.0, 40.0]
    rows, cols = np.mgrid[0:200:2, 0:400:2]
    z = np.zeros(rows.shape)
    for (c0, c1, r0, r1), h in zip(boxes, heights):
        color[r0:r1, c0:c1] = 20
        z[(cols >= c0) & (cols < c1) & (rows >= r0) & (rows < r1)] = h
    for a, b in links:
        ca, cb = (boxes[a][0] + boxes[a][1]) // 2, (boxes[b][0] + boxes[b][1]) // 2
        color[98:102, min(ca, cb):max(ca, cb)] = 20  # a cable lying flat: dark, but no height
    points = np.column_stack([cols.ravel(), rows.ravel(), z.ravel()]).astype(float)
    pixels = np.column_stack([rows.ravel(), cols.ravel()])
    objects = [DetectedObject(i, np.array([(c0 + c1) / 2, (r0 + r1) / 2, h / 2]), np.array([c1 - c0, r1 - r0, h]), 0.0, 100)
               for i, ((c0, c1, r0, r1), h) in enumerate(zip(boxes, heights))]
    return color, points, pixels, objects


def run(links, targets=(0,)):
    color, points, pixels, objects = scene(links)
    found = [objects[i] for i in targets]
    skipped = [o for i, o in enumerate(objects) if i not in targets]
    return join_connected(found, skipped, points, pixels, points[:, 2], color, WorkspaceConfig())


def test_a_piece_connected_by_a_cable_joins_its_target():
    joined, left = run([(0, 1)], targets=(0,))
    assert len(left) == 1 and joined[0].size[0] > 150  # target 0 now spans pieces 0 and 1
    assert joined[0].size[2] == 40.0  # the target's own height is kept


def test_pieces_with_table_between_them_stay_apart():
    joined, left = run([], targets=(0,))
    assert len(left) == 2 and joined[0].size[0] == 40


def test_a_piece_touching_two_targets_stays_an_obstacle():
    joined, left = run([(0, 1), (1, 2)], targets=(0, 2))
    assert len(left) == 1 and [o.size[0] for o in joined] == [40, 40]


def test_a_coiled_cable_counts_as_standing():
    # A ring of cable 24mm tall around bare table: most of the outline is table.
    center = np.array([300.0, 250.0])
    u = np.arange(-100, 100, 2.0)
    xy = np.array([(x, y) for x in u for y in u]) + center
    r = np.linalg.norm(xy - center, axis=1)
    z = np.where(np.abs(r - 80) < 4, 24.0, 0.0)
    points = np.column_stack([xy, z])
    coil = DetectedObject(0, np.array([*center, 12.0]), np.array([196.0, 196.0, 24.0]), 0.0, 100)
    assert views_agree(coil, [(points, z)] * 3) == (3, 3)


def test_flat_loops_grow_the_box():
    # A plug (40mm tall) with cable lying flat beside it (5mm tall: under the survey's
    # 8mm minimum, so in no piece): the box grows over the cable, which is dark like the plug.
    color = np.full((200, 400, 3), TABLE, dtype=np.uint8)
    rows, cols = np.mgrid[0:200:2, 0:400:2]
    z = np.zeros(rows.shape)
    color[90:110, 20:60] = 20
    z[(cols >= 20) & (cols < 60) & (rows >= 90) & (rows < 110)] = 40.0
    color[96:104, 60:200] = 20
    z[(cols >= 60) & (cols < 200) & (rows >= 96) & (rows < 104)] = 5.0
    points = np.column_stack([cols.ravel(), rows.ravel(), z.ravel()]).astype(float)
    pixels = np.column_stack([rows.ravel(), cols.ravel()])
    plug = DetectedObject(0, np.array([40.0, 100.0, 20.0]), np.array([40.0, 20.0, 40.0]), 0.0, 100)
    joined, _ = join_connected([plug], [], points, pixels, points[:, 2], color, WorkspaceConfig())
    assert joined[0].size[0] > 150 and joined[0].size[2] == 40.0
