''' The pictures for people: the height view and the numbered survey map. '''

import numpy as np
from PIL import Image

from recon.rgbd import Intrinsics, View, height_preview
from recon.scene import DetectedObject, TablePlane
from recon.survey_map import TARGET, draw_map

LENS = Intrinsics(width=120, height=80, fx=100.0, fy=100.0, cx=60.0, cy=40.0)
TABLE = TablePlane(np.array([0.0, 0.0, 1.0]), 0.0)


def looking_down(depth):
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])  # straight down from 400mm
    pose[2, 3] = 400.0
    return View(np.zeros((80, 120, 3), np.uint8), depth, pose, LENS, {}, "test")


def test_the_table_is_one_gray_and_a_block_on_it_is_colored():
    depth = np.full((80, 120), 400, np.uint16)
    depth[30:50, 50:70] = 340  # a block 60mm tall
    depth[0:5, 0:5] = 0
    image = height_preview(looking_down(depth), TABLE)
    assert (image[70, 10] == (215, 220, 226)).all()
    assert not (image[40, 60] == (215, 220, 226)).all()
    assert (image[2, 2] == 0).all()


def test_the_map_boxes_and_numbers_the_targets():
    blank = Image.new("RGB", (400, 400), (200, 200, 200))
    target = DetectedObject(3, np.array([200.0, 0.0, 30.0]), np.array([60.0, 60.0, 60.0]), 0.0, 100)
    image = np.asarray(draw_map(blank, [target], [], (0.0, 600.0), (-300.0, 300.0)))
    assert (np.abs(image.astype(int) - TARGET).sum(axis=-1) < 30).sum() > 200  # its green box and label
