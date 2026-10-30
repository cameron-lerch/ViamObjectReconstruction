''' A numbered map of the table after a survey: a picture of the table from straight above,
put together from the survey views, with each scan target boxed in green and numbered as
the terminal lists it, and everything else on the table (kept as obstacles) in gray.

Saved as survey_map.png in the session, and shown in the live viewer.
'''

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .rgbd import back_project, load_view
from .scene import DetectedObject

CELL = 1.5  # mm per map pixel
TARGET, OBSTACLE, LABEL_BACK = (46, 158, 91), (72, 82, 96), (255, 255, 255)
BELOW_TABLE = 30.0  # points this far below the table (the floor) are dimmed (mm)


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def top_view(folders: list[Path], x_range, y_range, min_depth: float, max_depth: float,
             table_z: float | None = None) -> Image.Image:
    ''' The table seen from straight above: every survey view's points, the highest one per
    1.5mm cell, in its camera color, small gaps filled from their neighbors and the floor
    (well below the table) dimmed. +x points up the image, +y to the left (as seen from
    above, standing at the arm's base looking out over the table). '''
    (x0, x1), (y0, y1) = x_range, y_range
    rows, cols = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    height = np.full((rows, cols), -np.inf)
    image = np.full((rows, cols, 3), 40, np.uint8)
    score = np.full((rows, cols), -np.inf)
    for folder in folders:
        view = load_view(folder)
        points, colors, _ = back_project(view, 1, min_depth, max_depth)
        r = ((x1 - points[:, 0]) / CELL).astype(int)
        c = ((y1 - points[:, 1]) / CELL).astype(int)
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        r, c, points, colors = r[inside], c[inside], points[inside], colors[inside]
        # Higher wins, in 30mm steps (an object over the table); within a step, the view
        # looking most directly down on the cell (each view has its own exposure, so taking
        # the highest reading on flat table would mottle it).
        off_axis = np.linalg.norm(points[:, :2] - view.camera_position[:2], axis=1)
        # Steps measured from the table, which (its depth wanders about +-10mm across the
        # surveyed area) sits inside one; scan targets stand at least 25mm tall.
        above = points[:, 2] - (table_z if table_z is not None else 0.0)
        points_score = np.floor((above + 15.0) / 30.0) * 1e4 - off_axis
        order = np.argsort(points_score)  # later (better) points overwrite earlier ones in a cell
        r, c, points, colors, points_score = r[order], c[order], points[order], colors[order], points_score[order]
        better = points_score > score[r, c]
        score[r[better], c[better]] = points_score[better]
        height[r[better], c[better]] = points[better, 2]
        image[r[better], c[better]] = (colors[better] * 255).astype(np.uint8)
    seen = np.isfinite(height)
    if table_z is not None:
        floor = seen & (height < table_z - BELOW_TABLE)
        image[floor] = (image[floor] * 0.35).astype(np.uint8)
    # Fill cells no point landed in from the nearest seen cell, if one is within 3 cells.
    distance, (near_r, near_c) = ndimage.distance_transform_edt(~seen, return_indices=True)
    fill = ~seen & (distance <= 3)
    image[fill] = image[near_r[fill], near_c[fill]]
    return Image.fromarray(image)


def draw_map(image: Image.Image, objects: list[DetectedObject], clutter: list[DetectedObject],
             x_range, y_range) -> Image.Image:
    ''' Boxes and numbers over the top view: targets green, obstacles gray. '''
    (_, x1), (_, y1) = x_range, y_range
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(max(14, image.width // 45))

    def to_pixel(x, y):
        return (y1 - y) / CELL, (x1 - x) / CELL

    for kind, things, color in (("obstacle", clutter, OBSTACLE), ("target", objects, TARGET)):
        for thing in things:
            corners = [thing.pose @ np.array([sx * thing.size[0] / 2, sy * thing.size[1] / 2, 0.0, 1.0])
                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            outline = [to_pixel(p[0], p[1]) for p in corners]
            draw.polygon(outline, outline=color, width=3 if kind == "target" else 2)
            if kind == "target":
                u, v = to_pixel(*thing.center[:2])
                text = str(thing.id)
                box = draw.textbbox((0, 0), text, font=font)
                w, h = box[2] - box[0], box[3] - box[1]
                top = min(p[1] for p in outline) - h - 10
                draw.rounded_rectangle((u - w / 2 - 6, top - 3, u + w / 2 + 6, top + h + 7), 6, fill=TARGET)
                draw.text((u - w / 2, top - box[1]), text, font=font, fill=LABEL_BACK)
    # Which way is which, in the corner.
    small = _font(max(12, image.width // 70))
    draw.rectangle((8, 8, 190, 84), fill=(255, 255, 255))
    draw.text((16, 12), "+x up, +y left", font=small, fill=(31, 41, 51))
    draw.text((16, 34), "green: scan targets", font=small, fill=TARGET)
    draw.text((16, 56), "gray: obstacles", font=small, fill=OBSTACLE)
    return image


def survey_map(session, objects: list[DetectedObject], clutter: list[DetectedObject], config,
               table_z: float | None = None) -> Path:
    ''' Make and save the session's survey_map.png (`table_z`: the table's height, to dim the floor). '''
    folders = sorted(path.parent for path in session.survey_dir.glob("view_*/view.json"))
    workspace = config.workspace
    image = top_view(folders, workspace.x_range, workspace.y_range, config.camera.min_depth, config.camera.max_depth,
                     table_z)
    image = draw_map(image, objects, clutter, workspace.x_range, workspace.y_range)
    path = session.root / "survey_map.png"
    image.save(path)
    return path
