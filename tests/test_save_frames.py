''' --full-scan keeps every raw frame beside the view's median depth. '''

import numpy as np
import open3d as o3d

from recon.rgbd import save_frames


def test_every_frame_is_saved_losslessly(tmp_path):
    depths = [np.full((4, 5), 400 + k, np.uint16) for k in range(3)]
    save_frames(tmp_path / "frames", [b"jpeg-bytes-%d" % k for k in range(3)], depths)
    names = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert names == ["color_000.jpg", "color_001.jpg", "color_002.jpg", "depth_000.png", "depth_001.png", "depth_002.png"]
    back = np.asarray(o3d.io.read_image(str(tmp_path / "frames" / "depth_002.png")))
    assert back.dtype == np.uint16 and (back == 402).all()
