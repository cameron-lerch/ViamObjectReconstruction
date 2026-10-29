''' Write depth_preview.jpg (colored, for looking at) beside every depth.png in a session.

    python tools/depth_previews.py data/20261029-131913

New captures get one automatically; this is for sessions saved before that.
'''

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon.rgbd import depth_preview  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    count = 0
    for path in sorted(Path(sys.argv[1]).rglob("depth.png")):
        depth = np.asarray(o3d.io.read_image(str(path))).astype(np.uint16)
        o3d.io.write_image(str(path.with_name("depth_preview.jpg")), o3d.geometry.Image(depth_preview(depth)), quality=85)
        count += 1
    print(f"wrote {count} depth previews")


if __name__ == "__main__":
    main()
