''' Render meshes to PNG images offscreen (numpy z-buffer, no window or GPU).

Used for README figures and quick checks. Open the real files in Blender to
inspect them interactively.

    python tools/render_mesh.py out.png data/<session>/object_01/recon/mesh.ply [more meshes ...]
        [--elevation 25] [--azimuth -60] [--size 480] [--labels "1 view,25 views"]
'''

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon.render import contact_sheet, render, render_points, save_png  # noqa: E402,F401  (also imported by other tools)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output")
    parser.add_argument("meshes", nargs="+")
    parser.add_argument("--elevation", type=float, default=25.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--colors", action="store_true", help="use the mesh's vertex colors")
    args = parser.parse_args()
    meshes = [o3d.io.read_triangle_mesh(path) for path in args.meshes]
    # Frame every mesh the same way, around the last one (usually the most complete).
    reference = np.asarray(meshes[-1].vertices)
    center = reference.mean(axis=0)
    radius = np.linalg.norm(reference - center, axis=1).max()
    images = [render(m, args.size, args.elevation, args.azimuth, center, radius, args.colors) for m in meshes]
    save_png(contact_sheet(images), Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
