# 3D-Printed Reference Objects

Objects printed from CAD, so their true shape is known everywhere, not just at a few caliper measurements. A scan of one can be compared with its CAD model surface-to-surface. That shows *where* the reconstruction is wrong (edges, curved walls, cavities, the base), not only by how much it is too wide.

## One folder per object

Copy `_template/` to a new folder named after the object (lowercase, underscores: `stepped_cylinder/`), then add:

| File | Needed? | Why |
|---|---|---|
| `<name>.step` | **Yes** | The exact CAD geometry: true planes, cylinders and edges, not triangles. The reference for measuring accuracy. |
| `<name>.stl` | **Yes** | The exact file that was printed. The comparison code reads this mesh (it must be in millimeters). |
| `info.toml` | **Yes** | How the object was printed and how it sits on the table (see the template). |
| `photo.jpg` | Nice to have | The printed part on the table, the way it will be scanned. |
| Native CAD file (`.f3d`, `.sldprt`, `.FCStd`, ...) | Optional | Only if we might change the design later. |

Export the STL in **millimeters**, with the part in the same position and orientation as in the STEP file.

## How it sits on the table matters

The scan is compared with the CAD model after the best rigid fit, so the object's placement doesn't need to be exact. But the comparison needs to know which face rests on the table: the bottom is never seen, and is filled in. Say so in `info.toml`, and model the part so that face is at z = 0, facing down, if you can.

## Printing for a depth camera

The RealSense projects an infrared dot pattern and matches it between two infrared cameras, so the surface finish matters as much as the shape:

- **Matte, light, opaque filament** (white or light gray PLA is ideal). Avoid glossy, silk, metallic, translucent or very dark filament: shiny surfaces reflect the pattern away, translucent ones let it in, and dark ones absorb it.
- **Measure the printed part with calipers** and note it in `info.toml`. FDM prints are typically off by 0.1-0.3mm, and the calipers tell us how far the print is from the CAD.
- **Size:** 20-150mm, like the objects we scan. Small features under about 3mm are below what the scans resolve (1.5mm voxels).
- **Useful shapes** (each tests something specific):
  - steps of known height: depth accuracy and edges;
  - cylinders of several diameters: curved walls, where the width error shows;
  - sharp edges and corners: how much smoothing rounds them;
  - a cavity or cup: concave surfaces;
  - an overhang: walls the camera can't see straight-on;
  - a flat-bottomed object with a domed or angled top: surfaces at different angles to the camera.
