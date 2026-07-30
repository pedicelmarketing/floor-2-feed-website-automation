"""
A tour that actually WALKS BETWEEN ROOMS: hall -> through the doorway -> main room ->
up to the bathroom door.

This needs door_pairing first. Rooms are extracted independently, so a door recorded on one
room cuts an opening in that room only, and the neighbouring room's wall -- typically 0.10
to 0.25 m behind it -- stays solid. Facing such a doorway looks correct; travelling through
it does not, because the camera hits the far wall a few centimetres later.

Reachability in this unit, from the source data rather than assumption:
    Hall  <->  Main room  <->  Bathroom
    Laundry: unreachable. No door was extracted for it and no other room's door lands on its
    wall, so there is no way in. That is a fact about the CAD extraction, not something this
    script should invent a hole to work around.

The passable slot between hall and main room is narrow, and the route has to be built
around it rather than aimed at it approximately:

    bedroom-side opening   x = 218.472,  y 21.248 .. 22.048
    hall-side opening      x = 218.372,  y 20.747 .. 21.547
    PASSABLE OVERLAP                     y 21.248 .. 21.547   (0.299 m)

The camera must be inside that 0.299 m band *at the moment it crosses x = 218.4*, which
means squaring up to the opening while still in the hall. An earlier version aimed the
straight leg at a waypoint of y = 21.4 but placed that waypoint at x = 218.9 -- past the
wall. Interpolating to it crossed the wall plane at y = 21.0, a quarter metre south of the
opening, so the camera travelled through solid wall for five frames. Every ray hit a
surface 1-2 cm away, the depth map went flat, and the generated video cut hard.
"""
import os
import sys
import json
import subprocess

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from importlib import import_module
Room3DBuilder = import_module("3d_room_builder").Room3DBuilder
from camera_paths import waypoint_path, describe          # noqa: E402
from door_pairing import pair_doors, connectivity          # noqa: E402

HERE = os.path.dirname(__file__)
FRAME_COUNT = 121         # 4n+1; ~7.5 s at 16 fps
FPS = 16

WAYPOINTS = [
    ([217.30, 18.20, 1.6], [217.40, 21.00, 1.50]),   # hall, south end, looking north
    ([217.90, 20.60, 1.6], [218.90, 21.40, 1.50]),   # nearing the doorway, turning east
    ([218.30, 21.40, 1.6], [219.60, 21.60, 1.50]),   # squared up to the slot, still in the hall
    ([218.90, 21.40, 1.6], [220.60, 22.00, 1.50]),   # THROUGH it -- crosses x=218.4 at y=21.40
    ([220.50, 23.00, 1.6], [219.50, 25.20, 1.50]),   # main room, heading north
    ([219.90, 24.70, 1.6], [218.40, 25.49, 1.45]),   # bathroom doorway in view
]

# A frame whose visible depth barely varies means the camera is jammed against a surface --
# every ray landing 1-2 cm away. It is NOT caught by the void check below: a camera buried
# in a wall sees geometry in every direction, so its void fraction is a healthy 0%. That
# blind spot is exactly how the wall-clipping above passed an automated check.
MIN_DEPTH_SPREAD_M = 0.35


def main():
    with open(os.path.join(HERE, "la_meridiana_unit.json")) as f:
        rooms = json.load(f)

    paired, log = pair_doors(rooms)
    print("Door pairing:")
    for line in log:
        print(line)
    print("\nReachable rooms:")
    for room, neighbours in connectivity(paired).items():
        print(f"  {room:26s} -> {neighbours if neighbours else 'UNREACHABLE'}")

    meshes = []
    for room in paired:
        b = Room3DBuilder(room)
        if b.build_mesh():
            meshes.append(b.mesh)
    mesh = trimesh.util.concatenate(meshes)
    print(f"\nMesh: {len(mesh.faces)} faces")

    builder = Room3DBuilder(paired[0])
    builder.mesh = mesh

    path = waypoint_path(WAYPOINTS, FRAME_COUNT)
    print("Camera:", describe(path))

    frames_dir = os.path.join(HERE, "output", "tour_frames")
    builder.render_camera_path(path, frames_dir, width=480, height=832, fov_deg=70)

    grids = [np.load(os.path.join(frames_dir, f"depth_{i:04d}.npy")) for i in range(FRAME_COUNT)]

    voids = [float((~np.isfinite(g)).mean()) for g in grids]
    # Almost all void means the camera left the modelled space entirely.
    escaped = [i for i, v in enumerate(voids) if v > 0.60]
    print(f"void per frame: mean {np.mean(voids):.3f}  max {max(voids):.3f}")
    print(f"frames where the camera left the model (void > 60%): "
          f"{len(escaped)}{' -> ' + str(escaped[:8]) if escaped else ' (none)'}")

    spreads = [float(g[np.isfinite(g)].std()) if np.isfinite(g).any() else 0.0 for g in grids]
    buried = [i for i, s in enumerate(spreads) if s < MIN_DEPTH_SPREAD_M]
    print(f"depth spread per frame: min {min(spreads):.3f} m  median {np.median(spreads):.3f} m")
    print(f"frames with the camera jammed against a surface (spread < {MIN_DEPTH_SPREAD_M} m): "
          f"{len(buried)}{' -> ' + str(buried[:8]) if buried else ' (none)'}")
    if buried:
        print("  FAIL: the route passes through geometry instead of through an opening.")

    videos_dir = os.path.join(HERE, "output", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    for kind in ["depth", "edges"]:
        out = os.path.join(videos_dir, f"{kind}_tour.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(frames_dir, f"{kind}_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out,
        ], check=True)
        print("Wrote", out)


if __name__ == "__main__":
    main()
