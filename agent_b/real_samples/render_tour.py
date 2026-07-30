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

The passable slot between hall and main room is narrow. The two mirrored openings overlap
over roughly 0.3 m of wall, so the route threads y = 21.4 to pass cleanly through both.
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
    ([217.30, 18.00, 1.6], [217.40, 21.00, 1.50]),   # hall, south end, looking north
    ([217.60, 20.40, 1.6], [218.60, 21.30, 1.50]),   # nearing the doorway, turning east
    ([218.90, 21.40, 1.6], [220.50, 22.00, 1.50]),   # THROUGH the doorway into the main room
    ([220.50, 23.00, 1.6], [219.50, 25.20, 1.50]),   # main room, heading north
    ([219.90, 24.70, 1.6], [218.40, 25.49, 1.45]),   # bathroom doorway in view
]


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

    voids = [float((~np.isfinite(np.load(os.path.join(frames_dir, f"depth_{i:04d}.npy")))).mean())
             for i in range(FRAME_COUNT)]
    # A frame that is almost entirely void means the camera left the modelled space -- most
    # likely it clipped a wall while threading the doorway rather than passing through it.
    escaped = [i for i, v in enumerate(voids) if v > 0.60]
    print(f"void per frame: mean {np.mean(voids):.3f}  max {max(voids):.3f}")
    print(f"frames where the camera left the model (void > 60%): "
          f"{len(escaped)}{' -> ' + str(escaped[:8]) if escaped else ' (none)'}")

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
