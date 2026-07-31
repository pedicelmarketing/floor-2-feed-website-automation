"""
A real walkthrough of the whole La Meridiana unit, rather than the single straight dolly
used to validate the pipeline.

Route: enters the main room at its south end, walks north up the long axis, and turns west
along the way so both doorways in the west wall come into view -- the one at y=21.65 that
leads toward the hall/laundry, then the bathroom door at y=25.49.

Two things this exercises that the validation clip never did:
  - the camera ROTATES while travelling, so the generator has to hold a room together
    through a turn, not just a push-in
  - it is 97 frames rather than 49, which probes how long a shot can run before the model
    loses coherence -- an open question in the design spec, not something previously measured

Camera stays at 1.6 m eye height, well inside the room's 2.6 m ceiling, and every waypoint
sits comfortably clear of the walls (main room spans x 218.24..223.37, y 16.50..26.44).
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
from camera_paths import waypoint_path, describe  # noqa: E402

HERE = os.path.dirname(__file__)
FRAME_COUNT = 97          # 4n+1, as video models prefer; ~6 s at 16 fps
FPS = 16

# (camera position, look-at target). Heights: eye at 1.6 m, gaze slightly below eye level.
#
# The camera track deliberately hugs the EAST side of the room (x ~221-222) while the
# doorways being shown are on the WEST wall (x ~218.4). An earlier version walked closer to
# the west wall and the last ~20 frames filled with a flat near surface -- at a 70 degree
# field of view, standing 1.5 m from a wall leaves no room to read a doorway as a doorway.
# Keeping 2.5-3.5 m of standoff keeps both the opening and its surrounding wall in frame,
# which is also what the QA aperture check needs in order to measure anything.
WAYPOINTS = [
    ([222.0, 17.8, 1.6], [221.0, 22.5, 1.50]),   # south end, looking up the long axis
    ([221.6, 20.5, 1.6], [219.8, 23.5, 1.50]),   # moving north, beginning to turn west
    ([221.2, 23.0, 1.6], [218.6, 24.8, 1.50]),   # mid-room, facing north-west
    ([220.8, 24.5, 1.6], [218.4, 25.5, 1.45]),   # bathroom doorway in view, 2.4 m off the wall
]


def main():
    with open(os.path.join(HERE, "la_meridiana_unit.json")) as f:
        rooms = json.load(f)

    meshes = []
    for room in rooms:
        builder = Room3DBuilder(room)
        if builder.build_mesh():
            meshes.append(builder.mesh)
    mesh = trimesh.util.concatenate(meshes)
    print(f"Unit mesh: {len(mesh.faces)} faces")

    builder = Room3DBuilder(rooms[0])
    builder.mesh = mesh

    path = waypoint_path(WAYPOINTS, FRAME_COUNT)
    print("Camera path:", describe(path))

    frames_dir = os.path.join(HERE, "output", "walkthrough_frames")
    builder.render_camera_path(path, frames_dir, width=480, height=832, fov_deg=70)

    voids = []
    for i in range(FRAME_COUNT):
        d = np.load(os.path.join(frames_dir, f"depth_{i:04d}.npy"))
        voids.append(float((~np.isfinite(d)).mean()))
    verifiable = sum(1 for v in voids if v <= 0.40)
    print(f"void per frame: mean {np.mean(voids):.3f}  max {max(voids):.3f}")
    print(f"frames verifiable (void <= 40%): {verifiable}/{FRAME_COUNT}")

    videos_dir = os.path.join(HERE, "output", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    for kind in ["depth", "edges"]:
        out = os.path.join(videos_dir, f"{kind}_walkthrough.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(frames_dir, f"{kind}_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out,
        ], check=True)
        print("Wrote", out)


if __name__ == "__main__":
    main()
