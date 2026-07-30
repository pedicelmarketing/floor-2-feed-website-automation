"""
Same camera path as render_camera_path.py, but cast against the WHOLE apartment mesh
(all four rooms) instead of the single "Bedroom / Living" room.

Why: QA on the single-room render reported 8 of 49 frames unverifiable. Nothing was modelled
past the doorway, so once the camera reached the threshold most of the frame was void and
there was no ground truth to score the generation against. Building the mesh from every room
we extracted turns that space into real geometry, which should make those frames checkable.

The camera path is deliberately IDENTICAL to the single-room version so the two runs differ
in exactly one variable -- mesh completeness -- and the QA numbers are directly comparable.

Caveat found while building this: of the four extracted rooms only the big room, the hall and
the bathroom carry door openings; the laundry's walls came out solid (no A-PUERTAS insert was
assigned to it). So an opening in one room's wall does not necessarily line up with an
opening in its neighbour's. That is a real property of the extraction, not of this script,
and it is exactly what the void statistics printed below are meant to expose.
"""
import os
import sys
import json
import subprocess
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from importlib import import_module
m3d = import_module("3d_room_builder")
Room3DBuilder = m3d.Room3DBuilder

HERE = os.path.dirname(__file__)
FRAME_COUNT = 49
FPS = 16

# Identical to render_camera_path.py -- do not "improve" these without re-running the
# single-room baseline too, or the comparison stops being an A/B.
START_POS = np.array([222.5, 18.5, 1.6])
END_POS = np.array([219.6, 21.3, 1.6])
TARGET = np.array([218.36, 21.71, 1.4])


def build_unit_mesh(rooms):
    meshes = []
    for room in rooms:
        builder = Room3DBuilder(room)
        if builder.build_mesh():
            meshes.append(builder.mesh)
            print(f"  + {room['room_name']}: {len(builder.mesh.faces)} faces")
        else:
            print(f"  ! skipped {room['room_name']} (build_mesh failed)")
    return trimesh.util.concatenate(meshes)


def main():
    with open(os.path.join(HERE, "la_meridiana_unit.json")) as f:
        rooms = json.load(f)

    print(f"Building combined mesh from {len(rooms)} rooms:")
    mesh = build_unit_mesh(rooms)
    print(f"Combined: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Bounds: {mesh.bounds.tolist()}")

    builder = Room3DBuilder(rooms[0])
    builder.mesh = mesh

    camera_path = []
    for i in range(FRAME_COUNT):
        t = i / (FRAME_COUNT - 1)
        camera_path.append((START_POS + (END_POS - START_POS) * t, TARGET))

    frames_dir = os.path.join(HERE, "output", "unit_path_frames")
    builder.render_camera_path(camera_path, frames_dir, width=480, height=832, fov_deg=70)

    # The number that matters: how much of each frame is now real geometry rather than void.
    fractions = []
    for i in range(FRAME_COUNT):
        depth = np.load(os.path.join(frames_dir, f"depth_{i:04d}.npy"))
        fractions.append(float((~np.isfinite(depth)).mean()))
    verifiable = sum(1 for f in fractions if 0 < f <= 0.40)
    all_solid = sum(1 for f in fractions if f == 0.0)

    print("\n--- void fraction per frame (full unit mesh) ---")
    print(f"  first: {fractions[0]:.3f}   mid: {fractions[len(fractions)//2]:.3f}   "
          f"last: {fractions[-1]:.3f}")
    print(f"  max:   {max(fractions):.3f}  mean: {float(np.mean(fractions)):.3f}")
    print(f"  frames with a bounded opening (0 < void <= 0.40): {verifiable}/{FRAME_COUNT}")
    print(f"  frames fully enclosed (void == 0, no opening to score): {all_solid}/{FRAME_COUNT}")

    videos_dir = os.path.join(HERE, "output", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    for kind in ["depth", "edges"]:
        out_path = os.path.join(videos_dir, f"{kind}_unit.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(frames_dir, f"{kind}_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path,
        ], check=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
