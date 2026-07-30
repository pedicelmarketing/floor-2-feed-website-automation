"""
First camera-path test (spec section 12's "smallest first test"): a short dolly toward the
real doorway in the real La Meridiana "Bedroom / Living" room, rendered as depth + edge
control videos for a Wan VACE video-restyle pass.

Camera moves from deep in the room toward the doorway on wall edge 7 (where two door
inserts sit close together in the real DWG), gaze fixed on that opening throughout -- a
"walking toward a doorway" shot, matching the spec's own suggested first test framing.
Beyond the doorway there's no built geometry (this room's mesh only), so rays passing
through it correctly return a miss/void, same as the still-image test.
"""
import os
import sys
import json
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from importlib import import_module
m3d = import_module("3d_room_builder")
Room3DBuilder = m3d.Room3DBuilder

HERE = os.path.dirname(__file__)
FRAME_COUNT = 49
FPS = 16  # ~3 seconds at 49 frames, matching the spec's suggested first-test length

with open(os.path.join(HERE, "la_meridiana_unit.json")) as f:
    rooms = json.load(f)

room = rooms[0]  # "Bedroom / Living" -- the room with a real doorway, already validated as a still
builder = Room3DBuilder(room)
builder.build_mesh()

start_pos = np.array([222.5, 18.5, 1.6])
end_pos = np.array([219.6, 21.3, 1.6])
target = np.array([218.36, 21.71, 1.4])  # fixed gaze on the doorway (edge 7) throughout

camera_path = []
for i in range(FRAME_COUNT):
    t = i / (FRAME_COUNT - 1)
    pos = start_pos + (end_pos - start_pos) * t
    camera_path.append((pos, target))

frames_dir = os.path.join(HERE, "output", "camera_path_frames")
builder.render_camera_path(camera_path, frames_dir, width=480, height=832, fov_deg=70)

videos_dir = os.path.join(HERE, "output", "videos")
os.makedirs(videos_dir, exist_ok=True)

for kind in ["depth", "edges"]:
    out_path = os.path.join(videos_dir, f"{kind}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, f"{kind}_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_path,
    ], check=True)
    print(f"Wrote {out_path}")
