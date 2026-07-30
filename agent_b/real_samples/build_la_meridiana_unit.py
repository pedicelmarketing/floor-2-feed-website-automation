import os
import sys
import json
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from importlib import import_module
m3d = import_module("3d_room_builder")
Room3DBuilder = m3d.Room3DBuilder

HERE = os.path.dirname(__file__)

with open(os.path.join(HERE, "la_meridiana_unit.json")) as f:
    rooms = json.load(f)

meshes = []
for room in rooms:
    builder = Room3DBuilder(room)
    if builder.build_mesh():
        meshes.append(builder.mesh)
    else:
        print(f"Skipped {room['room_name']} -- build_mesh() failed")

combined = trimesh.util.concatenate(meshes)
print(f"Combined apartment mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")
print("Bounds:", combined.bounds)

out_dir = os.path.join(HERE, "output")
os.makedirs(out_dir, exist_ok=True)
glb_path = os.path.join(out_dir, "la_meridiana_unit.glb")
combined.export(glb_path)
print("Exported", glb_path)

# One eye-level render for sanity, same convention as the synthetic-fixture validation
centroid = combined.centroid
builder = Room3DBuilder(rooms[0])
builder.mesh = combined
builder.room_data = {"polygon": [[centroid[0]-3, centroid[1]-3], [centroid[0]+3, centroid[1]-3],
                                  [centroid[0]+3, centroid[1]+3], [centroid[0]-3, centroid[1]+3]],
                      "ceiling_height_m": 2.6}
builder.render_control_maps(out_dir, camera_position=[centroid[0]-1, centroid[1]-1, 1.6],
                             camera_target=[centroid[0]+1, centroid[1]+1, 1.4], width=640, height=480, fov_deg=80)
print("Rendered control maps to", out_dir)
