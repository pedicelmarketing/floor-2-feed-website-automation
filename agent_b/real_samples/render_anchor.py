"""
Render THE anchor scene (agent_b/anchor_scene.py) -- the one shot every experiment shares.

Deliberately thin. All the parameters live in anchor_scene.py so that a change to the shot is a
change to one frozen file rather than something that drifts quietly across a dozen scripts,
which is exactly how this project ended up with a pile of numbers that were not comparable to
each other.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import importlib                                                     # noqa: E402
import anchor_scene as A                                             # noqa: E402
_b = importlib.import_module("3d_room_builder")                      # noqa: E402
from camera_paths import describe                                    # noqa: E402
from pdf_blockout import blockout_from_page, wall_polygons           # noqa: E402
from pdf_vector import extract                                       # noqa: E402
from route_planner import plan_route, to_waypoints                   # noqa: E402
from wall_regions import rooms_for_page                              # noqa: E402

OUT = os.path.join(HERE, "output", f"anchor_v{A.ANCHOR_VERSION}")
CLAY_EXPOSURE = 1.6
SUN_ELEVATION = 0.69
NEAR_CLIP_M = 0.30


def main() -> int:
    print(A.summary())
    page = extract(A.PDF, pages=[A.PAGE])["pages"][0]
    rooms = {r["name"]: np.array(r["pos"]) * (A.MM_PER_PT / 1000.0)
             for r in rooms_for_page(page)["rooms"]}
    missing = [n for n in A.ROUTE if n not in rooms]
    if missing:
        print(f"anchor route rooms missing from this page: {missing}")
        return 1

    blockout = blockout_from_page(page, A.MM_PER_PT, region_m=A.REGION_M)
    mesh, materials = blockout["mesh"], blockout.get("face_materials")
    if mesh is None:
        print(f"no blockout: {blockout.get('reason')}")
        return 1

    polygons, _ = wall_polygons(page, A.MM_PER_PT, A.REGION_M)
    obstacles = polygons + list(blockout.get("furniture_footprints", []))
    route = plan_route(obstacles, A.REGION_M, [rooms[n] for n in A.ROUTE])
    if not route["ok"]:
        print("FAIL: no route keeps the camera clear of geometry")
        return 1

    from shapely.ops import unary_union
    attractors = [{"point": (f.centroid.x, f.centroid.y), "weight": f.area}
                  for f in blockout.get("furniture_footprints", [])]
    waypoints = to_waypoints(route, count=A.FRAME_COUNT, eye_height_m=A.EYE_HEIGHT_M,
                             attractors=attractors,
                             blocker=unary_union(polygons) if polygons else None)
    path = [(np.asarray(p, dtype=float), np.asarray(t, dtype=float)) for p, t in waypoints]
    print("camera:", describe(path))

    P = np.array([p[0] for p in path])
    travel = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    seconds = A.FRAME_COUNT / A.FPS
    print(f"{travel:.1f} m over {seconds:.1f} s = {travel / seconds:.2f} m/s")

    sun_dir = None
    if materials is not None and (materials == "window").any():
        glazing = mesh.triangles_center[materials == "window"].mean(axis=0)
        inward = mesh.centroid[:2] - glazing[:2]
        inward = inward / (np.linalg.norm(inward) + 1e-9)
        sun_dir = np.array([inward[0] * 0.72, inward[1] * 0.72, -SUN_ELEVATION])
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
        print(f"sun through the glazing, direction {np.round(sun_dir, 2).tolist()}")

    builder = _b.Room3DBuilder({"room_name": "anchor", "polygon": [[0, 0], [1, 0], [1, 1]],
                                "ceiling_height_m": blockout["ceiling_height_m"],
                                "doors": [], "windows": []})
    builder.mesh = mesh
    frames = os.path.join(OUT, "frames")
    builder.render_camera_path(path, frames, width=A.WIDTH, height=A.HEIGHT, fov_deg=A.FOV_DEG,
                               face_materials=materials,
                               tint_map=_b.MATERIAL_TINT_LEGIBLE, semantic=True,
                               exposure=CLAY_EXPOSURE, sun_dir=sun_dir)

    grids = [np.load(os.path.join(frames, f"depth_{i:04d}.npy")) for i in range(A.FRAME_COUNT)]
    nearest = [float(g[np.isfinite(g)].min()) if np.isfinite(g).any() else 0.0 for g in grids]
    buried = [i for i, n in enumerate(nearest) if n < NEAR_CLIP_M]
    print(f"nearest surface {min(nearest):.2f} m | frames inside geometry: {len(buried)}")
    if buried:
        print("  FAIL: the route passes through geometry")

    videos = os.path.join(OUT, "videos")
    os.makedirs(videos, exist_ok=True)
    for kind in ("material", "clay", "semantic", "depth", "edges"):
        if not os.path.exists(os.path.join(frames, f"{kind}_0000.png")):
            continue
        target = os.path.join(videos, f"{kind}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(A.FPS),
                        "-i", os.path.join(frames, f"{kind}_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", target],
                       check=True)
        print("wrote", target)
    return 0 if not buried else 1


if __name__ == "__main__":
    sys.exit(main())
