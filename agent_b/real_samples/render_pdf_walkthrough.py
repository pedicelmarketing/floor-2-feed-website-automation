"""
Render a walkthrough of an apartment taken entirely from the PDF floor plan. No DWG involved.

The route is built from the room labels themselves: each label states there is a room at that
point, so the label positions are known-good interior waypoints. That is the same fact the
room-separation work leans on, reused here for camera placement.

Checks that run on every render, both of which caught real defects on the DWG path:

  void fraction     the camera left the modelled space entirely.
  depth spread      the camera is jammed against a surface. A camera buried inside a wall
                    sees geometry in every direction, so its void fraction is a healthy 0%
                    and the first check passes it. This is the one that catches it.
"""
import os
import subprocess
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module                                  # noqa: E402

Room3DBuilder = import_module("3d_room_builder").Room3DBuilder
from camera_paths import describe, waypoint_path                     # noqa: E402
from pdf_blockout import blockout_from_page, wall_polygons            # noqa: E402
from route_planner import plan_route, to_waypoints                    # noqa: E402
from pdf_vector import extract                                       # noqa: E402
from wall_regions import rooms_for_page                              # noqa: E402

PDF = ("/home/openclaw/floor-2-feed-website-automation/uploads/"
       "12f909d4-7599-4940-aa09-79108a7625d8_floor-plans-estado-reformado.pdf")
PAGE = 0
# Verified against 278 doors of known width; see measure_pdf_accuracy.py.
MM_PER_PT = 36.1
# One apartment out of a floor that holds several: living (GG), bedroom (D4), hall (HL),
# bathroom (A1). Chosen by where it is, not by any identifier internal to the file.
REGION_M = (1.2, 15.0, 9.8, 27.0)
# Hall -> living only. The three-room route is 10.9 m, and at any frame count Wan or Cosmos
# can hold that is 1.4-1.8 m/s -- faster than a relaxed walk, where an architectural
# walkthrough wants 0.5-0.8. Shortening the route buys the speed without a stitching step
# whose seams would be a new variable in the generator comparison.
ROUTE = ["HL", "GG"]
FRAME_COUNT = 97                    # 4n+1
FPS = 16
EYE_HEIGHT_M = 1.60
# Anything nearer than this means the camera is inside or against geometry. This is the
# failure signal; it is a DISTANCE, not a variance.
NEAR_CLIP_M = 0.30
# Advisory only. A frame with little depth variety may be a camera in a wall, or may be an
# honest view of a plain wall a couple of metres off. Calibrated on the DWG tour, where jammed
# frames read 0.002 m spread at 0.01 m range, it fired on 27 frames here whose nearest surface
# was 1.7 m away -- correct views of a small room. Reported, never failed on.
LOW_STRUCTURE_SPREAD_M = 0.35

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "pdf_walkthrough")


def main() -> int:
    page = extract(PDF, pages=[PAGE])["pages"][0]

    scale = MM_PER_PT / 1000.0
    rooms = {r["name"]: np.array(r["pos"]) * scale for r in rooms_for_page(page)["rooms"]}
    missing = [name for name in ROUTE if name not in rooms]
    if missing:
        print(f"route rooms not found on this page: {missing}")
        return 1

    blockout = blockout_from_page(page, MM_PER_PT, region_m=REGION_M)
    mesh = blockout["mesh"]
    if mesh is None:
        print(f"no blockout: {blockout.get('reason')}")
        return 1
    print(f"walls: {blockout['wall_polygons']} polygons from {blockout['stats']}")
    print(f"mesh: {len(mesh.faces)} faces, bounds {np.round(mesh.bounds, 2).tolist()}")
    print(f"ceiling {blockout['ceiling_height_m']} m ({blockout['height_confidence']} "
          f"-- the PDF states no height anywhere)")

    # Plan through free space rather than straight between room centres. Straight lines put
    # 47 of 97 frames inside a wall here, which is what this replaces.
    polygons, _ = wall_polygons(page, MM_PER_PT, REGION_M)
    route = plan_route(polygons, REGION_M, [rooms[name] for name in ROUTE])
    print(f"route: {len(route['path'])} points, minimum clearance "
          f"{route['min_clearance_m']:.2f} m (camera radius {route['camera_radius_m']} m)")
    if not route["ok"]:
        print("  FAIL: no route keeps the camera clear of the walls -- "
              "these rooms are not connected in the extracted geometry.")
        return 1
    # One waypoint per frame. Thinning the 369-point route to a dozen and joining those with
    # straight lines cuts exactly the corners the planner routed around, which put 25 frames
    # back inside walls even though the planned route never came within 0.40 m of one.
    # One waypoint per frame, used AS the path. waypoint_path() re-interpolates by arc length
    # with easing, which silently undoes the per-frame turn limit: measured, the worst frame
    # leaked to 41 deg/s through it against a clamp of 24. Since to_waypoints already emits
    # exactly FRAME_COUNT entries there is nothing left for it to interpolate.
    waypoints = to_waypoints(route, count=FRAME_COUNT, eye_height_m=EYE_HEIGHT_M)

    path = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in waypoints]
    print("camera:", describe(path))

    P = np.array([p[0] for p in path]); T = np.array([p[1] for p in path])
    travel = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    d = T - P
    head = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    step = np.abs((np.diff(head) + 180) % 360 - 180)
    print(f"speed {travel / (FRAME_COUNT / FPS):.2f} m/s (walkthrough wants 0.5-0.8) | "
          f"rotation {step.sum() / (FRAME_COUNT / FPS):.1f} deg/s (wants <15) | "
          f"worst frame {step.max() * FPS:.1f} deg/s")

    builder = Room3DBuilder({"room_name": "pdf", "polygon": [[0, 0], [1, 0], [1, 1]],
                             "ceiling_height_m": blockout["ceiling_height_m"],
                             "doors": [], "windows": []})
    builder.mesh = mesh

    frames = os.path.join(OUT, "frames")
    builder.render_camera_path(path, frames, width=480, height=832, fov_deg=70)

    grids = [np.load(os.path.join(frames, f"depth_{i:04d}.npy")) for i in range(FRAME_COUNT)]
    voids = [float((~np.isfinite(g)).mean()) for g in grids]
    escaped = [i for i, v in enumerate(voids) if v > 0.60]
    nearest = [float(g[np.isfinite(g)].min()) if np.isfinite(g).any() else 0.0 for g in grids]
    buried = [i for i, n in enumerate(nearest) if n < NEAR_CLIP_M]
    spreads = [float(g[np.isfinite(g)].std()) if np.isfinite(g).any() else 0.0 for g in grids]
    flat = [i for i, s in enumerate(spreads) if s < LOW_STRUCTURE_SPREAD_M]

    print(f"void per frame:  mean {np.mean(voids):.3f}  max {max(voids):.3f}")
    print(f"frames outside the model (void > 60%): {len(escaped)}{escaped[:8] if escaped else ' (none)'}")
    print(f"nearest surface: {min(nearest):.2f} m over the whole clip")
    print(f"frames with the camera inside geometry (< {NEAR_CLIP_M} m): "
          f"{len(buried)}{buried[:8] if buried else ' (none)'}")
    if buried:
        print("  FAIL: the route passes through geometry instead of through an opening.")
    print(f"advisory - low-structure frames (spread < {LOW_STRUCTURE_SPREAD_M} m): {len(flat)}"
          f" — a plain wall in view, not necessarily a fault")

    videos = os.path.join(OUT, "videos")
    os.makedirs(videos, exist_ok=True)
    for kind in ("depth", "edges", "clay"):
        target = os.path.join(videos, f"{kind}_pdf.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(frames, f"{kind}_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", target],
                       check=True)
        print("wrote", target)
    return 0 if not buried and not escaped else 1


if __name__ == "__main__":
    sys.exit(main())
