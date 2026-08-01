"""
Render the measured apartment as a REFERENCE clip, not as a per-frame control track.

Different job from render_pdf_walkthrough.py, hence a separate script rather than a flag.
That one produces the control a video model is steered by every frame, and is bound by the
generator's limits: 97 frames because Wan and Cosmos cap there, 480 or 720 wide because that is
what they consume. Nothing here is fed to a control branch, so none of those caps apply. This is
the clip a person -- or a tool like Google Flow, which takes a reference and not a control -- is
handed to see what the flat actually is.

So it is longer, larger, walks every room the extraction found rather than two, and emits a flat
colour pass whose only job is to say which surface is a window, a door, a wall or a floor.

Outputs, all at FPS:
  material_ground_truth.mp4  shaded render in plausible colour -- the primary reference
  clay_ground_truth.mp4      the same shading in grey, for models that expect a clay pass
  semantic_ground_truth.mp4  flat colour by material -- the legend is in MATERIAL_RGB
  depth_ground_truth.mp4     distance, one shared scale for the whole sequence
  edges_ground_truth.mp4     edge map from depth and normal discontinuities
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from camera_paths import describe                                    # noqa: E402
import importlib                                                     # noqa: E402
_b = importlib.import_module("3d_room_builder")                      # noqa: E402
Room3DBuilder = _b.Room3DBuilder
MATERIAL_TINT_LEGIBLE = _b.MATERIAL_TINT_LEGIBLE
MATERIAL_RGB = _b.MATERIAL_RGB
from pdf_blockout import blockout_from_page, wall_polygons            # noqa: E402
from pdf_vector import extract                                       # noqa: E402
from route_planner import plan_route, to_waypoints                   # noqa: E402
from wall_regions import rooms_for_page                              # noqa: E402

PDF = ("/home/openclaw/floor-2-feed-website-automation/uploads/"
       "12f909d4-7599-4940-aa09-79108a7625d8_floor-plans-estado-reformado.pdf")
PAGE = 0
MM_PER_PT = 36.1
REGION_M = (1.2, 15.0, 9.8, 27.0)

# Every room the extraction finds in this flat, not the two-room hop the generators could hold.
# Ordered so the walk reads as a viewing: in at the hall, through the living room, then the
# bedroom, then the bathroom.
ROUTE = ["HL", "GG", "D4", "A1"]
# 4n+1 is a Wan constraint and does not apply here, but it costs nothing to keep and means this
# clip can still be cut down and used as a control later without resampling.
FRAME_COUNT = 361
FPS = 12
# Full portrait HD. The control track is 720x1280 because that is what the generator consumes;
# a reference clip is only limited by ray-casting time.
WIDTH, HEIGHT = 1080, 1920
EYE_HEIGHT_M = 1.60
NEAR_CLIP_M = 0.30
# Fixed gain on the shaded pass. Without it this clip renders at a mean pixel value of 65-89 out
# of 255 -- legible on a monitor in a dark room and nowhere else. 1.6 puts the mean at 105-145
# and the 95th percentile around 160-224, which is a normally exposed picture. Constant, never
# per-frame: see the note in _save_clay_render for why auto-levelling each frame is wrong.
CLAY_EXPOSURE = 1.6

OUT = os.path.join(HERE, "output", "ground_truth")


def main() -> int:
    page = extract(PDF, pages=[PAGE])["pages"][0]
    scale = MM_PER_PT / 1000.0
    rooms = {r["name"]: np.array(r["pos"]) * scale for r in rooms_for_page(page)["rooms"]}

    available = [n for n in ROUTE if n in rooms]
    if len(available) < 2:
        print(f"need at least two rooms on the route; found {available} of {ROUTE}")
        print(f"rooms this page actually has: {sorted(rooms)}")
        return 1
    if available != ROUTE:
        print(f"NOTE: {[n for n in ROUTE if n not in rooms]} not on this page; "
              f"walking {available}")

    blockout = blockout_from_page(page, MM_PER_PT, region_m=REGION_M)
    mesh = blockout["mesh"]
    if mesh is None:
        print(f"no blockout: {blockout.get('reason')}")
        return 1

    materials = blockout.get("face_materials")
    counts = {m: int((materials == m).sum()) for m in sorted(set(materials.tolist()))} \
        if materials is not None else {}
    print(f"walls: {blockout['wall_polygons']} polygons | "
          f"furniture: {blockout.get('furniture_objects', 0)} objects | "
          f"{len(mesh.faces)} faces")
    print(f"ceiling {blockout['ceiling_height_m']} m ({blockout['height_confidence']})")
    print(f"windows: {blockout.get('window_polygons', 0)} openings, "
          f"sill {blockout.get('window_sill_m')} m to head {blockout.get('window_head_m')} m "
          f"(both ASSUMED -- the PDF states no vertical dimension)")
    print(f"faces by material: {counts}")
    if not counts.get("window"):
        print("  WARNING: no face labelled 'window'. The colour pass will show no windows.")

    polygons, _ = wall_polygons(page, MM_PER_PT, REGION_M)
    obstacles = polygons + list(blockout.get("furniture_footprints", []))
    route = plan_route(obstacles, REGION_M, [rooms[name] for name in available])
    print(f"route: {len(route['path'])} points, minimum clearance "
          f"{route['min_clearance_m']:.2f} m")
    if not route["ok"]:
        print("  FAIL: no route keeps the camera clear -- these rooms are not connected "
              "in the extracted geometry.")
        return 1

    from shapely.ops import unary_union

    attractors = [{"point": (f.centroid.x, f.centroid.y), "weight": f.area}
                  for f in blockout.get("furniture_footprints", [])]
    waypoints = to_waypoints(route, count=FRAME_COUNT, eye_height_m=EYE_HEIGHT_M,
                             attractors=attractors,
                             blocker=unary_union(polygons) if polygons else None)
    path = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in waypoints]
    print("camera:", describe(path))

    P = np.array([p[0] for p in path])
    travel = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    seconds = FRAME_COUNT / FPS
    print(f"{travel:.1f} m over {seconds:.1f} s = {travel / seconds:.2f} m/s "
          f"(a walkthrough wants 0.5-0.8)")

    builder = Room3DBuilder({"room_name": "pdf", "polygon": [[0, 0], [1, 0], [1, 1]],
                             "ceiling_height_m": blockout["ceiling_height_m"],
                             "doors": [], "windows": []})
    builder.mesh = mesh

    frames = os.path.join(OUT, "frames")
    builder.render_camera_path(path, frames, width=WIDTH, height=HEIGHT, fov_deg=70,
                               face_materials=materials,
                               tint_map=MATERIAL_TINT_LEGIBLE, semantic=True,
                               exposure=CLAY_EXPOSURE)

    grids = [np.load(os.path.join(frames, f"depth_{i:04d}.npy")) for i in range(FRAME_COUNT)]
    nearest = [float(g[np.isfinite(g)].min()) if np.isfinite(g).any() else 0.0 for g in grids]
    buried = [i for i, n in enumerate(nearest) if n < NEAR_CLIP_M]
    voids = [float((~np.isfinite(g)).mean()) for g in grids]
    escaped = [i for i, v in enumerate(voids) if v > 0.60]
    print(f"nearest surface {min(nearest):.2f} m | frames inside geometry: "
          f"{len(buried)} | frames outside the model: {len(escaped)}")
    if buried:
        print("  FAIL: the route passes through geometry instead of through an opening.")

    videos = os.path.join(OUT, "videos")
    os.makedirs(videos, exist_ok=True)
    for kind in ("material", "clay", "semantic", "depth", "edges"):
        if not os.path.exists(os.path.join(frames, f"{kind}_0000.png")):
            print(f"skipped {kind}: not rendered")
            continue
        target = os.path.join(videos, f"{kind}_ground_truth.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(frames, f"{kind}_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", target],
                       check=True)
        print("wrote", target)

    print("\ncolour key for the semantic pass:")
    for name, rgb in MATERIAL_RGB.items():
        print(f"  {name:<10} rgb{rgb}")
    return 0 if not buried and not escaped else 1


if __name__ == "__main__":
    sys.exit(main())
