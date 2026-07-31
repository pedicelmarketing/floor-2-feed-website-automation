"""
Check that the render contains the doors and windows the drawing says are in shot.

This catches a failure the other two checks cannot see. The depth check compares relative
distance ordering; the edge check compares where lines fall. A render that invents a second
window on a blank wall, or omits a door, can satisfy both -- the wall is at the right depth
and its outline is in the right place, there is simply an extra hole in it. Counting
openings is the rule-based check for that, and it is the one thing here that needs vision.

Ground truth is per frame, not per apartment. "This flat has 5 doors" is not a claim any
single frame can be judged against -- what matters is how many openings should be visible
from THIS camera at THIS moment. That is computed here from the geometry: for each opening,
sample points across its width, and test whether any of them is inside the view frustum and
reachable from the camera without passing through something else first. An opening the
camera cannot see is not expected in the frame, and counting it would manufacture failures.

Occlusion is tested by ray casting against the same mesh used to render the control maps, so
"can the camera see it" is answered by the same geometry the model was conditioned on.

The counting side uses Gemini directly rather than a Comfy node: it bills the cheaper quota
and returns a parsed object instead of a job handle and a file download. Without a key the
check degrades to reporting expected counts only, and says so, rather than failing.
"""
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "gemini-2.5-flash"
# Samples across each opening's width. One centre point is too brittle -- a door half behind
# a wall would read as fully hidden -- and many is wasted ray casts on a 0.8 m opening.
SAMPLES_PER_OPENING = 5
# A render is not expected to be exact. Off by one on a partially-visible edge-of-frame
# opening is normal; off by two means something was invented or dropped.
COUNT_TOLERANCE = 1


def _opening_samples(room: Dict[str, Any], kind: str) -> List[Tuple[np.ndarray, str]]:
    """Sample points along each opening of `kind`, at mid-height, in world coordinates."""
    polygon = np.asarray(room["polygon"], dtype=float)
    n = len(polygon)
    out = []
    for opening in room.get(kind, []):
        i = opening["wall_edge_index"]
        a, b = polygon[i], polygon[(i + 1) % n]
        length = float(np.linalg.norm(b - a))
        if length <= 0:
            continue
        u = (b - a) / length
        sill = float(opening.get("sill_m", 0.0))
        head = float(opening.get("head_m", 2.1))
        z = (sill + head) / 2.0
        for t in np.linspace(0.15, 0.85, SAMPLES_PER_OPENING):
            p = a + u * (opening["offset_along_wall_m"] + opening["width_m"] * t)
            out.append((np.array([p[0], p[1], z]), kind))
    return out


def _in_frustum(point: np.ndarray, eye: np.ndarray, target: np.ndarray,
                fov_deg: float, aspect: float) -> bool:
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm <= 0:
        return False
    forward = forward / norm
    to_point = point - eye
    depth = float(np.dot(to_point, forward))
    if depth <= 1e-6:
        return False
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    rn = np.linalg.norm(right)
    if rn <= 1e-9:
        return False
    right = right / rn
    up = np.cross(right, forward)

    half_v = np.tan(np.radians(fov_deg) / 2.0)
    half_h = half_v * aspect
    return (abs(float(np.dot(to_point, right))) <= half_h * depth
            and abs(float(np.dot(to_point, up))) <= half_v * depth)


def expected_visible(rooms: List[Dict[str, Any]], mesh, eye: np.ndarray, target: np.ndarray,
                     fov_deg: float = 70.0, width: int = 480, height: int = 832) -> Dict[str, int]:
    """
    How many doors and windows the camera can actually see from this pose.

    An opening counts as visible when at least one sample point is inside the frustum and the
    ray from the camera reaches it without hitting geometry first.
    """
    aspect = width / height
    counts = {"doors": 0, "windows": 0}
    # One physical doorway exists TWICE in the scene graph: door_pairing mirrors each opening
    # onto the facing room so both walls are pierced. Counting records would then expect two
    # doors where a viewer sees one, and every frame would look like the render had dropped
    # openings. Collapse by world position first.
    seen_positions: List[Tuple[str, Tuple[int, int, int]]] = []

    for room in rooms:
        for kind in ("doors", "windows"):
            for opening_index, opening in enumerate(room.get(kind, [])):
                samples = _opening_samples({**room, kind: [opening]}, kind)
                if not samples:
                    continue
                centre = np.mean([p for p, _ in samples], axis=0)
                # 0.25 m buckets: the two records of one doorway sit 0.10-0.25 m apart
                # across the wall, while genuinely distinct openings are far further apart.
                key = (kind, tuple(np.round(centre / 0.25).astype(int)))
                if key in seen_positions:
                    continue
                seen_positions.append(key)

                visible = False
                for point, _ in samples:
                    if not _in_frustum(point, eye, target, fov_deg, aspect):
                        continue
                    direction = point - eye
                    distance = float(np.linalg.norm(direction))
                    if distance <= 1e-6:
                        continue
                    locations, _, _ = mesh.ray.intersects_location(
                        ray_origins=eye.reshape(1, 3),
                        ray_directions=(direction / distance).reshape(1, 3))
                    if len(locations) == 0:
                        visible = True
                        break
                    # Nearest hit beyond the opening means nothing blocks the line of sight.
                    nearest = float(np.min(np.linalg.norm(locations - eye, axis=1)))
                    if nearest >= distance - 0.05:
                        visible = True
                        break
                if visible:
                    counts[kind] += 1
    return counts


def count_in_frame(image_path: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Ask Gemini how many doorways and windows are visible. Degrades rather than raising."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"skipped": True, "reason": "GEMINI_API_KEY not set"}
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        with open(image_path, "rb") as f:
            data = f.read()
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/png"),
                "Count the architectural openings visible in this interior photograph. "
                "A doorway is an opening you could walk through, whether or not a door leaf "
                "is present. A window is a glazed opening to the outside. Do not count "
                "mirrors, pictures, alcoves, open shelving or cupboard fronts.",
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "doors": {"type": "INTEGER"},
                        "windows": {"type": "INTEGER"},
                        "note": {"type": "STRING"},
                    },
                    "required": ["doors", "windows"],
                },
            ),
        )
        import json

        return {"skipped": False, **json.loads(response.text)}
    except Exception as exc:                     # noqa: BLE001 - an outage must not read as FAIL
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


def check_frames(rooms: List[Dict[str, Any]], mesh,
                 camera_path: Sequence[Tuple[Sequence[float], Sequence[float]]],
                 render_dir: str, frame_indices: Sequence[int],
                 fov_deg: float = 70.0) -> Dict[str, Any]:
    """
    Compare expected against counted openings on a sample of frames.

    Sampled, not exhaustive: every frame is a vision API call, and consecutive frames of a
    slow camera move carry almost the same information.
    """
    results = []
    for i in frame_indices:
        eye = np.asarray(camera_path[i][0], dtype=float)
        target = np.asarray(camera_path[i][1], dtype=float)
        expected = expected_visible(rooms, mesh, eye, target, fov_deg)

        render_png = os.path.join(render_dir, f"result_{i:04d}.png")
        seen = count_in_frame(render_png) if os.path.exists(render_png) else {
            "skipped": True, "reason": "frame not found"}

        entry = {"frame": int(i), "expected": expected, "counted": seen}
        if not seen.get("skipped"):
            entry["door_delta"] = int(seen.get("doors", 0)) - expected["doors"]
            entry["window_delta"] = int(seen.get("windows", 0)) - expected["windows"]
            entry["within_tolerance"] = (abs(entry["door_delta"]) <= COUNT_TOLERANCE
                                         and abs(entry["window_delta"]) <= COUNT_TOLERANCE)
        results.append(entry)

    scored = [r for r in results if "within_tolerance" in r]
    return {
        "frames": results,
        "frames_scored": len(scored),
        "frames_skipped": len(results) - len(scored),
        "frames_off": [r["frame"] for r in scored if not r["within_tolerance"]],
        "tolerance": COUNT_TOLERANCE,
    }
