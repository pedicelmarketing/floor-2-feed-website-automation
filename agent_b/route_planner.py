"""
Find a camera route between rooms that goes through the doorways instead of through the walls.

Straight lines between room centres do not work, and the failure is not subtle: routing a
three-room walkthrough that way put 47 of 97 frames inside a wall. Nor is it a matter of
nudging waypoints by hand -- that was tried on the DWG path over several rounds, and the last
attempt still crossed a wall 0.25 m from a 0.30 m gap.

So plan it. Rasterise the walls, measure every free pixel's distance to the nearest wall, and
find the cheapest path where cost falls as clearance rises. The camera then hugs the middle of
each room and passes through the middle of each doorway, because the doorway centre is the
highest-clearance point of the only route between two rooms. No opening has to be located,
identified or paired: it is simply where the cheap path goes.

The clearance map does double duty as a guarantee. A route is rejected outright if any point
on it is closer to a wall than the camera's own radius, so "the camera never clips a wall" is
established before a single frame is rendered rather than measured afterwards.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Grid resolution in pixels per metre. 40 is 2.5 cm per pixel -- finer than the 1-4 cm the
# extracted geometry is accurate to, so the grid is not the limiting factor.
PIXELS_PER_M = 40.0
# A camera is not a point. Keeping this much clear of every wall stops the near plane from
# grazing a surface, which is what flattens a depth map even when the camera is technically
# in open space.
CAMERA_RADIUS_M = 0.30


def _rasterise(wall_polys: List[Any], bounds: Sequence[float],
               pixels_per_m: float) -> Tuple[np.ndarray, Tuple[float, float]]:
    from PIL import Image, ImageDraw

    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) * pixels_per_m) + 2
    height = int((maxy - miny) * pixels_per_m) + 2
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for polygon in wall_polys:
        for part in getattr(polygon, "geoms", [polygon]):
            if part.geom_type != "Polygon":
                continue
            ring = [(int((x - minx) * pixels_per_m), int((y - miny) * pixels_per_m))
                    for x, y in part.exterior.coords]
            if len(ring) >= 3:
                draw.polygon(ring, fill=1)
            for interior in part.interiors:
                hole = [(int((x - minx) * pixels_per_m), int((y - miny) * pixels_per_m))
                        for x, y in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)

    return np.array(canvas, dtype=bool), (minx, miny)


def plan_route(wall_polys: List[Any], bounds: Sequence[float],
               stops_m: Sequence[Sequence[float]],
               pixels_per_m: float = PIXELS_PER_M,
               camera_radius_m: float = CAMERA_RADIUS_M) -> Dict[str, Any]:
    """
    A route in metres visiting `stops_m` in order, staying clear of walls throughout.

    Returns {"path": [(x, y), ...], "min_clearance_m": float, "ok": bool}. `ok` is False when
    any point ends up closer to a wall than the camera radius, which happens when two rooms
    are genuinely not connected in the extracted geometry -- a fact about the drawing worth
    surfacing rather than a route worth rendering.
    """
    from scipy import ndimage
    from skimage.graph import route_through_array

    walls, (minx, miny) = _rasterise(wall_polys, bounds, pixels_per_m)
    clearance_px = ndimage.distance_transform_edt(~walls)
    clearance_m = clearance_px / pixels_per_m

    # Cheap where there is room, expensive near a wall. The +1e-3 keeps a wall pixel finite so
    # the search fails with a bad route it can report, rather than with an exception.
    cost = 1.0 / (clearance_m + 1e-3)

    def to_px(x, y):
        return (int(round((y - miny) * pixels_per_m)), int(round((x - minx) * pixels_per_m)))

    def to_m(r, c):
        return (c / pixels_per_m + minx, r / pixels_per_m + miny)

    full: List[Tuple[int, int]] = []
    for start, end in zip(stops_m[:-1], stops_m[1:]):
        a, b = to_px(*start[:2]), to_px(*end[:2])
        indices, _ = route_through_array(cost, a, b, fully_connected=True, geometric=True)
        full.extend(indices if not full else indices[1:])

    if not full:
        return {"path": [], "min_clearance_m": 0.0, "ok": False, "reason": "no route found"}

    clearances = [float(clearance_m[r, c]) for r, c in full]
    return {
        "path": [to_m(r, c) for r, c in full],
        "clearances_m": clearances,
        "min_clearance_m": float(min(clearances)),
        "ok": min(clearances) >= camera_radius_m,
        "camera_radius_m": camera_radius_m,
    }


def to_waypoints(route: Dict[str, Any], count: int, eye_height_m: float = 1.60,
                 look_ahead: int = 12) -> List[Tuple[List[float], List[float]]]:
    """
    Thin a dense route into camera waypoints, each looking at the route further along.

    Aiming at a point on the path rather than at the final destination is what keeps the view
    pointing through a doorway while passing through it. Aiming at the destination would have
    the camera stare at a wall for the whole approach and swing round only after arriving.
    """
    path = route["path"]
    if len(path) < 2:
        return []
    picks = np.linspace(0, len(path) - 1, max(2, count)).astype(int)
    waypoints = []
    for i in picks:
        here = path[i]
        ahead = path[min(i + look_ahead, len(path) - 1)]
        if ahead == here:
            ahead = path[-1]
        waypoints.append(([here[0], here[1], eye_height_m],
                          [ahead[0], ahead[1], eye_height_m - 0.08]))
    return waypoints
