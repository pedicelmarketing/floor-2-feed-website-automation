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

import math

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


# How far ahead on the route the camera looks, in metres. Far enough that the aim leads the
# turn through a doorway, near enough that it does not point at a wall behind the corner.
LOOK_AHEAD_M = 1.6
# Ceiling on how fast the view may swing, in degrees per frame. 1.5 at 16 fps is 24 deg/s,
# comfortably inside what reads as a steady architectural pan.
MAX_TURN_DEG_PER_FRAME = 1.5
# Positions are averaged over this many frames to take the staircase out of a grid-planned
# route. Kept small: heavy smoothing cuts corners, and cutting a corner means a wall.
POSITION_SMOOTH_FRAMES = 5
# How far the aim may be pulled off the direction of travel to look at something, in degrees.
# Beyond this the camera stops reading as a walkthrough and starts reading as a head swivel.
MAX_COMPOSE_OFFSET_DEG = 55.0
# Nothing further than this is worth turning the head for -- it is across the flat, and
# probably through a wall.
ATTRACTOR_RANGE_M = 6.0


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average that holds the endpoints, so the route still starts and ends where planned."""
    if window < 2 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    out = values.copy()
    for axis in range(values.shape[1]):
        padded = np.pad(values[:, axis], (window // 2, window // 2), mode="edge")
        out[:, axis] = np.convolve(padded, kernel, mode="valid")[:len(values)]
    out[0] = values[0]
    out[-1] = values[-1]
    return out


def _visible_attractors(camera: np.ndarray, travel_deg: float,
                        attractors: List[Dict[str, Any]], blocker,
                        range_m: float, max_offset_deg: float) -> List[Tuple[float, float]]:
    """
    Which objects the camera could actually look at from here: (heading_deg, weight).

    Three filters, and each removes a different way of framing nothing. Out of range means
    across the flat. Behind means the camera would have to walk backwards to see it. Blocked
    means there is a wall in between -- and without that test the camera turns to admire a bed
    it cannot see, which is worse than looking down the corridor.
    """
    from shapely.geometry import LineString

    out = []
    for item in attractors:
        target = np.asarray(item["point"], dtype=float)
        offset = target - camera
        distance = float(np.linalg.norm(offset))
        if distance < 0.4 or distance > range_m:
            continue
        heading = math.degrees(math.atan2(offset[1], offset[0]))
        if abs((heading - travel_deg + 180.0) % 360.0 - 180.0) > max_offset_deg:
            continue
        if blocker is not None and LineString([tuple(camera), tuple(target)]).intersects(blocker):
            continue
        # Bigger and nearer wins. Size matters because a wardrobe is worth framing and a
        # bedside table is not; 1/distance because the thing you are walking past dominates.
        out.append((heading, float(item.get("weight", 1.0)) / max(distance, 0.5)))
    return out


def to_waypoints(route: Dict[str, Any], count: int, eye_height_m: float = 1.60,
                 look_ahead_m: float = LOOK_AHEAD_M,
                 max_turn_deg: float = MAX_TURN_DEG_PER_FRAME,
                 position_smooth: int = POSITION_SMOOTH_FRAMES,
                 attractors: Optional[List[Dict[str, Any]]] = None,
                 blocker: Any = None,
                 compose_strength: float = 0.65,
                 max_offset_deg: float = MAX_COMPOSE_OFFSET_DEG,
                 attractor_range_m: float = ATTRACTOR_RANGE_M
                 ) -> List[Tuple[List[float], List[float]]]:
    """
    Turn a dense route into camera waypoints with a steady position and a steady aim.

    Aiming along the route rather than at the destination is what keeps the view pointing
    through a doorway while passing through it. But aiming at a single route point does not
    work, and the failure is severe rather than cosmetic: the route comes off a grid planner,
    so consecutive points sit on grid cells and the direction between them hops. Measured on
    the three-room walkthrough, the camera performed 637 degrees of rotation to achieve a turn
    of 120 -- 81% of all turning was jitter, and one frame swung 90 degrees on its own. Every
    one of those frames handed the video model a view that had lurched for no reason and asked
    it to invent whatever the lurch revealed, which is a large part of why the output looked
    synthetic.

    Three things fix it, in order of how much they contribute:

      the aim direction is taken from the average of the route over `look_ahead_m` ahead, so
      grid quantisation cancels instead of accumulating;

      the heading carries frame to frame and may change by at most `max_turn_deg`, so no
      single frame can swing;

      positions are lightly averaged, which removes the staircase without cutting corners.
      Light on purpose -- corner cutting puts the camera in a wall, and the clearance check in
      plan_route no longer applies once the points move.

    Callers should re-check clearance after smoothing rather than assuming the planned route's
    guarantee survives.
    """
    path = np.asarray(route["path"], dtype=float)
    if len(path) < 2:
        return []

    count = max(2, count)
    # Resample to one position per frame along the route, then take the staircase out.
    idx = np.linspace(0, len(path) - 1, count)
    positions = np.stack([np.interp(idx, np.arange(len(path)), path[:, axis])
                          for axis in range(2)], axis=1)
    positions = _smooth(positions, position_smooth)

    # Distance along the route, so "look 1.6 m ahead" means metres rather than array steps --
    # array steps are meaningless when the planner's spacing depends on grid resolution.
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(steps)])

    headings = []
    for i in range(count):
        ahead = travelled[i] + look_ahead_m
        window = positions[(travelled >= travelled[i]) & (travelled <= ahead)]
        if len(window) < 2:
            window = positions[i:] if i < count - 1 else positions[-2:]
        direction = window[-1] - window[0] if len(window) >= 2 else positions[-1] - positions[i]
        if np.linalg.norm(direction) < 1e-9:
            direction = positions[-1] - positions[i]
        travel_deg = math.degrees(math.atan2(direction[1], direction[0]))

        # COMPOSITION. Aiming purely along the direction of travel is what stopped the camera
        # walking into walls, and it is also why it walked straight past every piece of
        # furniture in the flat looking at bare plaster. The route is correct; the framing was
        # never considered. Blend the travel direction towards whatever is worth looking at.
        #
        # Blended as a weighted sum of unit vectors, not by averaging angles: averaging 179 and
        # -179 degrees gives 0, pointing the camera exactly backwards.
        seen = _visible_attractors(positions[i], travel_deg, attractors or [], blocker,
                                   attractor_range_m, max_offset_deg)
        if seen:
            vector = np.array([math.cos(math.radians(travel_deg)),
                               math.sin(math.radians(travel_deg))]) * (1.0 - compose_strength)
            total = sum(w for _, w in seen)
            for heading, weight in seen:
                vector += np.array([math.cos(math.radians(heading)),
                                    math.sin(math.radians(heading))]) * (compose_strength * weight / total)
            if np.linalg.norm(vector) > 1e-9:
                travel_deg = math.degrees(math.atan2(vector[1], vector[0]))
        headings.append(travel_deg)

    # Rate-limit the heading. Wrapping through +/-180 has to be handled explicitly or the
    # camera spins the long way round at the wrap point.
    limited = [headings[0]]
    for target in headings[1:]:
        delta = (target - limited[-1] + 180.0) % 360.0 - 180.0
        limited.append(limited[-1] + max(-max_turn_deg, min(max_turn_deg, delta)))

    waypoints = []
    for i in range(count):
        angle = math.radians(limited[i])
        aim = positions[i] + np.array([math.cos(angle), math.sin(angle)]) * look_ahead_m
        waypoints.append(([float(positions[i][0]), float(positions[i][1]), eye_height_m],
                          [float(aim[0]), float(aim[1]), eye_height_m - 0.08]))
    return waypoints
