"""
Draw the camera route on the floor plan, from above, before spending anything on video.

Why this exists: the tour camera once crossed a wall 0.25 m south of a 0.30 m doorway gap
and travelled through solid brick for five frames. Finding that took measuring depth
variance over 121 frames and then testing polygon containment point by point. On a top-down
plan it is a line visibly missing a gap -- obvious in the time it takes to look at it.

This is the cheap 80% of what a 3D tool's viewport would give: somewhere to SEE the route
against the geometry. It is not a renderer and does not replace the depth pass; it is the
check you run first, because it costs a second and catches the class of mistake that is
expensive to catch any other way.

    from plan_preview import plan_preview
    plan_preview(rooms, path, "preview.png")
"""
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# Matplotlib is only needed for this preview, never on the render path -- import lazily so
# the geometry pipeline keeps working (and CI keeps passing) without it installed.


def _openings(room: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """World-space start/end of every door and window on a room, tagged by kind."""
    P = np.asarray(room["polygon"], dtype=float)
    n = len(P)
    out = []
    for kind in ("doors", "windows"):
        for o in room.get(kind, []):
            i = o["wall_edge_index"]
            a, b = P[i], P[(i + 1) % n]
            length = np.linalg.norm(b - a)
            if length <= 0:
                continue
            u = (b - a) / length
            start = a + u * o["offset_along_wall_m"]
            out.append((start, start + u * o["width_m"], kind))
    return out


def walkable_region(rooms: List[Dict[str, Any]], jamb_reach_m: float = 0.45):
    """
    The area a camera may legitimately occupy: the rooms, plus the doorways between them.

    Room polygons do not touch -- adjoining rooms sit 0.10-0.25 m apart with the wall in
    between -- so a camera crossing a doorway is briefly inside NO room. Testing room
    containment alone therefore flags every correct doorway transit as a failure, which is
    useless. Each doorway is bridged here by a thin rectangle spanning the opening, so the
    only points left outside the region are the ones genuinely inside solid wall.

    A bridge is built only where TWO facing openings OVERLAP -- the intersection of the two
    rooms' door slots, not either one alone. Bridging each opening independently is wrong and
    silently hides the exact bug this was written for: the hall's doorway and the bedroom's
    doorway share only 0.30 m of their 0.80 m width, so a route can sit squarely inside the
    hall's opening while the bedroom behind it is still solid wall. Intersecting the two
    leaves precisely the slot that goes all the way through.

    `jamb_reach_m` is how far each opening's slot extends perpendicular to its wall; it must
    exceed the room-to-room gap (0.10-0.25 m here) for the two to meet at all.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    parts = [Polygon(r["polygon"]) for r in rooms]

    slots = []
    for room in rooms:
        for start, end, kind in _openings(room):
            if kind == "doors":
                slots.append(LineString([start, end]).buffer(jamb_reach_m, cap_style=2))

    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            shared = slots[i].intersection(slots[j])
            if not shared.is_empty and shared.area > 1e-6:
                parts.append(shared)

    return unary_union(parts)


def _clearance(point: np.ndarray, rooms: List[Dict[str, Any]]) -> float:
    """Distance from a point to the nearest wall of whichever room contains it.

    Returns 0.0 when the point is in no room -- used only for the informational readout,
    never for pass/fail, because a doorway transit legitimately scores 0.0 here.
    """
    from shapely.geometry import Point, Polygon

    pt = Point(float(point[0]), float(point[1]))
    for room in rooms:
        poly = Polygon(room["polygon"])
        if poly.contains(pt):
            return float(poly.exterior.distance(pt))
    return 0.0


def plan_preview(rooms: List[Dict[str, Any]],
                 path: Sequence[Tuple[Sequence[float], Sequence[float]]],
                 out_path: str,
                 look_every: int = 12) -> Dict[str, Any]:
    """
    Render the plan with the camera route overlaid. Returns a report dict.

    `path` is the list of (position, look_at) pairs the renderer will use, so this shows the
    exact route that will be rendered rather than an idealised version of it.

    A frame FAILS when it sits outside the walkable region -- inside solid wall. Frames are
    drawn red when they do. Clearance to the nearest room wall is reported alongside, but it
    is informational: it drops to zero during any correct doorway crossing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 11))

    for room in rooms:
        P = np.asarray(room["polygon"], dtype=float)
        ax.fill(P[:, 0], P[:, 1], facecolor="#E8EBEE", edgecolor="#131A22",
                linewidth=1.6, zorder=1)
        c = P.mean(axis=0)
        ax.text(c[0], c[1], room["room_name"][:18], ha="center", va="center",
                fontsize=8, color="#4B5768", zorder=4)

        for start, end, kind in _openings(room):
            # Draw openings over the wall so a doorway reads as a gap, not a line.
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color="#FFFFFF", linewidth=4.0, solid_capstyle="butt", zorder=2)
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color="#1E4E8C" if kind == "doors" else "#2A7A4F",
                    linewidth=1.8, solid_capstyle="butt", zorder=3)

    from shapely.geometry import Point

    positions = np.array([p[0][:2] for p in path], dtype=float)
    clearances = np.array([_clearance(p, rooms) for p in positions])

    region = walkable_region(rooms)
    ok = np.array([region.contains(Point(float(x), float(y))) for x, y in positions])
    buried = [int(i) for i in np.flatnonzero(~ok)]

    ax.plot(positions[:, 0], positions[:, 1], color="#93601F", linewidth=1.4,
            alpha=0.85, zorder=5)
    ax.scatter(positions[ok, 0], positions[ok, 1], s=9, c="#93601F", zorder=6)
    if (~ok).any():
        ax.scatter(positions[~ok, 0], positions[~ok, 1], s=34, c="#A03434",
                   marker="x", linewidths=1.6, zorder=7, label="inside solid wall")

    # Sight lines: without these the route reads as a path but not as a camera.
    for i in range(0, len(path), max(1, look_every)):
        p = np.asarray(path[i][0][:2], dtype=float)
        t = np.asarray(path[i][1][:2], dtype=float)
        d = t - p
        norm = np.linalg.norm(d)
        if norm > 1e-9:
            d = d / norm * 0.7
            ax.arrow(p[0], p[1], d[0], d[1], head_width=0.11, head_length=0.13,
                     fc="#4B5768", ec="#4B5768", alpha=0.75, zorder=8, length_includes_head=True)

    ax.scatter(*positions[0], s=70, marker="o", facecolor="#FFFFFF",
               edgecolor="#131A22", linewidth=1.5, zorder=9)
    ax.scatter(*positions[-1], s=90, marker="s", facecolor="#131A22",
               edgecolor="#131A22", zorder=9)

    ax.set_aspect("equal")
    ax.set_title(f"Camera route, {len(path)} frames"
                 + (f"  —  {len(buried)} FRAME(S) INSIDE SOLID WALL" if buried
                    else "  —  route stays walkable"),
                 fontsize=11, color="#A03434" if buried else "#2A7A4F")
    ax.set_xlabel("metres (world X)", fontsize=8)
    ax.set_ylabel("metres (world Y)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#D3D8DC", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    if buried:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "out_path": out_path,
        "frames": len(path),
        "median_clearance_m": round(float(np.median(clearances)), 3),
        "frames_in_wall": buried,
        "ok": not buried,
    }
