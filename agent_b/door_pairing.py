"""
Punch a shared doorway through BOTH rooms it connects.

Why this is needed: rooms are extracted independently, so a door recorded on room A's wall
cuts an opening in A only. Room B's wall on the other side stays solid, and they are
typically 0.1-0.25 m apart. The result looks fine from inside A -- there is a hole -- but a
camera moving through it hits B's unbroken wall a few centimetres later, so the two rooms
are visually connected and physically not.

That is invisible in a static shot facing a doorway, which is why it survived until a
walkthrough tried to actually travel between rooms.

This does not invent geometry: it only mirrors an opening the CAD file already recorded onto
the matching wall of the room immediately behind it. Rooms with no door within reach are
left alone and remain unreachable, which is a true statement about the source data rather
than something to paper over.
"""
import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np

# How close another room's wall must be to count as "the other side of this door".
# Observed real gaps between adjoining room polygons here are 0.10-0.25 m.
ADJACENCY_TOLERANCE_M = 0.35


def _door_world_midpoint(polygon: np.ndarray, door: Dict[str, Any]) -> np.ndarray:
    n = len(polygon)
    i = door["wall_edge_index"]
    a, b = polygon[i], polygon[(i + 1) % n]
    u = (b - a) / np.linalg.norm(b - a)
    return a + u * (door["offset_along_wall_m"] + door["width_m"] / 2)


# Two walls facing each other through a doorway must be near-parallel. Requiring this
# rejects a perpendicular wall that merely happens to be close.
PARALLEL_TOLERANCE = 0.9   # |cos(angle)| between the two wall directions


def _closest_edge(polygon: np.ndarray, point: np.ndarray,
                  parallel_to: np.ndarray = None) -> Tuple[int, float, float]:
    """
    Returns (edge_index, distance, offset_along_that_edge) for the nearest wall.

    When `parallel_to` (the source door's wall direction) is given, only walls roughly
    parallel to it are considered. Without that filter the search breaks at corners: a door
    sitting near the junction of two walls is almost equidistant from both, so the nearest
    edge can be the PERPENDICULAR one. Mirroring onto it cuts the opening through a wall
    facing 90 degrees the wrong way -- the rooms still look connected head-on, but a camera
    passing through meets a jamb and the whole view flattens to a featureless surface.
    Observed exactly this on the hall doorway before the filter existed.
    """
    best = (0, math.inf, 0.0)
    n = len(polygon)
    for j in range(n):
        a, b = polygon[j], polygon[(j + 1) % n]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            continue
        if parallel_to is not None:
            direction = ab / math.sqrt(denom)
            if abs(float(np.dot(direction, parallel_to))) < PARALLEL_TOLERANCE:
                continue
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        dist = float(np.linalg.norm(point - proj))
        if dist < best[1]:
            best = (j, dist, t * math.sqrt(denom))
    return best


def _edge_direction(polygon: np.ndarray, edge_index: int) -> np.ndarray:
    n = len(polygon)
    ab = polygon[(edge_index + 1) % n] - polygon[edge_index]
    return ab / np.linalg.norm(ab)


def pair_doors(rooms: List[Dict[str, Any]],
               tolerance_m: float = ADJACENCY_TOLERANCE_M) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (rooms_with_paired_doors, log_lines). Input is not mutated.

    An opening is mirrored onto another room only when that room's wall lies within
    `tolerance_m` of the door and does not already have an opening at that spot.
    """
    out = copy.deepcopy(rooms)
    polygons = [np.array(r["polygon"], dtype=float) for r in out]
    log: List[str] = []

    # Snapshot the originals first: mirrored copies must not themselves be mirrored back.
    originals = [(idx, copy.deepcopy(d)) for idx, r in enumerate(out) for d in r["doors"]]

    for src_idx, door in originals:
        mid = _door_world_midpoint(polygons[src_idx], door)
        src_dir = _edge_direction(polygons[src_idx], door["wall_edge_index"])
        src_name = out[src_idx]["room_name"][:24]
        matched = False

        for dst_idx, dst in enumerate(out):
            if dst_idx == src_idx:
                continue
            edge_idx, dist, offset = _closest_edge(polygons[dst_idx], mid, parallel_to=src_dir)
            if dist > tolerance_m:
                continue

            edge_len = float(np.linalg.norm(
                polygons[dst_idx][(edge_idx + 1) % len(polygons[dst_idx])] - polygons[dst_idx][edge_idx]))
            width = door["width_m"]
            start = max(0.0, min(edge_len - width, offset - width / 2))

            already = any(
                e["wall_edge_index"] == edge_idx and abs(e["offset_along_wall_m"] - start) < 0.25
                for e in dst["doors"] + dst["windows"])
            if already:
                log.append(f"  {src_name} -> {dst['room_name'][:24]}: opening already present")
                matched = True
                continue

            dst["doors"].append({
                "wall_edge_index": edge_idx,
                "offset_along_wall_m": round(start, 3),
                "wall_length_m": round(edge_len, 3),
                "width_m": width,
                "sill_m": door["sill_m"],
                "head_m": door["head_m"],
                "height_confidence": door.get("height_confidence", "assumed-default"),
                "mirrored_from": src_name,
            })
            log.append(f"  {src_name} -> {dst['room_name'][:24]}: opening mirrored "
                       f"(walls {dist:.2f} m apart)")
            matched = True

        if not matched:
            log.append(f"  {src_name} door at ({mid[0]:.2f},{mid[1]:.2f}): no room behind it "
                       f"-- exterior or unmodelled space, left as is")

    return out, log


def connectivity(rooms: List[Dict[str, Any]],
                 tolerance_m: float = ADJACENCY_TOLERANCE_M) -> Dict[str, List[str]]:
    """Which rooms can actually be walked between, given the doors present."""
    polygons = [np.array(r["polygon"], dtype=float) for r in rooms]
    graph: Dict[str, List[str]] = {r["room_name"][:24]: [] for r in rooms}
    for i, r in enumerate(rooms):
        for door in r["doors"]:
            mid = _door_world_midpoint(polygons[i], door)
            src_dir = _edge_direction(polygons[i], door["wall_edge_index"])
            for j, other in enumerate(rooms):
                if i == j:
                    continue
                _, dist, _ = _closest_edge(polygons[j], mid, parallel_to=src_dir)
                if dist <= tolerance_m:
                    a, b = r["room_name"][:24], other["room_name"][:24]
                    if b not in graph[a]:
                        graph[a].append(b)
                    if a not in graph[b]:
                        graph[b].append(a)
    return graph
