"""
Camera paths for walkthrough rendering.

A single straight dolly (start -> end, fixed gaze) was enough to validate the pipeline, but
it is not a walkthrough: it never turns, so it never tests whether the generator holds a
room together while the view rotates. This builds paths from waypoints, with the camera
turning smoothly as it travels.

Two things that matter for output quality, both learned the hard way elsewhere in this
pipeline:

- Ease in and out rather than starting and stopping abruptly. Constant-velocity paths give
  the video model a hard discontinuity at each end, which reads as a jump cut.
- Interpolate the LOOK-AT TARGET, not a yaw angle. Interpolating angles invites the
  short-way/long-way ambiguity and gimbal awkwardness near vertical; moving the point being
  looked at keeps the turn well defined and always takes the sensible arc.
"""
import math
from typing import List, Sequence, Tuple

import numpy as np

Waypoint = Tuple[Sequence[float], Sequence[float]]   # (camera_position, look_at_target)


def _smoothstep(t: float) -> float:
    """3t^2 - 2t^3: zero velocity at both ends, so the move eases in and out."""
    return t * t * (3.0 - 2.0 * t)


def waypoint_path(waypoints: List[Waypoint], n_frames: int,
                  ease: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expands waypoints into n_frames (position, target) pairs.

    Frames are distributed by ARC LENGTH, not evenly per segment, so the camera moves at a
    steady speed regardless of how long each leg is. Splitting frames evenly per segment
    would make short legs crawl and long legs race.
    """
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints")
    if n_frames < 2:
        raise ValueError("need at least two frames")

    positions = np.array([w[0] for w in waypoints], dtype=float)
    targets = np.array([w[1] for w in waypoints], dtype=float)

    seg_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if seg_lengths.sum() <= 0:
        raise ValueError("waypoints describe zero travel")
    # Cumulative distance along the path, normalised to 0..1
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    cumulative /= cumulative[-1]

    path = []
    for i in range(n_frames):
        raw = i / (n_frames - 1)
        u = _smoothstep(raw) if ease else raw

        seg = int(np.searchsorted(cumulative, u, side="right") - 1)
        seg = max(0, min(seg, len(seg_lengths) - 1))
        span = cumulative[seg + 1] - cumulative[seg]
        local = 0.0 if span <= 0 else (u - cumulative[seg]) / span

        pos = positions[seg] + (positions[seg + 1] - positions[seg]) * local
        tgt = targets[seg] + (targets[seg + 1] - targets[seg]) * local
        path.append((pos, tgt))
    return path


def describe(path: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Human-readable summary -- total travel and how far the view rotates."""
    positions = np.array([p for p, _ in path])
    travel = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

    def heading(i):
        f = path[i][1] - path[i][0]
        return math.degrees(math.atan2(f[1], f[0]))

    turn = abs(((heading(-1) - heading(0)) + 180) % 360 - 180)
    return (f"{len(path)} frames, {travel:.1f} m travelled, "
            f"view rotates {turn:.0f} degrees")
